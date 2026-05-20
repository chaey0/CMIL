import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _to_percent(x):
    x = float(x)
    if 0.0 <= x <= 1.0:
        return x * 100.0
    return x


def _get_first_existing(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _normalize_scope(scope: str):
    scope = str(scope).lower().strip()

    if scope in {"local", "context", "masked", "task-local", "task_local"}:
        return "task"

    if scope in {"open", "opened", "global-open", "global_open", "all"}:
        return "global"

    if scope in {"task", "global", "both", "auto"}:
        return scope

    raise ValueError(
        f"Unknown scope: {scope}. "
        "Use one of ['global', 'task', 'both', 'auto']."
    )


def _infer_scope_from_path(path: str):
    name = Path(path).name.lower()

    if "task" in name or "local" in name:
        return "task"

    if "global" in name or "open" in name:
        return "global"

    return None


def _sort_and_deduplicate(sessions, names, accs):
    """
    If the same session appears multiple times, keep the last occurrence.
    """
    by_session = {}
    order = []

    for s, n, a in zip(sessions, names, accs):
        if s not in by_session:
            order.append(s)
        by_session[s] = (n, a)

    ordered_sessions = sorted(order)
    ordered_names = [by_session[s][0] for s in ordered_sessions]
    ordered_accs = [by_session[s][1] for s in ordered_sessions]

    return ordered_sessions, ordered_names, ordered_accs


def _empty_scope_store():
    return {
        "global": {"sessions": [], "names": [], "accs": []},
        "task": {"sessions": [], "names": [], "accs": []},
    }


def _append_scope_item(store, scope, session_idx, session_name, acc):
    scope = _normalize_scope(scope)
    if scope not in {"global", "task"}:
        raise ValueError(f"Internal error: invalid concrete scope={scope}")

    store[scope]["sessions"].append(int(session_idx))
    store[scope]["names"].append(str(session_name))
    store[scope]["accs"].append(_to_percent(acc))


def _finalize_scope_store(store):
    out = {}

    for scope in ["global", "task"]:
        sessions = store[scope]["sessions"]
        names = store[scope]["names"]
        accs = store[scope]["accs"]

        if len(sessions) == 0:
            continue

        out[scope] = _sort_and_deduplicate(sessions, names, accs)

    return out


# -----------------------------------------------------------------------------
# JSON parsing
# -----------------------------------------------------------------------------
def _get_session_items_from_json_root(data, json_path):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise TypeError(f"Unsupported JSON root type: {type(data)}")

    # current CIL cumulative JSON
    if isinstance(data.get("steps"), list):
        return data["steps"]

    # current joint / custom result format
    if isinstance(data.get("results"), list):
        return data["results"]

    # current joint_test format
    if isinstance(data.get("session_wise_masked_results"), list):
        return data["session_wise_masked_results"]

    # older format
    if isinstance(data.get("sessions"), list):
        return data["sessions"]

    raise ValueError(
        f"No session list found in JSON: {json_path}. "
        "Expected one of keys: steps, results, session_wise_masked_results, sessions."
    )


def _extract_json_metric_for_scope(item, requested_scope, metric="acc"):
    """
    Extract a metric value (acc / bacc / wf1) for the given scope.

    Supported item formats:
      1) flat scoped:  {"subtype_acc_global": 87.1, "subtype_bacc_global": 85.0}
      2) nested block: {"subtype_global": {"acc": 87.1, "bacc": 85.0}}
      3) cumulative:   {"eval_scope": "global", "subtype": {"acc": 87.1}}
      4) plain:        {"subtype_acc": 87.1}
    """
    requested_scope = _normalize_scope(requested_scope)
    m = str(metric).lower()  # acc | bacc | wf1

    if requested_scope in {"global", "task"}:
        # flat keys: subtype_acc_global, subtype_bacc_global, subtype_wf1_global
        for prefix in ["subtype", "fine"]:
            key = f"{prefix}_{m}_{requested_scope}"
            if key in item:
                return _to_percent(item[key])

        # nested block: subtype_global.{acc|bacc|wf1}
        for block_key in [f"subtype_{requested_scope}", f"fine_{requested_scope}"]:
            block = item.get(block_key)
            if isinstance(block, dict) and m in block:
                return _to_percent(block[m])

    item_scope = item.get("eval_scope", item.get("scope", None))
    if item_scope is not None:
        item_scope = _normalize_scope(item_scope)
        if requested_scope in {"global", "task"} and item_scope != requested_scope:
            return None

    subtype_block = item.get("subtype")
    if isinstance(subtype_block, dict) and m in subtype_block:
        return _to_percent(subtype_block[m])

    # plain fallback (acc only)
    if m == "acc":
        for key in ["subtype_acc", "fine_acc", "acc"]:
            if key in item:
                return _to_percent(item[key])

    return None


# keep old name as alias for backward compat
def _extract_json_acc_for_scope(item, requested_scope, fallback_scope=None):
    return _extract_json_metric_for_scope(item, requested_scope, metric="acc")


def load_session_accs_by_scope_from_json(json_path: str, metric: str = "acc"):
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = _get_session_items_from_json_root(data, json_path)

    store = _empty_scope_store()
    inferred_scope = _infer_scope_from_path(json_path)

    for i, item in enumerate(items):
        sess_idx = int(_get_first_existing(item, ["session_idx", "idx", "session"], i))
        sess_name = str(_get_first_existing(item, ["session_name", "name"], f"session_{sess_idx}"))

        for scope in ["global", "task"]:
            val = _extract_json_metric_for_scope(item, scope, metric=metric)
            if val is not None:
                _append_scope_item(store, scope, sess_idx, sess_name, val)

        has_any = any(
            len(store[scope]["sessions"]) > 0 and store[scope]["sessions"][-1] == sess_idx
            for scope in ["global", "task"]
        )

        if not has_any:
            val = _extract_json_metric_for_scope(item, inferred_scope or "global", metric=metric)
            if val is not None:
                _append_scope_item(store, inferred_scope or "global", sess_idx, sess_name, val)

    out = _finalize_scope_store(store)

    if not out:
        raise ValueError(f"No session '{metric}' data found in JSON: {json_path}")

    return out


# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------
def parse_session_accs_by_scope_from_log(log_text: str, mode: str = "cil"):
    """
    Parses both old and new formats.

    New train-time format:
      [End-of-Session Test] Session 1 (lung)
        Global-open ACC: 87.11%  [primary]
        Task-local ACC : 87.43%

    New test-time format:
      [A_1 | global] Session 1 (lung) | Subtype ACC=87.11% | ...
      [A_1 | task]   Session 1 (lung) | Subtype ACC=87.43% | ...

    Old format:
      [A_1] Session 1 (lung) | Subtype ACC=87.11% | ...
    """
    mode = mode.lower()
    store = _empty_scope_store()

    # ------------------------------------------------------------------
    # 1) New train-time multi-line block
    # ------------------------------------------------------------------
    block_pattern = re.compile(
        r"\[End-of-Session Test\]\s+Session\s+(\d+)\s+\((.*?)\)\s*"
        r"(?:\n|\r\n)"
        r"(?:.*?\n)*?"
        r"\s*Global-open\s+ACC\s*:\s*([\d.]+)%.*?"
        r"(?:\n|\r\n)"
        r"\s*Task-local\s+ACC\s*:\s*([\d.]+)%",
        re.MULTILINE,
    )

    for sess_idx, sess_name, global_acc, task_acc in block_pattern.findall(log_text):
        _append_scope_item(store, "global", sess_idx, sess_name, global_acc)
        _append_scope_item(store, "task", sess_idx, sess_name, task_acc)

    # Also support reversed order: Task-local first, Global-open second.
    block_pattern_reversed = re.compile(
        r"\[End-of-Session Test\]\s+Session\s+(\d+)\s+\((.*?)\)\s*"
        r"(?:\n|\r\n)"
        r"(?:.*?\n)*?"
        r"\s*Task-local\s+ACC\s*:\s*([\d.]+)%.*?"
        r"(?:\n|\r\n)"
        r"\s*Global-open\s+ACC\s*:\s*([\d.]+)%",
        re.MULTILINE,
    )

    for sess_idx, sess_name, task_acc, global_acc in block_pattern_reversed.findall(log_text):
        _append_scope_item(store, "global", sess_idx, sess_name, global_acc)
        _append_scope_item(store, "task", sess_idx, sess_name, task_acc)

    # ------------------------------------------------------------------
    # 2) New test-time scope-explicit format
    # ------------------------------------------------------------------
    scoped_test_pattern = re.compile(
        r"\[A_\d+\s*\|\s*(global|task|local|context|masked|opened|all)\]\s*"
        r"Session\s+(\d+)\s+\((.*?)\)\s*\|\s*"
        r"Subtype\s+ACC=([\d.]+)%",
        re.MULTILINE,
    )

    for scope, sess_idx, sess_name, acc in scoped_test_pattern.findall(log_text):
        scope = _normalize_scope(scope)
        if scope in {"global", "task"}:
            _append_scope_item(store, scope, sess_idx, sess_name, acc)

    # ------------------------------------------------------------------
    # 3) Joint / finetune / old formats
    # ------------------------------------------------------------------
    patterns = []

    if mode == "joint":
        patterns.append(
            re.compile(
                r"\[JOINT TEST.*?\]\s+Session\s+(\d+)\s+\((.*?)\)\s*->\s*.*?"
                r"Subtype\s*\[ACC=([\d.]+)%",
                re.MULTILINE,
            )
        )

    elif mode == "finetune":
        patterns.append(
            re.compile(
                r"\[FINETUNE TEST.*?\]\s+Session\s+(\d+)\s+\((.*?)\)\s*->\s*.*?"
                r"Subtype\s*\[ACC=([\d.]+)%",
                re.MULTILINE,
            )
        )

    else:
        # old CIL cumulative test format
        patterns.append(
            re.compile(
                r"\[A_\d+\]\s+Session\s+(\d+)\s+\((.*?)\)\s*\|\s*"
                r"Subtype\s+ACC=([\d.]+)%",
                re.MULTILINE,
            )
        )

        # old train-time one-line format
        patterns.append(
            re.compile(
                r"\[End-of-Session Test\]\s+Session\s+(\d+)\s+\((.*?)\)\s*\|\s*"
                r"Subtype\s+ACC=([\d.]+)%",
                re.MULTILINE,
            )
        )

        # older generic format
        patterns.append(
            re.compile(
                r"^Session\s+(\d+)\s+\((.*?)\)\s*->\s*.*?"
                r"Subtype\s*\[ACC=([\d.]+)%",
                re.MULTILINE,
            )
        )

    # Old/plain formats are treated as global-open by default.
    for pattern in patterns:
        for sess_idx, sess_name, acc in pattern.findall(log_text):
            _append_scope_item(store, "global", sess_idx, sess_name, acc)

    out = _finalize_scope_store(store)

    if not out:
        raise ValueError(
            f"No session accuracy lines found for mode='{mode}'. "
            "Use JSON output if the log format is different."
        )

    return out


# -----------------------------------------------------------------------------
# Unified loader
# -----------------------------------------------------------------------------
def load_session_accs_by_scope(path: str, mode: str, metric: str = "acc"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() in {".pth", ".pt", ".ckpt"}:
        raise ValueError(
            f"{path} is a checkpoint file. "
            "Use a metrics JSON or evaluation log for plot.py."
        )

    if p.suffix.lower() == ".json":
        return load_session_accs_by_scope_from_json(str(p), metric=metric)

    return parse_session_accs_by_scope_from_log(
        p.read_text(encoding="utf-8"),
        mode=mode,
    )


def select_scopes(scope_data, requested_scope):
    requested_scope = _normalize_scope(requested_scope)

    if requested_scope == "both":
        selected = {}
        for scope in ["global", "task"]:
            if scope in scope_data:
                selected[scope] = scope_data[scope]
        return selected

    if requested_scope == "auto":
        if "global" in scope_data:
            return {"global": scope_data["global"]}
        if "task" in scope_data:
            return {"task": scope_data["task"]}
        return {}

    if requested_scope in scope_data:
        return {requested_scope: scope_data[requested_scope]}

    return {}


def _validate_extra_methods(methods, method_jsons):
    """
    Validate arbitrary method-name / metrics-json pairs.

    Example:
      --methods DER++ LwF EWC \
      --method_jsons derpp.json lwf.json ewc.json
    """
    methods = [] if methods is None else list(methods)
    method_jsons = [] if method_jsons is None else list(method_jsons)

    methods = [str(x).strip() for x in methods if str(x).strip()]
    method_jsons = [str(x).strip() for x in method_jsons if str(x).strip()]

    if len(methods) == 0 and len(method_jsons) == 0:
        return []

    if len(methods) != len(method_jsons):
        raise ValueError(
            "--methods and --method_jsons must have the same number of values.\n"
            f"Got {len(methods)} method names: {methods}\n"
            f"Got {len(method_jsons)} json paths: {method_jsons}"
        )

    pairs = []
    seen = set()
    for method_name, method_json in zip(methods, method_jsons):
        if method_name in seen:
            raise ValueError(f"Duplicate method name in --methods: {method_name}")
        seen.add(method_name)
        pairs.append((method_name, method_json))

    return pairs


def build_input_specs(args):
    """
    Build plot input specs.

    Built-in arguments keep backward compatibility:
      --cil, --joint, --finetune

    Arbitrary additional methods:
      --methods DER++ LwF EWC \
      --method_jsons derpp_metrics_global.json lwf_metrics_global.json ewc_metrics_global.json
    """
    inputs = [
        ("CIL", args.cil, "cil"),
        ("JOINT", args.joint, "joint"),
        ("FINETUNE", args.finetune, "finetune"),
    ]

    for method_name, method_json in _validate_extra_methods(args.methods, args.method_jsons):
        # JSON parsing is format-driven, so a generic mode is enough here.
        # If a non-JSON log is accidentally passed, mode='cil' gives the
        # most general cumulative-session parser.
        inputs.append((method_name, method_json, "cil"))

    return [(label, path, mode) for label, path, mode in inputs if path is not None]


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_curves(
    curves,
    save_path=None,
    title=None,
    ylim_min=0.0,
    ylim_max=100.0,
    show_values=False,
    xlabel="Organ Session",
    ylabel="Subtype Accuracy (%)",
    metric="acc",
):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    reference_sessions = None
    reference_names = None

    for label, sessions, names, accs in curves:
        ax.plot(sessions, accs, marker="o", linewidth=2, label=label)

        if show_values:
            for s, a in zip(sessions, accs):
                ax.annotate(
                    f"{a:.1f}",
                    xy=(s, a),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        if reference_sessions is None:
            reference_sessions = sessions
            reference_names = names

    if reference_sessions is not None:
        ax.set_xticks(reference_sessions)
        if len(reference_sessions) == len(reference_names):
            tick_labels = [f"S{s}" for s in reference_sessions]
            ax.set_xticklabels(tick_labels, fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Session-wise Subtyping Accuracy")
    ax.set_ylim(float(ylim_min), float(ylim_max))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        if save_path.suffix == "":
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "subtype_acc_scope_comparison.png"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot session-wise subtyping accuracy from JSON or logs. "
            "Supports global-open/task-local scopes and arbitrary extra methods via --methods/--method_jsons."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--cil", type=str, default=None, help="Path to CIL metrics JSON or CIL log")
    parser.add_argument("--joint", type=str, default=None, help="Path to JOINT metrics JSON or JOINT log")
    parser.add_argument("--finetune", type=str, default=None, help="Path to FINETUNE metrics JSON or FINETUNE log")

    # Additional arbitrary methods, e.g. DER++, LwF, EWC, BiC, etc.
    # The number/order of --methods and --method_jsons must match.
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[],
        help=(
            "Display names for additional methods. Example: "
            "--methods DER++ LwF EWC"
        ),
    )
    parser.add_argument(
        "--method_jsons",
        nargs="*",
        default=[],
        help=(
            "Metrics JSON/log paths corresponding to --methods. Example: "
            "--method_jsons derpp.json lwf.json ewc.json"
        ),
    )

    parser.add_argument(
        "--scope",
        type=str,
        default="global",
        choices=["global", "task", "both", "auto"],
        help=(
            "Which evaluation scope to plot. "
            "'global' is the primary cumulative open-set metric. "
            "'task' is organ/task-local masked evaluation. "
            "'both' plots both if available."
        ),
    )

    parser.add_argument("--metric", type=str, default="acc",
                        choices=["acc", "bacc", "wf1"],
                        help="Metric to plot: acc | bacc | wf1 (default: acc)")
    parser.add_argument("--save_path", type=str, default="./plots")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--ylim_min", type=float, default=0.0)
    parser.add_argument("--ylim_max", type=float, default=100.0)
    parser.add_argument("--show_values", action="store_true", default=True)

    args = parser.parse_args()

    inputs = build_input_specs(args)
    if not inputs:
        raise ValueError(
            "At least one input must be provided. Use --cil, --joint, --finetune, "
            "or additional methods with --methods DER++ LwF --method_jsons derpp.json lwf.json."
        )

    curves = []

    metric = args.metric

    for method_label, path, mode in inputs:

        scope_data = load_session_accs_by_scope(path, mode=mode, metric=metric)
        selected = select_scopes(scope_data, args.scope)

        if not selected:
            available = ", ".join(sorted(scope_data.keys()))
            print(
                f"[Skip] {method_label}: requested scope='{args.scope}' not found. "
                f"Available scopes: {available}"
            )
            continue

        for scope, (sessions, names, accs) in selected.items():
            if args.scope == "both":
                curve_label = f"{method_label}-{scope}"
            else:
                curve_label = method_label

            curves.append((curve_label, sessions, names, accs))

    if not curves:
        raise ValueError("No curves to plot. Check --scope and input files.")

    for label, sessions, names, accs in curves:
        print(f"[{label}]")
        for s, n, a in zip(sessions, names, accs):
            print(f"  Session {s} ({n}): {a:.2f}%")

    _metric_label = {"acc": "Accuracy", "bacc": "Balanced Accuracy", "wf1": "Weighted F1"}
    _ylabel_map = {"acc": "Subtype Accuracy (%)", "bacc": "Balanced Accuracy (%)", "wf1": "Weighted F1"}

    title = args.title
    if title is None:
        m_str = _metric_label.get(metric, metric.upper())
        if args.scope == "global":
            title = f"Global-open Cumulative Subtyping {m_str}"
        elif args.scope == "task":
            title = f"Task-local Macro Subtyping {m_str}"
        elif args.scope == "both":
            title = f"Global-open vs Task-local Subtyping {m_str}"
        else:
            title = f"Session-wise Subtyping {m_str}"

    plot_curves(
        curves=curves,
        save_path=args.save_path,
        title=title,
        ylim_min=args.ylim_min,
        ylim_max=args.ylim_max,
        show_values=args.show_values,
        ylabel=_ylabel_map.get(metric, metric.upper()),
        metric=metric,
    )


if __name__ == "__main__":
    main()