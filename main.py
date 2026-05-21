import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from data.dataset import SlideCSVDataset, collate_fn
from data.session_manager import SessionManager
from engine.continual import (
    ReplayBuffer,
    configure_session_training_stage,
    evaluate,
    extract_base_subspace,
    print_eval_details,
    register_sessions_until,
    setup_session_eval,
    setup_session_model,
    train_session,
    update_replay_buffer,
)
from models.model import KnowledgeTreeCLModel
from utils.helpers import (
    create_optimizer,
    create_scheduler,
    ensure_dir,
    find_session_ckpt_path,
    prepare_model_structure_for_ckpt_load,
    resolve_ckpt_dir,
    save_checkpoint,
    save_eval_confusion_matrices,
    set_seed,
)
from utils.taxonomy_parser import TaxonomyParser


def _none_if_blank(x):
    if x is None:
        return None
    x = str(x).strip()
    return None if x == "" or x.lower() in {"none", "null", "-"} else x


def load_ckpt_and_get_session_idx(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        return ckpt, None
    for k in ["model_state_dict", "state_dict", "model", "model_state"]:
        if k in ckpt:
            return ckpt[k], ckpt.get("session_idx", None)
    return ckpt, ckpt.get("session_idx", None)


def make_loader(csv_path, taxonomy, fine_labels, batch_size, shuffle, split=None, split_col="split", dataset_filter=None, num_workers=0, pin_memory=False):
    ds = SlideCSVDataset(
        csv_path=csv_path,
        taxonomy=taxonomy,
        opened_fine_labels=fine_labels,
        split=split,
        split_col=split_col,
        dataset_filter=dataset_filter,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def setup_device_from_arg(gpu_arg):
    gpu_arg = _none_if_blank(gpu_arg) or "0"
    if str(gpu_arg).lower() in {"cpu", "none", "null", "-1"}:
        print("[Device] use CPU")
        return torch.device("cpu")
    if "," in str(gpu_arg):
        raise ValueError("Single-GPU only. Use one GPU id, e.g. --gpu 0.")
    if not torch.cuda.is_available():
        print("[Device] CUDA is not available. Use CPU.")
        return torch.device("cpu")
    gpu_id = int(gpu_arg)
    torch.cuda.set_device(gpu_id)
    print(f"[Device] use cuda:{gpu_id}")
    return torch.device(f"cuda:{gpu_id}")


# -----------------------------------------------------------------------------
# text refinement
# -----------------------------------------------------------------------------
def run_text_refinement_pre_stage(args, session_idx=None, prev_refined_anchor_path=None):
    if not args.use_text_refinement:
        return None

    base = Path(args.text_refine_out_dir or Path(args.out_dir) / "text_refinement")
    out_dir = base / ("global" if session_idx is None else f"session_{session_idx:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "final_refined_text_anchors.pt"

    cmd = [
        sys.executable,
        args.text_refinement_script,
        "--config", args.config,
        "--taxonomy_json", args.taxonomy_json,
        "--out_dir", str(out_dir),
    ]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    if session_idx is not None:
        cmd += ["--session_idx", str(session_idx)]
    if prev_refined_anchor_path is not None:
        cmd += ["--prev_refined_anchor_path", str(prev_refined_anchor_path), "--freeze_old_anchors", "--refine_new_only"]

    print("[TextRefine]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not artifact.exists():
        raise FileNotFoundError(f"Text refinement artifact not found: {artifact}")
    return str(artifact)


# -----------------------------------------------------------------------------
# settings / printing
# -----------------------------------------------------------------------------
def resolve_eval_settings(args, cfg):
    eval_cfg = cfg.get("eval", {})
    scope = getattr(args, "scope", None) or eval_cfg.get("scope", "global")
    scope = str(scope).lower()
    if scope not in {"global", "task", "both"}:
        raise ValueError("--scope must be one of: global, task, both")
    scopes = ["global", "task"] if scope == "both" else [scope]
    return scopes, scope


def get_used_labels_from_sessions(cfg, taxonomy):
    fine, seen = [], set()
    for s in cfg.get("sessions", []):
        for lab in s["fine_labels"]:
            if lab not in seen:
                fine.append(lab)
                seen.add(lab)
    coarse = []
    for lab in fine:
        c = taxonomy.fine_to_coarse[lab]
        if c not in coarse:
            coarse.append(c)
    return fine, coarse


def print_session_info(title, session_idx=None, session_name=None, **items):
    print("\n" + "=" * 80)
    head = f"[{title}]" if session_idx is None else f"[{title}] Session {session_idx}: {session_name}"
    print(head)
    for k, v in items.items():
        if v is not None:
            print(f"{k} ({len(v)}): {v}")
    print("=" * 80)


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------
def _pct(x):
    return None if x is None else float(x) * 100.0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def _fmt_pct(x):
    return "N/A" if x is None else f"{x:.2f}%"

def _fmt_score(x):
    return "N/A" if x is None else f"{x:.4f}"

def compute_metrics_from_details(details):
    y_true = details["fine_true"].detach().cpu().numpy()
    y_pred = details["fine_pred"].detach().cpu().numpy()
    if len(y_true) == 0:
        return {"subtype_acc": 0.0, "subtype_wf1": 0.0, "subtype_bacc": 0.0}
    return {
        "subtype_acc": 100.0 * float((y_true == y_pred).mean()),
        "subtype_wf1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "subtype_bacc": _pct(balanced_accuracy_score(y_true, y_pred)),
    }


def _task_key(organ, task):
    task = str(task)
    return str(organ) if task in {"", "None", "none", "null", "default"} else f"{organ}/{task}"


def compute_dataset_breakdown_from_details(details):
    y_true = details["fine_true"].detach().cpu()
    y_pred = details["fine_pred"].detach().cpu()
    datasets = details.get("context_datasets", [])
    if not datasets or len(datasets) != len(y_true):
        return {}

    correct, count = {}, {}
    for i, ds in enumerate(datasets):
        correct[ds] = correct.get(ds, 0) + int(y_true[i].item() == y_pred[i].item())
        count[ds] = count.get(ds, 0) + 1

    return {
        "per_dataset_acc": {ds: 100.0 * correct[ds] / count[ds] for ds in sorted(count)},
        "per_dataset_count": {ds: count[ds] for ds in sorted(count)},
    }


def compute_task_breakdown_from_details(details):
    y_true = details["fine_true"].detach().cpu()
    y_pred = details["fine_pred"].detach().cpu()
    organs = details.get("context_organs", [])
    tasks = details.get("context_tasks", [])
    if len(organs) != len(y_true) or len(tasks) != len(y_true):
        return {"task_macro_acc": None, "task_pooled_acc": None, "per_task_acc": {}, "per_task_count": {}, "opened_tasks": []}

    correct, count = {}, {}
    for i, (o, t) in enumerate(zip(organs, tasks)):
        k = _task_key(o, t)
        correct[k] = correct.get(k, 0) + int(y_true[i].item() == y_pred[i].item())
        count[k] = count.get(k, 0) + 1

    per_task_acc = {k: 100.0 * correct[k] / count[k] for k in sorted(count)}
    return {
        "task_macro_acc": _mean(per_task_acc.values()),
        "task_pooled_acc": 100.0 * sum(correct.values()) / sum(count.values()) if count else None,
        "per_task_acc": per_task_acc,
        "per_task_count": count,
        "opened_tasks": sorted(count),
    }


@torch.no_grad()
def evaluate_cumulative_open_set(
    model,
    session_manager,
    session_idx,
    taxonomy,
    data_csv,
    batch_size,
    device,
    test_split=None,
    split_col="split",
    eval_scope="task",
    return_details=False,
    dataset_filter=None,
    num_workers=0,
    pin_memory=False,
):
    info = setup_session_eval(model, session_manager, session_idx, device, getattr(model, "cfg", {}))
    loader = make_loader(data_csv, taxonomy, info["opened_fine"], batch_size, False, split=test_split, split_col=split_col, dataset_filter=dataset_filter, num_workers=num_workers, pin_memory=pin_memory)
    _, _, details = evaluate(model, loader, device, return_details=True, eval_scope=eval_scope)

    metrics = compute_metrics_from_details(details)
    metrics.update({
        "eval_scope": eval_scope,
        "num_opened_fine": len(info["opened_fine"]),
        "num_opened_coarse": len(info["opened_coarse"]),
    })
    if eval_scope == "task":
        metrics.update(compute_task_breakdown_from_details(details))
    metrics.update(compute_dataset_breakdown_from_details(details))
    if return_details:
        metrics["details"] = details
        metrics["opened_fine_labels"] = list(info["opened_fine"])
        metrics["opened_coarse_labels"] = list(info["opened_coarse"])
    return metrics


def build_cumulative_result_row(session_idx, session_name, ckpt_path, metrics_by_scope, scope_mode):
    row = {"session_idx": int(session_idx), "session_name": session_name, "ckpt_path": ckpt_path, "scope": scope_mode}
    for scope, m in metrics_by_scope.items():
        row["num_opened_coarse"] = m["num_opened_coarse"]
        row["num_opened_fine"] = m["num_opened_fine"]
        if scope == "global":
            row.update({
                "subtype_acc_global": m["subtype_acc"],
                "subtype_wf1_global": m["subtype_wf1"],
                "subtype_bacc_global": m["subtype_bacc"],
            })
        else:
            row.update({
                "subtype_acc_task": m.get("task_macro_acc", m["subtype_acc"]),
                "subtype_wf1_task": m["subtype_wf1"],
                "subtype_bacc_task": m["subtype_bacc"],
                "subtype_acc_task_pooled": m["subtype_acc"],
                "per_task_acc": m.get("per_task_acc", {}),
                "per_task_count": m.get("per_task_count", {}),
                "opened_tasks": m.get("opened_tasks", []),
            })
        if m.get("per_dataset_acc"):
            row["per_dataset_acc"] = m["per_dataset_acc"]
            row["per_dataset_count"] = m["per_dataset_count"]
    return row


def summarize_cumulative_results(results, scope_mode):
    """Print and build summary for given scope_mode (global, task, or both)."""
    active_scopes = ["global", "task"] if scope_mode == "both" else [scope_mode]
    summaries = {}

    for sc in active_scopes:
        k_acc = f"subtype_acc_{sc}"
        k_wf1 = f"subtype_wf1_{sc}"
        k_bacc = f"subtype_bacc_{sc}"
        print("\n" + "-" * 80)
        print(f"[Cumulative Results | scope={sc}]")
        for r in results:
            print(
                f"A_{r['session_idx']} | {r['session_name']:10s} | "
                f"ACC={_fmt_pct(r.get(k_acc))}  "
                f"wF1={_fmt_score(r.get(k_wf1))}  "
                f"bACC={_fmt_pct(r.get(k_bacc))}"
            )
        final = results[-1] if results else {}
        s = {
            "scope": sc,
            "average_incremental": {
                "acc": _mean([r.get(k_acc) for r in results]),
                "wf1": _mean([r.get(k_wf1) for r in results]),
                "bacc": _mean([r.get(k_bacc) for r in results]),
            },
            "final_cumulative": {
                "acc": final.get(k_acc),
                "wf1": final.get(k_wf1),
                "bacc": final.get(k_bacc),
                "session_idx": final.get("session_idx"),
                "session_name": final.get("session_name"),
            },
        }
        ai = s["average_incremental"]
        fc = s["final_cumulative"]
        print(
            f"[Summary]  Avg  {sc}: "
            f"ACC={_fmt_pct(ai['acc'])}  "
            f"wF1={_fmt_score(ai['wf1'])}  "
            f"bACC={_fmt_pct(ai['bacc'])}"
        )
        print(
            f"           Final {sc}: "
            f"ACC={_fmt_pct(fc['acc'])}  "
            f"wF1={_fmt_score(fc['wf1'])}  "
            f"bACC={_fmt_pct(fc['bacc'])}"
        )

        print("-" * 80)
        summaries[sc] = s

    return summaries


def build_payload(results, summary, mode, scope):
    steps = []
    for r in results:
        step = {
            "session_idx": r["session_idx"],
            "session_name": r["session_name"],
            "ckpt_path": r.get("ckpt_path"),
            "num_opened_fine": r.get("num_opened_fine", 0),
            "num_opened_coarse": r.get("num_opened_coarse", 0),
            "scope": scope,
        }
        if "subtype_acc_global" in r:
            step["subtype_global"] = {"acc": r["subtype_acc_global"], "wf1": r["subtype_wf1_global"], "bacc": r["subtype_bacc_global"]}
        if "subtype_acc_task" in r:
            step["subtype_task"] = {
                "acc": r["subtype_acc_task"],
                "wf1": r["subtype_wf1_task"],
                "bacc": r["subtype_bacc_task"],
                "pooled_acc": r.get("subtype_acc_task_pooled"),
            }
            step["per_task_acc"] = r.get("per_task_acc", {})
            step["per_task_count"] = r.get("per_task_count", {})
            step["opened_tasks"] = r.get("opened_tasks", [])
        step["per_dataset_acc"] = r.get("per_dataset_acc", {})
        step["per_dataset_count"] = r.get("per_dataset_count", {})
        steps.append(step)
    return {
        "mode": mode,
        "scope": scope,
        "metric_definition": {
            "global_open": "all opened fine labels compete in one global logit vector",
            "task_local": "known organ-task mask applied; task ACC = macro avg over opened tasks",
        },
        "steps": steps,
        "summary": summary,
    }


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Saved JSON] {path}")


def print_end_session_eval(row, scope_mode):
    print("\n" + "-" * 80)
    print(f"[End-of-Session Test] Session {row['session_idx']} ({row['session_name']})")
    if "subtype_acc_global" in row:
        print(f"  Global-open ACC     : {row['subtype_acc_global']:.2f}%")
    if "subtype_acc_task" in row:
        print(f"  Task-local ACC      : {row['subtype_acc_task']:.2f}%")
        print(f"  Task-pooled ACC     : {row.get('subtype_acc_task_pooled', 0.0):.2f}%")
        for k, acc in row.get("per_task_acc", {}).items():
            print(f"    {k:32s}: {acc:.2f}%  n={row.get('per_task_count', {}).get(k, 0)}")
    print("-" * 80)


# -----------------------------------------------------------------------------
# eval model
# -----------------------------------------------------------------------------
def build_eval_model_for_session_ckpt(ckpt_path, taxonomy, cfg, session_manager, fallback_session_idx, device):
    state, loaded_idx = load_ckpt_and_get_session_idx(ckpt_path, device)
    session_idx = fallback_session_idx if loaded_idx is None else int(loaded_idx)
    model = KnowledgeTreeCLModel(taxonomy, cfg).to(device)
    prepare_model_structure_for_ckpt_load(model, session_manager, session_idx, device, cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Load warning] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load warning] unexpected keys: {len(unexpected)}")
    model.eval_model_session_idx = session_idx
    model.eval()
    return model, session_idx


def evaluate_model_for_scopes(model, session_manager, session_idx, taxonomy, data_csv, batch_size, device, test_split, split_col, eval_scopes, return_details=False, dataset_filter=None, num_workers=0, pin_memory=False):
    return {
        scope: evaluate_cumulative_open_set(
            model,
            session_manager,
            session_idx,
            taxonomy,
            data_csv,
            batch_size,
            device,
            test_split=test_split,
            split_col=split_col,
            eval_scope=scope,
            return_details=return_details,
            dataset_filter=dataset_filter,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for scope in eval_scopes
    }


def evaluate_joint_model_for_scopes(
    model,
    taxonomy,
    all_fine,
    all_coarse,
    data_csv,
    batch_size,
    device,
    test_split,
    split_col,
    eval_scopes,
    return_details=False,
    num_workers=0,
    pin_memory=False,
):
    loader = make_loader(
        data_csv,
        taxonomy,
        all_fine,
        batch_size,
        False,
        split=test_split,
        split_col=split_col,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    metrics_by_scope = {}
    for scope in eval_scopes:
        _, _, details = evaluate(model, loader, device, return_details=True, eval_scope=scope)
        metrics = compute_metrics_from_details(details)
        metrics.update({
            "eval_scope": scope,
            "num_opened_fine": len(all_fine),
            "num_opened_coarse": len(all_coarse),
        })
        if scope == "task":
            metrics.update(compute_task_breakdown_from_details(details))
        if return_details:
            metrics["details"] = details
            metrics["opened_fine_labels"] = list(all_fine)
            metrics["opened_coarse_labels"] = list(all_coarse)
        metrics_by_scope[scope] = metrics
    return metrics_by_scope

# -----------------------------------------------------------------------------
# Method / mode config overrides
# -----------------------------------------------------------------------------
def _apply_method_cfg_overrides(cfg: dict, method: str, mode: str) -> dict:
    """Mutate cfg in-place based on --method and --mode

      ours        – use YAML as-is (LoRA + replay + sibling + adaptor)
      finetune    – sequential per-session SGD, no CL mechanisms
      lwf / derpp – CL baselines: no LoRA, no ours replay, no sibling
      joint       – joint upper bound: no adaptor, no LoRA, no CL losses
    """
    m, t = cfg.setdefault("model", {}), cfg.setdefault("train", {})

    if method == "joint":
        m["use_mil_lora"] = False
        m["use_organ_task_adaptor"] = False
        m["organ_task_adaptor_mode"] = "none"
        t["train_shared_mil_after_base"] = True
        for key in ("lambda_replay", "lambda_replay_cls", "lambda_replay_align",
                    "lambda_sib", "lambda_weight_anchor",
                    "lambda_hyp_align", "lambda_hyp_sibling", "lambda_replay_hyp_align"):
            t[key] = 0.0

    elif method == "finetune":
        m["use_mil_lora"] = False
        m["use_organ_task_adaptor"] = False
        m["organ_task_adaptor_mode"] = "none"
        t["train_shared_mil_after_base"] = True
        for key in ("lambda_replay", "lambda_replay_cls", "lambda_replay_align",
                    "lambda_sib", "lambda_weight_anchor",
                    "lambda_hyp_align", "lambda_hyp_sibling", "lambda_replay_hyp_align"):
            t[key] = 0.0

    elif method in ("lwf", "derpp"):
        m["use_mil_lora"] = False
        t["train_shared_mil_after_base"] = True
        t["lambda_align"] = 0.0
        t["lambda_replay"] = 0.0
        t["lambda_sib"] = 0.0
        t["lambda_weight_anchor"] = 0.0

    # method == "ours": use YAML values unchanged
    print(
        f"[Config] method={method} | mode={mode} | "
        f"lora={m.get('use_mil_lora')} | "
        f"adaptor={m.get('organ_task_adaptor_mode')} | "
        f"replay={t.get('lambda_replay')} | "
        f"sibling={t.get('lambda_sib')}"
    )
    return cfg


def _apply_cli_overrides(cfg: dict, args) -> dict:
    """Apply CLI argument overrides on top of YAML + method overrides.

    Only overrides fields that were explicitly passed (non-None).
    """
    hyp = cfg.setdefault("hyperbolic", {})
    train = cfg.setdefault("train", {})

    if getattr(args, "hyp_enabled", None) is not None:
        hyp["enabled"] = bool(args.hyp_enabled)

    if getattr(args, "hyp_geometry", None) is not None:
        hyp["geometry"] = str(args.hyp_geometry).lower()

    if getattr(args, "lambda_orth", None) is not None:
        train["lambda_orth"] = float(args.lambda_orth)

    print(
        f"[Config] hyperbolic.enabled={hyp.get('enabled', False)} | "
        f"geometry={hyp.get('geometry', 'poincare')} | "
        f"lambda_orth={train.get('lambda_orth', 0.0)}"
    )
    return cfg

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ALL_15epoch.yaml")
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--data_csv", type=str, default="./data/all_conch_v15.csv")
    parser.add_argument("--split_col", type=str, default="split")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--taxonomy_json", type=str, default="./data/taxonomy_all.json")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--method", type=str, default="ours",
                        choices=["ours", "finetune", "lwf", "derpp", "joint"],
                        help=(
                            "ours     : our CL method (LoRA + replay + sibling)\n"
                            "finetune : sequential per-session training, no CL constraints\n"
                            "lwf      : Learning without Forgetting baseline\n"
                            "derpp    : DER++ replay baseline\n"
                            "joint    : joint upper bound (train/eval all labels at once)"
                        ))
    parser.add_argument("--scope", type=str, default="global", choices=["global", "task", "both"],
                        help="Eval scope: global | task | both")
    parser.add_argument("--use_text_refinement", action="store_true", default=False)
    parser.add_argument("--text_refinement_script", type=str, default="run_text_refinement.py")
    parser.add_argument("--text_refine_out_dir", type=str, default=None)

    # --- Hyperbolic override (overrides hyperbolic.enabled / geometry in YAML) ---
    hyp_grp = parser.add_mutually_exclusive_group()
    hyp_grp.add_argument("--hyp", dest="hyp_enabled", action="store_true",
                         help="Enable hyperbolic geometry (overrides config)")
    hyp_grp.add_argument("--no_hyp", dest="hyp_enabled", action="store_false",
                         help="Disable hyperbolic geometry (overrides config)")
    parser.set_defaults(hyp_enabled=None)
    parser.add_argument("--hyp_geometry", type=str, default="poincare", choices=["poincare", "lorentz"],
                        help="Hyperbolic geometry to use (overrides config): poincare | lorentz")

    # --- LoRA orthogonal loss override ---
    parser.add_argument("--lambda_orth", type=float, default=None,
                        help="LoRA orthogonal loss weight (overrides train.lambda_orth in config)")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_method_cfg_overrides(cfg, args.method, args.mode)
    cfg = _apply_cli_overrides(cfg, args)

    eval_scopes, scope_mode = resolve_eval_settings(args, cfg)
    set_seed(cfg.get("seed", 42))
    device = setup_device_from_arg(args.gpu)
    ensure_dir(args.out_dir)

    taxonomy = TaxonomyParser(args.taxonomy_json)
    session_manager = SessionManager(cfg, taxonomy)
    model = KnowledgeTreeCLModel(taxonomy, cfg).to(device)
    batch_size = cfg["train"].get("batch_size", 16)
    num_workers = cfg["train"].get("num_workers", 0)
    pin_memory = cfg["train"].get("pin_memory", False)
    use_amp = cfg["train"].get("use_amp", False)
    train_split = _none_if_blank(args.train_split)
    test_split = _none_if_blank(args.test_split)

    print(f"[Eval] scope={scope_mode}, eval_scopes={eval_scopes}")
    model_cfg = cfg.get("model", {})
    print(
        "[Model] "
        f"organ_task_adaptor_mode={model_cfg.get('organ_task_adaptor_mode', 'branch')} | "
        f"merge={model_cfg.get('organ_task_adaptor_merge', 'sum')} | "
        f"rank={model_cfg.get('organ_task_adaptor_rank', model_cfg.get('mil_lora_r', 16))}"
    )


    if args.mode == "train" and args.method == "joint":
        all_fine, all_coarse = get_used_labels_from_sessions(cfg, taxonomy)
        print_session_info("JOINT TRAIN", opened_fine=all_fine, opened_coarse=all_coarse)
        refined = run_text_refinement_pre_stage(args, session_idx=None)

        model.activate_nodes(all_coarse, all_fine, device)
        from engine.continual import apply_refined_text_anchors_to_model
        apply_refined_text_anchors_to_model(model, refined, all_fine, all_coarse, device)
        model.expand_classifiers(all_fine, device)
        register_sessions_until(model, session_manager, len(cfg["sessions"]) - 1, device, cfg)

        for p in model.parameters():
            p.requires_grad = True

        loader = make_loader(args.data_csv, taxonomy, all_fine, batch_size, True, split=train_split, split_col=args.split_col, num_workers=num_workers, pin_memory=pin_memory)
        optimizer = create_optimizer(model, cfg)
        scheduler = create_scheduler(optimizer, cfg, cfg["train"].get("epochs_joint", 30))

        for epoch in range(cfg["train"].get("epochs_joint", 30)):
            train_session(model, loader, optimizer, device, epoch, 0, None, cfg["train"].get("lambda_cls_task", 1.0), 0.0, train_cfg=cfg["train"], cfg=cfg)
            if scheduler is not None:
                scheduler.step()

        ckpt_path = os.path.join(args.out_dir, "ckpt_joint_last.pth")
        save_checkpoint(model, "joint_last", ckpt_path)

        metrics_by_scope = evaluate_joint_model_for_scopes(
            model,
            taxonomy,
            all_fine,
            all_coarse,
            args.data_csv,
            batch_size,
            device,
            test_split,
            args.split_col,
            eval_scopes,
            return_details=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        row = build_cumulative_result_row(
            len(cfg["sessions"]) - 1,
            "joint",
            ckpt_path,
            metrics_by_scope,
            scope_mode,
        )
        summaries = summarize_cumulative_results([row], scope_mode)
        for sc, sm in summaries.items():
            save_json(
                os.path.join(args.out_dir, f"joint_metrics_{sc}.json"),
                build_payload([row], sm, "joint", sc),
            )
        return

    if args.mode == "train" and args.method != "joint":
        replay_buffer = ReplayBuffer(feature_dim=cfg["model"].get("mil_hidden_dim", 512))
        results = []
        prev_refined = None

        # --- Method-specific state ---
        lwf_teacher = None
        der_buffer = None
        if args.method == "lwf":
            from engine.lwf import train_session_lwf, snapshot_teacher
            print("[Method] LwF (Learning without Forgetting)")
        elif args.method == "derpp":
            from engine.derpp import train_session_derpp, DERReplayBuffer, update_der_buffer
            der_buffer = DERReplayBuffer(
                max_size=cfg["train"].get("derpp_buffer_size", 500),
            )
            print("[Method] DER++ (Dark Experience Replay++)")
        else:
            print("[Method] Ours")

        for session_idx, session in enumerate(cfg["sessions"]):
            refined = run_text_refinement_pre_stage(args, session_idx=session_idx, prev_refined_anchor_path=prev_refined)
            info = setup_session_model(model, session_manager, session_idx, device, cfg, refined)
            prev_refined = refined

            print_session_info(
                "ORGAN-TASK TRAIN",
                session_idx,
                session.get("name", f"session_{session_idx}"),
                new_fine=info["new_fine"],
                new_coarse=info["new_coarse"],
                opened_fine=info["opened_fine"],
                opened_coarse=info["opened_coarse"],
            )

            session_datasets = session.get("datasets", None)
            loader = make_loader(args.data_csv, taxonomy, info["new_fine"], batch_size, True, split=train_split, split_col=args.split_col, dataset_filter=session_datasets, num_workers=num_workers, pin_memory=pin_memory)
            optimizer = scheduler = None
            prev_stage = None

            for epoch in range(cfg["train"].get("epochs_per_session", 10)):
                stage_changed, stage = configure_session_training_stage(model, info["task_key"], session_idx, epoch, cfg)
                if optimizer is None or stage_changed or stage != prev_stage:
                    optimizer = create_optimizer(model, cfg)
                    scheduler = create_scheduler(optimizer, cfg, cfg["train"].get("epochs_per_session", 10))
                    prev_stage = stage
                    print(f"[Session {session_idx}] stage -> {stage}")

                if args.method == "lwf":
                    train_session_lwf(
                        model, loader, optimizer, device, epoch, session_idx, cfg,
                        teacher=lwf_teacher,
                        lambda_fine=cfg["train"].get("lambda_cls_task", 1.0),
                    )
                elif args.method == "derpp":
                    train_session_derpp(
                        model, loader, optimizer, device, epoch, session_idx, cfg,
                        replay_buffer=der_buffer,
                        lambda_fine=cfg["train"].get("lambda_cls_task", 1.0),
                        use_amp=use_amp,
                    )
                else:  # ours
                    train_session(
                        model, loader, optimizer, device, epoch, session_idx,
                        replay_buffer=replay_buffer,
                        lambda_fine=cfg["train"].get("lambda_cls_task", cfg["train"].get("lambda_cls_fine", 1.0)),
                        lambda_align=cfg["train"].get("lambda_align", 1.0),
                        lambda_replay=cfg["train"].get("lambda_replay", 1.0),
                        train_cfg=cfg["train"],
                        cfg=cfg,
                        use_amp=use_amp,
                    )
                if scheduler is not None:
                    scheduler.step()

            if args.method == "lwf":
                lwf_teacher = snapshot_teacher(model)
            elif args.method == "derpp":
                update_der_buffer(model, loader, der_buffer, device, cfg)
            else:  # ours
                update_replay_buffer(model, loader, replay_buffer, device)

            # Session 0 (MIL full tuning) 후 base subspace 추출 → orth loss 기준
            if session_idx == 0:
                extract_base_subspace(model)

            ckpt_path = os.path.join(args.out_dir, f"ckpt_sess_{session_idx:02d}_last.pth")
            save_checkpoint(model, session_idx, ckpt_path)

            eval_dataset_filter = session_manager.opened_datasets(session_idx)
            metrics_by_scope = evaluate_model_for_scopes(
                model,
                session_manager,
                session_idx,
                taxonomy,
                args.data_csv,
                batch_size,
                device,
                test_split,
                args.split_col,
                eval_scopes,
                return_details=False,
                dataset_filter=eval_dataset_filter,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            row = build_cumulative_result_row(session_idx, session.get("name", f"session_{session_idx}"), ckpt_path, metrics_by_scope, scope_mode)
            results.append(row)
            print_end_session_eval(row, scope_mode)

        summaries = summarize_cumulative_results(results, scope_mode)
        for sc, sm in summaries.items():
            save_json(
                os.path.join(args.out_dir, f"train_end_session_metrics_{sc}.json"),
                build_payload(results, sm, "train_end_session", sc),
            )
        return

    if args.mode == "test" and args.method != "joint":
        if args.ckpt is None:
            raise ValueError("--ckpt is required for test")
        cm_root = os.path.join(resolve_ckpt_dir(args.ckpt), "confusion_matrices")
        results = {}

        for requested_idx in range(len(cfg["sessions"])):
            ckpt_path = find_session_ckpt_path(args.ckpt, requested_idx)
            if ckpt_path is None:
                print(f"[Skip] session {requested_idx} checkpoint not found")
                continue
            model_k, actual_idx = build_eval_model_for_session_ckpt(ckpt_path, taxonomy, cfg, session_manager, requested_idx, device)
            session_name = cfg["sessions"][actual_idx].get("name", f"session_{actual_idx}")

            eval_dataset_filter = session_manager.opened_datasets(actual_idx)
            metrics_by_scope = {}
            for scope in eval_scopes:
                m = evaluate_cumulative_open_set(
                    model_k,
                    session_manager,
                    actual_idx,
                    taxonomy,
                    args.data_csv,
                    batch_size,
                    device,
                    test_split=test_split,
                    split_col=args.split_col,
                    eval_scope=scope,
                    return_details=True,
                    dataset_filter=eval_dataset_filter,
                )
                metrics_by_scope[scope] = m
                save_eval_confusion_matrices(
                    m["details"],
                    m["opened_coarse_labels"],
                    m["opened_fine_labels"],
                    os.path.join(cm_root, scope, f"session_{actual_idx:02d}_{session_name}"),
                    f"A_{actual_idx}_{scope}",
                )
                display_acc = m.get("task_macro_acc", m["subtype_acc"])
                print(
                    f"[A_{actual_idx} | {scope}] {session_name} | "
                    f"ACC={display_acc:.2f}% "
                    f"wF1={m['subtype_wf1']:.4f} "
                    f"bACC={m['subtype_bacc']:.2f}%"
                )
            results[actual_idx] = build_cumulative_result_row(actual_idx, session_name, ckpt_path, metrics_by_scope, scope_mode)

        ordered = [results[k] for k in sorted(results)]
        summaries = summarize_cumulative_results(ordered, scope_mode)
        for sc, sm in summaries.items():
            save_json(
                os.path.join(resolve_ckpt_dir(args.ckpt), f"cil_cumulative_metrics_{sc}.json"),
                build_payload(ordered, sm, "test", sc),
            )
        return

    if args.mode == "test" and args.method == "joint":
        if args.ckpt is None:
            raise ValueError("--ckpt is required for test")

        # Joint checkpoint is trained with all labels opened, but evaluation should
        # still be reported cumulatively as A_0, A_1, ..., A_T for fair comparison
        # with continual results.
        all_fine, all_coarse = get_used_labels_from_sessions(cfg, taxonomy)
        model.activate_nodes(all_coarse, all_fine, device)
        model.expand_classifiers(all_fine, device)
        register_sessions_until(model, session_manager, len(cfg["sessions"]) - 1, device, cfg)

        state, _ = load_ckpt_and_get_session_idx(args.ckpt, device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Load warning] missing keys: {len(missing)}")
        if unexpected:
            print(f"[Load warning] unexpected keys: {len(unexpected)}")
        model.eval()

        cm_root = os.path.join(resolve_ckpt_dir(args.ckpt), "confusion_matrices_joint")
        results = []

        for session_idx, session in enumerate(cfg["sessions"]):
            session_name = session.get("name", f"session_{session_idx}")
            metrics_by_scope = {}

            for scope in eval_scopes:
                m = evaluate_cumulative_open_set(
                    model,
                    session_manager,
                    session_idx,
                    taxonomy,
                    args.data_csv,
                    batch_size,
                    device,
                    test_split=test_split,
                    split_col=args.split_col,
                    eval_scope=scope,
                    return_details=True,
                )
                metrics_by_scope[scope] = m

                save_eval_confusion_matrices(
                    m["details"],
                    m["opened_coarse_labels"],
                    m["opened_fine_labels"],
                    os.path.join(cm_root, scope, f"session_{session_idx:02d}_{session_name}"),
                    f"J_{session_idx}_{scope}",
                )

                display_acc = m.get("task_macro_acc", m["subtype_acc"])
                print(
                    f"[JOINT A_{session_idx} | {scope}] {session_name} | "
                    f"ACC={display_acc:.2f}% "
                    f"wF1={m['subtype_wf1']:.4f} "
                    f"bACC={m['subtype_bacc']:.2f}%"
                )
                
            row = build_cumulative_result_row(
                session_idx,
                session_name,
                args.ckpt,
                metrics_by_scope,
                scope_mode,
            )
            results.append(row)
            print_end_session_eval(row, scope_mode)

        summaries = summarize_cumulative_results(results, scope_mode)
        for sc, sm in summaries.items():
            save_json(
                os.path.join(resolve_ckpt_dir(args.ckpt), f"joint_metrics_{sc}.json"),
                build_payload(results, sm, "joint", sc),
            )
        return


if __name__ == "__main__":
    main()
