from collections import defaultdict
import json


class TaxonomyParser:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.coarse_nodes = data["coarse_nodes"]
        self.fine_nodes = data["fine_nodes"]

        self.coarse_labels = list(self.coarse_nodes.keys())
        self.fine_labels = list(self.fine_nodes.keys())

        self.coarse_to_idx = {k: i for i, k in enumerate(self.coarse_labels)}
        self.fine_to_idx = {k: i for i, k in enumerate(self.fine_labels)}

        self.idx_to_coarse = {i: k for i, k in enumerate(self.coarse_labels)}
        self.idx_to_fine = {i: k for i, k in enumerate(self.fine_labels)}

        self.fine_to_coarse = {
            fine: info["parent"]
            for fine, info in self.fine_nodes.items()
        }

        self.fine_to_task = {
            fine: info.get("task", "default")
            for fine, info in self.fine_nodes.items()
        }

        self.organ_task_to_fines = defaultdict(list)
        self.task_to_fines = defaultdict(list)

        for fine, info in self.fine_nodes.items():
            organ = info["parent"]
            task = info.get("task", "default")

            self.organ_task_to_fines[(organ, task)].append(fine)
            self.task_to_fines[task].append(fine)

    def get_candidate_fine_labels(self, organ: str = None, task: str = None):
        if organ is not None and task is not None:
            return list(self.organ_task_to_fines.get((organ, task), []))

        if task is not None:
            return list(self.task_to_fines.get(task, []))

        if organ is not None:
            return [
                fine
                for fine, parent in self.fine_to_coarse.items()
                if parent == organ
            ]

        return list(self.fine_labels)

    def get_coarse_text(self, label: str) -> str:
        info = self.coarse_nodes[label]
        synonyms = ", ".join(info.get("synonyms", []))
        return (
            f"organ group: {info['display_name']}. "
            f"definition: {info.get('definition', '')}. "
            f"synonyms: {synonyms}."
        )

    def get_fine_text(self, label: str) -> str:
        info = self.fine_nodes[label]
        synonyms = ", ".join(info.get("synonyms", []))
        morph = ", ".join(info.get("morphology", []))
        return (
            f"fine subtype: {info['display_name']}. "
            f"definition: {info.get('definition', '')}. "
            f"synonyms: {synonyms}. "
            f"morphology: {morph}."
        )

    def get_task_text(self, task_key: str, fine_labels: list = None) -> str:
        """Generate text description for a task node.

        task_key format: "ORGAN/task_name"  (e.g. "BRAIN/brain_glioma_subtype")
        fine_labels: list of fine label strings belonging to this task (e.g. ["LUAD", "LUSC"]).
                     When provided, task_axis is looked up directly from fine node metadata
                     rather than being inferred from the task key string.
        """
        if "/" in task_key:
            organ, task_name = task_key.split("/", 1)
        else:
            organ, task_name = task_key, "default"

        coarse_info = self.coarse_nodes.get(organ, {})
        organ_name = coarse_info.get("display_name", organ)

        # 1) Prefer task_axis from provided fine labels (most accurate)
        axis = None
        for fl in (fine_labels or []):
            node = self.fine_nodes.get(fl)
            if node and node.get("task_axis"):
                axis = node["task_axis"]
                break

        # 2) Fallback: any TCGA fine node under this organ
        if not axis:
            for fl, info in self.fine_nodes.items():
                if info.get("parent") == organ and "TCGA" in info.get("source_datasets", []):
                    axis = info.get("task_axis")
                    break

        # 3) Last resort: parse axis from task_name string
        if not axis:
            prefix = organ.lower() + "_"
            axis = task_name[len(prefix):] if task_name.lower().startswith(prefix) else task_name
            axis = axis.replace("_", " ")

        synonyms = ", ".join(coarse_info.get("synonyms", []))
        return (
            f"histological classification task: {axis} in {organ_name} tissue. "
            f"organ: {organ_name}. "
            f"task axis: {axis}. "
            f"synonyms: {synonyms}."
        )