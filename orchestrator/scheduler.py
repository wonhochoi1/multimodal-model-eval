from pathlib import Path
from mm_eval.agent.runner import run_trial
from mm_eval.orchestrator.result_store import ResultStore

# NOTE: simple single-process scheduler for now

def run_suite(cfg: dict, out_dir: Path) -> dict:
    store = ResultStore(out_dir)
    # Expand models × tasks × datasets
    for m in cfg.get("models", []):
        for task in cfg.get("tasks", []):
            for ds in task["datasets"]:
                repeats = task.get("repeats", 1)
                warmup = task.get("warmup", 0)
                for i in range(warmup + repeats):
                    trial = run_trial(task_cfg={**task, "dataset": ds}, model_cfg=m, warmup=(i < warmup))
                    if not trial: continue
                    store.write_trial(trial)
    return store.aggregate(cfg)