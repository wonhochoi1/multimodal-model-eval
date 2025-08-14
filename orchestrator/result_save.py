from pathlib import Path
import json, statistics as stats

class ResultStore:
    def __init__(self, out_dir: Path):
        self.out = out_dir; (self.out/"raw").mkdir(parents=True, exist_ok=True)
        self.trials = []
    def write_trial(self, obj: dict):
        self.trials.append(obj)
        (self.out/"raw"/f"{obj['trial_id']}.json").write_text(json.dumps(obj))
    def aggregate(self, cfg: dict) -> dict:
        # toy aggregator: compute a few means; extend as needed
        def pick(key):
            return [t[key] for t in self.trials if key in t]
        summary = {"suite": cfg.get("suite",""), "count": len(self.trials)}
        return summary