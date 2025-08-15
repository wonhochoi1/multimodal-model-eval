import time
import uuid
from typing import Dict, Any, List
from pathlib import Path
import json

from agent.probes.latency_probe import LatencyProbe
from agent.probes.power_probe import PowerProbe
from agent.probes.util_probe import UtilProbe
from agent.probes.memory_probe import MemoryProbe
from agent.adapters.base import Adapter
from metrics.accuracy import QualityMetrics
from metrics.deployability import DeployabilityMetrics

class LocalAgentRunner:
    """Agent runner for local laptop/macOS development"""
    
    def __init__(self, adapter: Adapter, sample_rate_hz: int = 2):
        self.adapter = adapter
        self.latency_probe = LatencyProbe()
        self.power_probe = PowerProbe(sample_rate_hz)
        self.util_probe = UtilProbe(sample_rate_hz)
        self.memory_probe = MemoryProbe()
        
        self.platform = self._detect_platform()
    
    def _detect_platform(self) -> str:
        import platform
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"
        else:
            return "linux"
    
    def run_trial(self, sample: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        trial_id = str(uuid.uuid4())
        
        print(f"Running trial {trial_id[:8]}:")
        
        self.latency_probe.start()
        self.power_probe.start()
        self.util_probe.start()
        self.memory_probe.start()
        
        t_start = time.time()
        

        with self.latency_probe.stage("prepare"):
            prepared = self.adapter.prepare(sample)
        
        with self.latency_probe.stage("infer"):
            self.memory_probe.sample()
            self.util_probe.sample()
            
            raw_output = self.adapter.infer(prepared)
        
        with self.latency_probe.stage("postprocess"):
            output = self.adapter.postprocess(raw_output)
        
        t_end = time.time()
        
        latency_metrics = self.latency_probe.stop()
        power_metrics = self.power_probe.stop()
        util_metrics = self.util_probe.stop()
        memory_metrics = self.memory_probe.stop()
        
        trial_record = {
            "trial_id": trial_id,
            "platform": self.platform,
            "input": sample,
            "output": output,
            "timing": {
                "t_start": t_start,
                "t_end": t_end,
                "t_first": output.get("timing", {}).get("t_first", t_end),
                "stages": latency_metrics["stages"],
                "total_ms": latency_metrics["total_ms"]
            },
            "resources": {
                "memory": memory_metrics,
                "power": power_metrics,
                "utilization": util_metrics
            },
            "config": config
        }
        
        print(f"  Trial completed: {trial_record['timing']['total_ms']:.1f}ms")
        return trial_record
            
    
    
    def run_suite(self, suite_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        for task in suite_config.get("tasks", []):
            print(f"\nTask: {task['id']}")
            task_results = self._run_task(task, suite_config, output_dir)
            all_results[task["id"]] = task_results
        
        self._save_results(all_results, output_dir)
        
        print(f"\n Results saved to {output_dir}")
        return all_results
    
    def _run_task(self, task: Dict[str, Any], suite_config: Dict[str, Any], 
                  output_dir: Path) -> List[Dict[str, Any]]:
        """Execute a single task"""
        task_id = task["id"]
        repeats = suite_config.get("run", {}).get("repeats", 1)
        
        samples = self._create_synthetic_samples(task, repeats)
        
        trials = []
        for i, sample in enumerate(samples):
            print(f"  Trial {i+1}/{len(samples)}")
            trial_result = self.run_trial(sample, task.get("adapter_config", {}))
            trial_result["task_id"] = task_id
            trial_result["repeat"] = i
            trials.append(trial_result)
        
        return trials
    
    def _create_synthetic_samples(self, task: Dict[str, Any], num_samples: int) -> List[Dict[str, Any]]:
        """Create synthetic test samples for local development"""
        samples = []
        
        for i in range(num_samples):
            if task["type"] == "vlm_caption":
                sample = {
                    "id": f"local_sample_{i:03d}",
                    "image": f"/mock/path/to/test_image_{i}.jpg",
                    "prompt": f"Describe what you see in image {i}"
                }
            elif task["type"] == "image_classification":
                sample = {
                    "id": f"local_sample_{i:03d}",
                    "image": f"/mock/path/to/test_image_{i}.jpg",
                    "label": f"class_{i % 10}"  # Mock labels
                }
            elif task["type"] == "asr":
                sample = {
                    "id": f"local_sample_{i:03d}",
                    "audio": f"/mock/path/to/test_audio_{i}.wav",
                    "text": f"This is test audio sample number {i}"
                }
            else:
                sample = {
                    "id": f"local_sample_{i:03d}",
                    "data": f"mock_data_{i}"
                }
            
            samples.append(sample)
        
        return samples
    
    def _save_results(self, all_results: Dict[str, List], output_dir: Path):

        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        for task_id, trials in all_results.items():
            task_file = raw_dir / f"{task_id}.jsonl"
            with open(task_file, "w") as f:
                for trial in trials:
                    f.write(json.dumps(trial) + "\n")
        
        metrics = self._compute_metrics(all_results)
        
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        summary = {
            "platform": self.platform,
            "timestamp": time.time(),
            "tasks": list(all_results.keys()),
            "total_trials": sum(len(trials) for trials in all_results.values()),
            "metrics": metrics
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    def _compute_metrics(self, all_results: Dict[str, List]) -> Dict[str, Any]:
        metrics = {}
        
        for task_id, trials in all_results.items():
            task_metrics = {
                "quality": {},
                "deployability": {}
            }
            
            if any("refs" in trial for trial in trials):
                task_metrics["quality"]["exact_match"] = QualityMetrics.compute_exact_match(trials)
            
            task_metrics["deployability"].update(
                DeployabilityMetrics.compute_ttft_ms(trials)
            )
            task_metrics["deployability"].update(
                DeployabilityMetrics.compute_e2e_ms(trials)
            )
            task_metrics["deployability"].update(
                DeployabilityMetrics.compute_tps(trials)
            )
            task_metrics["deployability"].update(
                DeployabilityMetrics.compute_resource_metrics(trials)
            )
            
            metrics[task_id] = task_metrics
        
        return metrics