from typing import Dict, List, Any
import json

class MetricUtils:
    """Common utilities for metric computation"""
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate multiple metric results"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m.get(key, 0.0) for m in metrics_list if m.get(key) is not None]
            if values:
                aggregated[f"{key}_avg"] = sum(values) / len(values)
                aggregated[f"{key}_min"] = min(values)
                aggregated[f"{key}_max"] = max(values)
        
        return aggregated
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], output_path: str):
        """Save metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    @staticmethod
    def load_metrics(input_path: str) -> Dict[str, Any]:
        """Load metrics from JSON file"""
        with open(input_path, 'r') as f:
            return json.load(f)