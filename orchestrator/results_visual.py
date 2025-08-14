import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

class ReportBuilder:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        
    def build_report(self) -> str:
        """Build HTML report from results"""
        # Load metrics
        metrics_file = self.results_dir / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Generate HTML
        html = self._generate_html(metrics)
        return html
    
    def _generate_html(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .task {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .metric-group {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ color: #0066cc; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>Multimodal Model Evaluation Report</h1>
            <p>Generated from: {self.results_dir}</p>
        """
        
        for task_id, task_metrics in metrics.items():
            html += f"""
            <div class="task">
                <h2>Task: {task_id}</h2>
                <div class="metrics">
                    <div class="metric-group">
                        <h3>Quality Metrics</h3>
                        {self._render_metrics(task_metrics.get("quality", {}))}
                    </div>
                    <div class="metric-group">
                        <h3>Deployability Metrics</h3>
                        {self._render_metrics(task_metrics.get("deployability", {}))}
                    </div>
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _render_metrics(self, metrics: Dict[str, float]) -> str:
        """Render metrics as HTML"""
        if not metrics:
            return "<p>No metrics available</p>"
        
        html = ""
        for name, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            html += f"""
            <div class="metric">
                <span class="metric-name">{name}:</span>
                <span class="metric-value">{formatted_value}</span>
            </div>
            """
        
        return html