from typing import Dict, List, Any
import statistics

class DeployabilityMetrics:
    """Compute deployability metrics for edge deployment assessment"""
    
    @staticmethod
    def compute_ttft_ms(trials: List[Dict]) -> Dict[str, float]:
        """Time-To-First-Token metrics (crucial for interactive applications)"""
        ttft_values = []
        for trial in trials:
            if "timing" in trial and "t_first" in trial["timing"]:
                ttft = (trial["timing"]["t_first"] - trial["timing"]["t_start"]) * 1000
                ttft_values.append(ttft)
        
        if not ttft_values:
            return {"ttft_ms_avg": 0.0, "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0}
        
        return {
            "ttft_ms_avg": statistics.mean(ttft_values),
            "ttft_ms_p50": statistics.median(ttft_values),
            "ttft_ms_p95": statistics.quantiles(ttft_values, n=20)[18] if len(ttft_values) >= 20 else max(ttft_values)
        }
    
    @staticmethod
    def compute_e2e_ms(trials: List[Dict]) -> Dict[str, float]:
        """End-to-end latency metrics"""
        e2e_values = []
        for trial in trials:
            if "timing" in trial:
                e2e = (trial["timing"]["t_end"] - trial["timing"]["t_start"]) * 1000
                e2e_values.append(e2e)
        
        if not e2e_values:
            return {"e2e_ms_avg": 0.0, "e2e_ms_p50": 0.0, "e2e_ms_p95": 0.0}
        
        return {
            "e2e_ms_avg": statistics.mean(e2e_values),
            "e2e_ms_p50": statistics.median(e2e_values),
            "e2e_ms_p95": statistics.quantiles(e2e_values, n=20)[18] if len(e2e_values) >= 20 else max(e2e_values)
        }
    
    @staticmethod
    def compute_tps(trials: List[Dict]) -> Dict[str, float]:
        """Tokens per second for generative tasks"""
        tps_values = []
        for trial in trials:
            if "output" in trial and "pred" in trial["output"]:
                pred = trial["output"]["pred"]
                if "tokens" in pred and "timing" in trial:
                    num_tokens = len(pred["tokens"])
                    if num_tokens > 0:
                        # Exclude TTFT from TPS calculation
                        generation_time = trial["timing"]["t_end"] - trial["timing"]["t_first"]
                        if generation_time > 0:
                            tps = num_tokens / generation_time
                            tps_values.append(tps)
        
        if not tps_values:
            return {"tps_avg": 0.0, "tps_p50": 0.0, "tps_p95": 0.0}
        
        return {
            "tps_avg": statistics.mean(tps_values),
            "tps_p50": statistics.median(tps_values),
            "tps_p95": statistics.quantiles(tps_values, n=20)[18] if len(tps_values) >= 20 else max(tps_values)
        }
    
    @staticmethod
    def compute_resource_metrics(trials: List[Dict]) -> Dict[str, float]:
        """Aggregated resource usage metrics"""
        power_values = []
        memory_values = []
        gpu_util_values = []
        cpu_util_values = []
        
        for trial in trials:
            if "resources" in trial:
                resources = trial["resources"]
                
                # Power
                if "power" in resources and "power_w_avg" in resources["power"]:
                    power_values.append(resources["power"]["power_w_avg"])
                
                # Memory
                if "memory" in resources and "ram_mb_peak" in resources["memory"]:
                    memory_values.append(resources["memory"]["ram_mb_peak"])
                
                # GPU utilization
                if "utilization" in resources and "gpu_util_pct_avg" in resources["utilization"]:
                    gpu_util_values.append(resources["utilization"]["gpu_util_pct_avg"])
                
                # CPU utilization
                if "utilization" in resources and "cpu_util_pct_avg" in resources["utilization"]:
                    cpu_util_values.append(resources["utilization"]["cpu_util_pct_avg"])
        
        return {
            "power_w_avg": statistics.mean(power_values) if power_values else 0.0,
            "power_w_max": max(power_values) if power_values else 0.0,
            "memory_mb_avg": statistics.mean(memory_values) if memory_values else 0.0,
            "memory_mb_max": max(memory_values) if memory_values else 0.0,
            "gpu_util_pct_avg": statistics.mean(gpu_util_values) if gpu_util_values else 0.0,
            "cpu_util_pct_avg": statistics.mean(cpu_util_values) if cpu_util_values else 0.0
        }
    
    @staticmethod
    def compute_rtf(trials: List[Dict]) -> Dict[str, float]:
        """Real-Time Factor for audio tasks"""
        rtf_values = []
        for trial in trials:
            if "timing" in trial and "audio_duration" in trial:
                processing_time = trial["timing"]["t_end"] - trial["timing"]["t_start"]
                audio_duration = trial["audio_duration"]
                if audio_duration > 0:
                    rtf = processing_time / audio_duration
                    rtf_values.append(rtf)
        
        if not rtf_values:
            return {"rtf_avg": 0.0, "rtf_p50": 0.0}
        
        return {
            "rtf_avg": statistics.mean(rtf_values),
            "rtf_p50": statistics.median(rtf_values)
        }