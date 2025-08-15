import time
import psutil
from typing import Dict, Any
import threading

class UtilProbe:
    """Utilization probe optimized for local development"""
    
    def __init__(self, sample_rate_hz: int = 2):
        self.sample_rate_hz = sample_rate_hz
        self.sampling = False
        self.samples = []
        self.sample_thread = None
        
    def start(self):
        """Start utilization sampling"""
        self.sampling = True
        self.samples.clear()
        self.sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.sample_thread.start()
    
    def _sample_loop(self):
        """Background sampling loop"""
        while self.sampling:
            sample = {
                "timestamp": time.time(),
                "cpu_util_pct": psutil.cpu_percent(interval=0.1),
                "ram_util_pct": psutil.virtual_memory().percent,
                "ram_mb": psutil.virtual_memory().used / (1024 * 1024),
                "gpu_util_pct": self._get_util(),
                "gpu_mem_pct": self._get_mem()
            }
            self.samples.append(sample)
            time.sleep(1.0 / self.sample_rate_hz)

    
    def _get_util(self) -> float:
        """Get GPU utilization percentage (local fallbacks)"""
        # Try NVML first
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            # if not, calculate cpu utilization
            return psutil.cpu_percent(interval=0.1)
    
    def _get_mem(self) -> float:
        """Get GPU memory utilization percentage (local fallbacks)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (mem_info.used / mem_info.total) * 100.0
        except:
            # if not, calculate virtual memory utilization
            return psutil.virtual_memory().percent
    
    def sample(self):
        """Take a single sample manually (in addition to background sampling)"""
        if self.sampling:
            # If background sampling is active, just return - samples are being collected
            return
        
        # If background sampling is not active, take a manual sample
        try:
            sample = {
                "timestamp": time.time(),
                "cpu_util_pct": psutil.cpu_percent(interval=0.1),
                "ram_util_pct": psutil.virtual_memory().percent,
                "ram_mb": psutil.virtual_memory().used / (1024 * 1024),
                "gpu_util_pct": self._get_util(),
                "gpu_mem_pct": self._get_mem()
            }
            self.samples.append(sample)
        except Exception as e:
            print(f"Util probe manual sampling error: {e}")
    
    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return aggregated metrics"""
        self.sampling = False
        if self.sample_thread:
            self.sample_thread.join(timeout=1.0)
        
        if not self.samples:
            return {
                "cpu_util_pct_avg": 0.0,
                "cpu_util_pct_max": 0.0,
                "ram_util_pct_avg": 0.0,
                "ram_mb_avg": 0.0,
                "gpu_util_pct_avg": 0.0,
                "gpu_util_pct_max": 0.0
            }
        
        # Aggregate samples
        cpu_utils = [s["cpu_util_pct"] for s in self.samples]
        ram_utils = [s["ram_util_pct"] for s in self.samples]
        ram_mb = [s["ram_mb"] for s in self.samples]
        gpu_utils = [s["gpu_util_pct"] for s in self.samples]
        
        return {
            "cpu_util_pct_avg": sum(cpu_utils) / len(cpu_utils),
            "cpu_util_pct_max": max(cpu_utils),
            "ram_util_pct_avg": sum(ram_utils) / len(ram_utils),
            "ram_mb_avg": sum(ram_mb) / len(ram_mb),
            "gpu_util_pct_avg": sum(gpu_utils) / len(gpu_utils),
            "gpu_util_pct_max": max(gpu_utils),
            "samples_count": len(self.samples)
        }

