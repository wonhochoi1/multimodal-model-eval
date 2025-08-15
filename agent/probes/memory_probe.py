import psutil
import time
from typing import Dict, Any

class MemoryProbe:
    def __init__(self):
        self.initial_ram = None
        self.initial_vram = None
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self.samples = []
    
    def start(self):
        self.initial_ram = psutil.virtual_memory().used / (1024 * 1024)  # MB
        self.initial_vram = self._get_vram_mb()
        self.peak_ram = self.initial_ram
        self.peak_vram = self.initial_vram
        self.samples = []
        
        self.samples.append({
            "timestamp": time.time(),
            "ram_mb": self.initial_ram,
            "vram_mb": self.initial_vram,
            "phase": "start"
        })
    
    def _get_vram_mb(self) -> float:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used / (1024 * 1024)
        except:
            return psutil.virtual_memory().used / (1024 * 1024)
    
    def sample(self):
        current_ram = psutil.virtual_memory().used / (1024 * 1024)
        current_vram = self._get_vram_mb()
        
        self.peak_ram = max(self.peak_ram, current_ram)
        self.peak_vram = max(self.peak_vram, current_vram)
        
        self.samples.append({
            "timestamp": time.time(),
            "ram_mb": current_ram,
            "vram_mb": current_vram,
            "phase": "inference"
        })
    
    def stop(self) -> Dict[str, Any]:
        final_ram = psutil.virtual_memory().used / (1024 * 1024)
        final_vram = self._get_vram_mb()
        
        self.samples.append({
            "timestamp": time.time(),
            "ram_mb": final_ram,
            "vram_mb": final_vram,
            "phase": "end"
        })
        
        return {
            "ram_mb_initial": self.initial_ram,
            "ram_mb_peak": self.peak_ram,
            "ram_mb_final": final_ram,
            "ram_mb_delta": final_ram - self.initial_ram,
            "vram_mb_initial": self.initial_vram,
            "vram_mb_peak": self.peak_vram,
            "vram_mb_final": final_vram,
            "vram_mb_delta": final_vram - self.initial_vram,
            "samples": self.samples
        }

