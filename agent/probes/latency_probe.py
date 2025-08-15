import time
from contextlib import contextmanager
from typing import Dict, Any

class LatencyProbe:
    def __init__(self):
        self.stage_timings = {}
        self.current_stage = None
        self.stage_start = None
    
    def start(self):
        self.stage_timings.clear()  
        self.current_stage = None
        self.stage_start = None
    
    @contextmanager
    def stage(self, stage_name: str):
        if self.stage_start:
            self.stage_timings[self.current_stage] = time.time() - self.stage_start
        
        self.current_stage = stage_name
        self.stage_start = time.time()
        try:
            yield
        finally:
            if self.stage_start:
                self.stage_timings[stage_name] = time.time() - self.stage_start
                self.stage_start = None
    
    def stop(self) -> Dict[str, Any]:
        if self.stage_start:
            self.stage_timings[self.current_stage] = time.time() - self.stage_start
        
        total_time = sum(self.stage_timings.values())
        
        return {
            "stages": self.stage_timings.copy(),
            "total_s": total_time,
            "total_ms": total_time * 1000,
            "stage_breakdown": {
                name: f"{time*1000:.1f}ms" 
                for name, time in self.stage_timings.items()
            }
        }
