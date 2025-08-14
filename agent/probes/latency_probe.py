import time
from contextlib import contextmanager
from typing import Dict, Any

class LatencyProbe:
    """Latency probe optimized for local development"""
    
    def __init__(self):
        self.stage_timings = {}
        self.current_stage = None
        self.stage_start = None
    
    def start(self):
        """Start timing a trial"""
        self.stage_timings.clear()
        self.current_stage = None
        self.stage_start = None
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing individual stages"""
        if self.stage_start:
            # End previous stage
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
        """End trial and return stage timings"""
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
