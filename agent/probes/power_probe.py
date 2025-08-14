class PowerProbe:
    def __init__(self, sample_rate_hz: int = 1):
        self.sample_rate_hz = sample_rate_hz
        self.samples = []
    
    def start(self): 
        self.samples = []
    
    def stop(self):
        # TODO: integrate NVML/tegrastats/RAPL
        return {"power_avg_W": None, "power_max_W": None, "energy_J": None}
