# multimodal-model-eval
Evaluation Metrics and Tool for Multimodal DL Models

**Evaluate VLMs and multimodal models (audio, image, video) on your local laptop/macOS or edge devices** with a consistent harness that measures **quality** and **deployability** (latency, throughput, memory, power, thermals). Works as a two-part system:

- **Orchestrator** (your laptop/server): schedules runs, collects logs, computes metrics, and builds reports.
- **Agent** (local laptop/edge box): runs the model adapter, streams telemetry (power/util/temps), and returns predictions + raw timings.

---
## Why this repo?

Modern VLMs and multimodal models must be **accurate** *and* **edge‑friendly**. EdgeEval‑MM runs repeatable suites against your adapters (ASR/TTS, image classifiers, captioners/VQA, video action, world‑model rollouts) while recording **TTFT**, **time‑per‑token/frame**, **throughput**, **real‑time factor (RTF)**, **memory**, **power**, **utilization**, and **thermals**. It then computes task‑appropriate **quality metrics** (WER, BLEU/ROUGE, PSNR/SSIM, FVD, etc.) and renders a report.

**Now with local-first development**: Test everything on your laptop/macOS before deploying to edge hardware!

---

## Quick start

### 1) **Install (local development environment)**

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2) **Test locally first (recommended)**

```bash
# Run a quick local test to verify everything works
python -m orchestrator.cli test

# Or run a specific local suite
python -m orchestrator.cli run suites/local_test.yaml
```

### 3) **Add your own VLM model**

The system is designed to easily integrate any VLM model. Simply:

1. **Create a new adapter** in `agent/adapters/`:
```python
# agent/adapters/my_vlm.py
from .base import Adapter

class MyVLMAdapter(Adapter):
    name = "MyVLM"
    modalities = ["image", "text"]
    
    def load(self, config):
        # Load your model here
        self.model = load_my_vlm_model(config)
    
    def prepare(self, sample):
        # Prepare input for your model
        return process_sample(sample)
    
    def infer(self, prepared):
        # Run inference with your model
        return self.model.generate(prepared)
    
    def postprocess(self, raw):
        # Format output for metrics
        return format_output(raw)
```

2. **Add it to your suite YAML**:
```yaml
# suites/my_vlm_test.yaml
adapter_config:
  model_type: "my_vlm"
  # your model-specific config here
```

3. **Run your evaluation**:
```bash
python -m orchestrator.cli run suites/my_vlm_test.yaml
```

### 4) **Use Hugging Face models**

The system includes built-in support for Hugging Face models:

```yaml
# suites/hf_vlm_test.yaml
adapter_config:
  model_name: "microsoft/DialoGPT-medium"  # or any HF model
  precision: "fp16"  # or "fp32"
  max_new_tokens: 50
  temperature: 0.7

run:
  warmup: 3
  repeats: 5

tasks:
  - id: vlm_captioning
    type: vlm_caption
    prompt: "Describe what you see in this image"
    image: "/path/to/image.jpg"
```

### 5) **Deploy to edge hardware (when ready)**

When you're ready for edge deployment:

1. **Update your suite configuration**:
```yaml
# suites/edge_deployment.yaml
device: jetson_orin  # or your edge device
agent:
  transport: ssh
  host: 192.168.1.50
  probes:
    power: tegrastats     # real edge power monitoring
    util: nvml            # real GPU monitoring
```

2. **Run on edge hardware**:
```bash
python -m orchestrator.cli run suites/edge_deployment.yaml
```

---

## Local Development Features

### **Local-First Architecture**
- **Works immediately** on your laptop/macOS/Windows
- **Real system monitoring** using psutil, NVML (if available)
- **Synthetic test data** for rapid development
- **No hardware deployment** needed for testing

### **Platform Detection**
The system automatically detects your platform and adapts:
- **macOS**: Optimized for Apple Silicon/Intel
- **Windows**: Windows-specific monitoring
- **Linux**: Full probe support
- **Edge**: Real hardware sensors when available

### **Probe System**
- **LatencyProbe**: Stage-by-stage timing with context managers
- **PowerProbe**: Local power monitoring + edge options (tegrastats, RAPL)
- **UtilProbe**: CPU/GPU utilization, memory usage
- **MemoryProbe**: Peak RAM/VRAM tracking

---

## Input/Output contracts

### Dataset sample schemas (JSONL manifests)

- **Image classification**
  ```json
  {"id": "img_001", "image": "/abs/path/cat.jpg", "label": "cat"}
  ```

- **Image captioning / VQA**
  ```json
  {"id":"ex_42","image":"/abs/path.jpg","question":"What is on the table?","answer":"mug"}
  ```

- **Video QA / Action classification**
  ```json
  {"id":"vid_12","video":"/abs/path.mp4","question":"What sport?","answer":"basketball"}
  ```

- **ASR**
  ```json
  {"id":"utt_7","audio":"/abs/path.wav","text":"reference transcript"}
  ```

### Adapter I/O (Python type hints)

```python
class AdapterInput(TypedDict, total=False):
    id: str
    image: str                  # path
    video: str                  # path
    audio: str                  # path
    question: str               # for VQA/VideoQA
    text: str                   # for TTS

class AdapterOutput(TypedDict):
    id: str
    pred: dict                  # task-specific prediction
    timing: dict                # raw timestamps: t_start, t_first, t_end, per-stage
    resources: dict             # peak_mem_mb, gpu_util_pct, cpu_util_pct, power_w, temps
    artifacts: dict             # optional paths: generated_audio, video, viz overlays
```

#### Task-specific `pred` examples

- Classification: `{"label":"dog","topk":[["dog",0.92],["cat",0.04],...]}`
- Caption/VQA: `{"text":"A person riding a bike."}`
- Video action: `{"label":"basketball","topk":[["basketball",0.81],...]}`
- ASR: `{"text":"hello world","words":[{"w":"hello","ts":[0.10,0.40]}, ...]}`
- TTS: `{"audio":"/tmp/tts.wav","sample_rate":22050}`
- World model (optional): `{"video":"/tmp/pred.mp4","fps":8}`

### Timing definitions

- **TTFT**: `t_first - t_start` (first token/byte/logit emitted)
- **E2E latency**: `t_end - t_start`
- **TPS / tokens_per_sec**: for generative text/VLM tasks after first token
- **FPS**: frames/sec for video decode/encode or world-model generation
- **RTF**: `processing_time / audio_duration` for ASR/TTS

Agent fills these in; probes populate `resources` concurrently.

---

## Adapters

All adapters subclass `adapters/base.py`:

```python
class Adapter(Protocol):
    def load(self, config: dict) -> None: ...
    def prepare(self, sample: dict) -> Any: ...
    def infer(self, prepared) -> Any: ...
    def postprocess(self, raw) -> dict: ...
```

### **Built-in Adapters**

- `vlm_example.py` — Simple VLM adapter for testing
- `hf_vlm_adapter.py` — Hugging Face model integration
- `image_classifier.py` — Image classification (Torch/TensorRT)
- `asr_tts.py` — ASR (WER/RTF) and TTS (RTF/MOS‑proxy)
- `video_worldmodel.py` — optional Genie‑style rollout adapter

### **Adding Your Own Adapter**

1. **Copy the base adapter**:
```bash
cp agent/adapters/base.py agent/adapters/my_model.py
```

2. **Implement the required methods**:
```python
class MyModelAdapter(Adapter):
    name = "MyModel"
    modalities = ["image", "text"]
    
    def load(self, config):
        # Load your model
        pass
    
    def prepare(self, sample):
        # Prepare input
        pass
    
    def infer(self, prepared):
        # Run inference
        pass
    
    def postprocess(self, raw):
        # Format output
        pass
```

3. **Add to your suite**:
```yaml
adapter_config:
  model_type: "my_model"
  # your config here
```

Each adapter must **emit timestamps** for TTFT/E2E and **optionally stream** tokens or bytes to compute TPS.

---

## Probes (agent)

- `latency_probe.py` — wall‑clock staging hooks around `prepare/infer/postprocess` with context managers
- `power_probe.py` — Local power monitoring + edge options (NVML, `tegrastats`, RAPL)
- `util_probe.py` — GPU util/mem via NVML; CPU util via psutil; temps via lm‑sensors/Jetson
- `memory_probe.py` — peak VRAM/RAM during trial

> Probes run in a side thread/process and sync via run‑scoped UUIDs so that measurements align with each sample.

---

## Metrics

- **Deployability** (`metrics/deployability.py`):
  - `ttft_ms`, `e2e_ms`, `tps`, `fps`, `rtf`
  - `peak_mem_mb`, `gpu_util_pct`, `cpu_util_pct`, `power_w_avg`, `temp_c_max`
- **Quality** (`metrics/quality.py`):
  - Text/VQA: `exact_match`, `BLEU`, `ROUGE`
  - Image/Video: `top1/top5`, `PSNR`, `SSIM`, `FVD` (if gen video present)
  - Audio: `WER` (ASR), MOS‑proxy or SNR (TTS/denoise)

You can mix metrics per task in suite YAML.

---

## Orchestrator flow

1. **Parse suite** → resolve datasets from `datasets/registry.yaml`.
2. **Warmup** per adapter (configurable) to stabilize caches and power state.
3. **Trial scheduling** with `repeats` × samples (optionally shuffled/limited).
4. **Local execution**: agent runs adapter + probes on your machine.
5. **Collect** per‑sample JSONL and any artifacts.
6. **Score** quality + deployability.
7. **Report**: HTML/CSV with filters and comparison tables.

---

## Result structure
results/<run_id>/
raw/ # JSONL logs (1 line/sample) + probe streams
artifacts/ # generated audio/video, overlays
scores/
per_task.csv
summary.csv
report/
index.html
assets/


Per‑sample JSONL schema (simplified):

```json
{
  "run_id": "2025-08-13T20-10-12Z",
  "task_id": "asr-librispeech",
  "adapter": "asr_tts",
  "input": {"id":"utt_7","audio":"/path.wav"},
  "output": {"text":"hello world","words":[...]},
  "timing": {"t_start":..., "t_first":..., "t_end":..., "stages":{"prepare_ms":..., "infer_ms":...}},
  "resources": {"peak_mem_mb":..., "gpu_util_pct":..., "power_w": [...], "temp_c": [...]},
  "refs": {"text":"reference transcript"}
}
```

---

## CLI

```bash
# list suites
python -m orchestrator.cli list

# test locally (recommended first step)
python -m orchestrator.cli test

# run a local suite
python -m orchestrator.cli run suites/local_test.yaml

# run with Hugging Face models
python -m orchestrator.cli run suites/hf_vlm_test.yaml

# run on edge hardware (when ready)
python -m orchestrator.cli run suites/edge_deployment.yaml

# build report only
python -m orchestrator.cli report --from results/<run_id>
```

---

## Local Development Workflow

### **1. Start Local**
```bash
# Test the system works
python -m orchestrator.cli test

# Run a simple local suite
python -m orchestrator.cli run suites/local_test.yaml
```

### **2. Add Your Model**
```bash
# Create your adapter
cp agent/adapters/base.py agent/adapters/my_vlm.py
# Edit my_vlm.py with your model

# Test locally
python -m orchestrator.cli run suites/my_vlm_test.yaml
```

### **3. Deploy to Edge (Optional)**
```bash
# When ready for edge hardware
python -m orchestrator.cli run suites/edge_deployment.yaml
```

---

## Notes on metrics & definitions

- **TTFT**: time until first token/byte/logit from request start; lower is more interactive.
- **TPS / inter‑token latency**: steady‑state generation rate after first output.
- **RTF (ASR/TTS)**: processing time divided by audio duration (real‑time if ≤ 1.0).
- **FPS (video/world)**: generation or processing frames/sec end‑to‑end.

See `docs/metrics.md` for precise formulas and references.

---

## Roadmap

- **Phase 1**: Local development and testing ✅
- **Phase 2**: Hugging Face model integration ✅
- **Phase 3**: Edge hardware deployment
- **Phase 4**: Advanced metrics and reporting
- **Phase 5**: Multi-agent distributed evaluation

---

## License

TBD.