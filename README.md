# multimodal-model-eval
Optimization Evaluation Metrics and Tool for Multimodal DL Models

**Evaluate Optimization of VLMs and multimodal models (audio, image, video) on your local laptop/macOS or edge devices** with a consistent harness that measures **quality** and **deployability** (latency, throughput, memory, power, thermals). Works as a two-part system:

- **Orchestrator** (your laptop/server): schedules runs, collects logs, computes metrics, and builds reports.
- **Agent** (local laptop/edge box): runs the model adapter, streams telemetry (power/util/temps), and returns predictions + raw timings.


## Quick Start

### Option 1: Jupyter Notebook
```bash
jupyter notebook eval.ipynb
```

### Option 2: CLI
```bash
# Install dependencies
pip install -r requirements.txt

# Run a test suite (Uses Microsoft's DiabloGPT)
python -m orchestrator.cli run suites/huggingface_test.yaml

# List available suites
python -m orchestrator.cli list

# View recent results
python -m orchestrator.cli results
```


## Architecture

- **Orchestrator** (your laptop/server): schedules runs, collects logs, computes metrics, builds reports
- **Agent** (local laptop/edge box): runs model adapters, streams telemetry, returns predictions + timing
- **Probes**: collect latency, memory, power, utilization metrics
- **Adapters**: interface with different model types (HuggingFace, custom, etc.)

## Implementation Status

###**Implemented**
- Core schemas and data models
- Local agent runner with real hardware probes
- HuggingFace VLM adapter with automatic model detection
- Latency, memory, power, and utilization probes
- Quality and deployability metrics computation
- CLI with suite execution
- Jupyter notebook interface
- Results storage and organization

### TO-DO
- Edge device deployment (Jetson, etc.)
- Remote agent communication
- Advanced video/audio processing
- Custom model format support
- Distributed evaluation
- Advanced reporting and visualization

## Supported Models

- **VLM Models**: DeepSeek, LLaVA, and similar architectures
- **Text Models**: GPT-2, LLaMA, and other causal language models
- **Custom Models**: Custom models can be supported as long as adapter is implemented accordingly. 

## Configuration

Create YAML suite files in `suites/` directory:

```yaml
name: "my_evaluation"
device: "local"  # or "cuda", "mps"
run:
  warmup: 3
  repeats: 5
tasks:
  - id: "vlm_test"
    type: "vlm_caption"
    adapter_config:
      model_name: "deepseek-ai/deepseek-vl-7b-base"
      device: "cuda"
      precision: "bf16"
```

## Results

Results are saved to `results/<suite_name>_<timestamp>/` with:
- Raw trial data (JSONL)
- Computed metrics (JSON)
- Summary statistics

## Development

- **Local Testing**: System works on local laptop/macOS for development
- **Hardware Testing**: Real hardware deployment is future enhancement
- **Extensible**: Add new adapters in `agent/adapters/`

## Requirements

See `requirements.txt` for dependencies