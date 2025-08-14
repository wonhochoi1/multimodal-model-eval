from pathlib import Path
import typer, json, yaml
from typing import Optional

app = typer.Typer(help="Multimodal evaluation CLI - Local Development Focus")

@app.command()
def run(suite: Path, out: Path = typer.Option(Path("results/local_run"))):
    """Execute a suite YAML locally on your laptop/macOS"""
    out.mkdir(parents=True, exist_ok=True)
    
    with open(suite) as f: 
        cfg = yaml.safe_load(f)
    
    # Use local agent runner
    from agent.runner import LocalAgentRunner
    from agent.adapters.vlm_example import AdapterImpl
    
    # Create adapter and runner
    adapter = AdapterImpl()
    runner = LocalAgentRunner(adapter)
    
    # Run the suite locally
    results = runner.run_suite(cfg, out)
    
    typer.echo(f"✅ Local run completed! Results in {out}")

@app.command()
def test():
    """Run a quick local test to verify everything works"""
    typer.echo(" Running local test...")
    
    # Create a minimal test suite
    test_suite = {
        "name": "local_test",
        "device": "local",
        "run": {"warmup": 1, "repeats": 2},
        "tasks": [
            {
                "id": "test_vlm",
                "type": "vlm_caption",
                "prompt": "Test prompt"
            }
        ]
    }
    
    # Run it
    from agent.runner import LocalAgentRunner
    from agent.adapters.vlm_example import AdapterImpl
    
    adapter = AdapterImpl()
    runner = LocalAgentRunner(adapter)
    
    test_dir = Path("results/test_run")
    results = runner.run_suite(test_suite, test_dir)
    
    typer.echo(f"✅ Test completed! Check {test_dir}")

@app.command()
def list():
    """List available test suites"""
    suites_dir = Path("suites")
    if suites_dir.exists():
        for suite_file in suites_dir.glob("*.yaml"):
            typer.echo(f"  {suite_file.name}")
    else:
        typer.echo("No suites directory found")

if __name__ == "__main__":
    app()
