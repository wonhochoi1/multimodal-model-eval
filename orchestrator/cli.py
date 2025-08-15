from pathlib import Path
import typer, json, yaml
from typing import Optional
from datetime import datetime
import re

app = typer.Typer(help="Multimodal evaluation CLI - Local Development Focus")

def create_unique_output_dir(suite_name: str, base_dir: Path = Path("results")) -> Path:
    """Create a unique output directory for each suite run"""

    dir_name = re.sub(r'[^a-zA-Z0-9_-]', '_', suite_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dir_name = f"{dir_name}_{timestamp}"
    output_dir = base_dir / dir_name
    
    counter = 1
    original_dir = output_dir
    while output_dir.exists():
        output_dir = base_dir / f"{dir_name}_{counter:02d}"
        counter += 1
    
    return output_dir

@app.command()
def run(suite: Path, out: Optional[Path] = typer.Option(None, help="Custom output directory")):
    """Execute a suite YAML locally on your laptop/macOS"""
    
    with open(suite) as f: 
        cfg = yaml.safe_load(f)
    
    # new output directory if not specified
    if out is None:
        suite_name = cfg.get("name", suite.stem)
        out = create_unique_output_dir(suite_name)
    
    out.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Running suite: {cfg.get('name', suite.stem)}")
    
    # local agent runner
    from agent.runner import LocalAgentRunner
    from agent.adapters.registry import get_adapter
    
    adapter = get_adapter(cfg)
    runner = LocalAgentRunner(adapter)
    
    results = runner.run_suite(cfg, out)


@app.command()
def test():
    """Run a quick local test to verify everything works"""
    typer.echo("�� Running local test...")
    
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
    from agent.adapters.registry import get_adapter
    
    adapter = get_adapter(test_suite)
    runner = LocalAgentRunner(adapter)
    
    # Create unique test directory
    test_dir = create_unique_output_dir("test_run")
    results = runner.run_suite(test_suite, test_dir)

@app.command()
def list():
    """List available test suites"""
    suites_dir = Path("suites")
    if suites_dir.exists():
        typer.echo("Suites:")
        for suite_file in suites_dir.glob("*.yaml"):
            typer.echo(f"  {suite_file.name}")
    else:
        typer.echo("No suites directory found")

@app.command()
def results():
    """List recent results directories"""
    results_dir = Path("results")
    if not results_dir.exists():
        typer.echo("No results directory found")
        return
    
    result_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            result_dirs.append((item, item.stat().st_mtime))
    
    if not result_dirs:
        typer.echo("No results found")
        return

    result_dirs.sort(key=lambda x: x[1], reverse=True)
    
    typer.echo("Recent results:")
    for i, (dir_path, mtime) in enumerate(result_dirs[:10]):  # last 10
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        typer.echo(f"  {i+1:2d}. {dir_path.name} ({timestamp})")
    
    if len(result_dirs) > 10:
        typer.echo(f"  ... and {len(result_dirs) - 10} more")

@app.command()
def clean():
    """Clean old results directories (keep last 5)"""
    results_dir = Path("results")
    if not results_dir.exists():
        typer.echo("No results directory found")
        return
    
    result_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            result_dirs.append((item, item.stat().st_mtime))
    
    if not result_dirs:
        typer.echo("No results found")
        return
    
    result_dirs.sort(key=lambda x: x[1], reverse=True)
    
    to_keep = result_dirs[:5]
    to_remove = result_dirs[5:]
    
    if not to_remove:
        typer.echo("No old results")
        return
    
    typer.echo(f"Cleaning {len(to_remove)} old result directories...")
    
    for dir_path, _ in to_remove:
        try:
            import shutil
            shutil.rmtree(dir_path)
            typer.echo(f"  Removed: {dir_path.name}")
        except Exception as e:
            typer.echo(f"  Failed to remove {dir_path.name}: {e}")


if __name__ == "__main__":
    app()