from pathlib import Path


def get_output_dir(agent_name: str, mode: str) -> Path:
    """Return (and create) results/{agent_name}/{mode}/ directory."""
    dir_path = Path("results") / agent_name / mode
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
