"""
Lightweight Streamlit front-end for the CleanRL scripts in this repo.

Features:
- Verifies core runtime dependencies (torch, gym, gymnasium, wandb).
- Lists available algorithm scripts under ./cleanrl and shows a readable preview.
- Generates ready-to-run CLI commands for short demo runs (CPU-friendly).

This app avoids long-running training jobs by default; it only prints commands.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from textwrap import dedent

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "cleanrl"


def check_module(name: str):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except Exception as exc:  # pragma: no cover - user feedback path
        return False, str(exc)


def list_algorithms():
    if not SCRIPTS_DIR.exists():
        return []
    return sorted(SCRIPTS_DIR.glob("*.py"))


def summarize_script(path: Path, max_lines: int = 24) -> str:
    try:
        content = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "Could not read file."
    snippet = content[:max_lines]
    return "\n".join(snippet)


def build_cli(algo_file: Path, env_id: str, timesteps: int, seed: int, log_dir: str):
    return dedent(
        f"""
        # Minimal demo command (CPU-friendly)
        python {algo_file.as_posix()} \\
          --env-id {env_id} \\
          --total-timesteps {timesteps} \\
          --seed {seed} \\
          --track False \\
          --tensorboard-logdir {log_dir}
        """
    ).strip()


st.set_page_config(
    page_title="CleanRL — Streamlit Frontend",
    page_icon="🤖",
    layout="wide",
)

st.title("CleanRL Streamlit Frontend")
st.caption(
    "Pick an algorithm script, review its code quickly, and grab a safe CLI command "
    "you can run in Streamlit Cloud or locally."
)

# Dependency checks
st.subheader("Environment quick-check")
cols = st.columns(4)
for col, module in zip(cols, ["torch", "gym", "gymnasium", "wandb"]):
    ok, info = check_module(module)
    with col:
        if ok:
            st.success(f"{module} {info}")
        else:
            st.error(f"{module} missing")
            st.code(info, language="text")

st.divider()

# Sidebar controls
st.sidebar.header("Demo settings")
algos = list_algorithms()
if not algos:
    st.sidebar.error("No algorithm scripts found under ./cleanrl")
    st.stop()

selected = st.sidebar.selectbox(
    "Algorithm script", options=algos, format_func=lambda p: p.name
)
env_id = st.sidebar.text_input("Gym env id", value="CartPole-v1")
timesteps = st.sidebar.slider("Total timesteps", 1_000, 200_000, 10_000, step=1_000)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=1, step=1)
log_dir = st.sidebar.text_input("Log dir", value="runs/streamlit-demo")

st.sidebar.info(
    "Click “Generate command” to copy/paste into a terminal. "
    "Streamlit Cloud will install CPU wheels via requirements.txt."
)

# Main content
left, right = st.columns([1.1, 1.0])
with left:
    st.subheader("Script preview")
    st.code(summarize_script(selected), language="python")

with right:
    st.subheader("Run it yourself")
    cli = build_cli(selected, env_id, timesteps, seed, log_dir)
    st.code(cli, language="bash")
    st.caption(
        "The demo command keeps timesteps low for quick, CPU-only runs. "
        "Remove `--track False` to enable Weights & Biases tracking."
    )

st.divider()

st.markdown(
    dedent(
        """
        ### Deployment notes
        - Python 3.10 is pinned via `python-version` for Streamlit Cloud.
        - `requirements.txt` uses the PyTorch CPU wheel index to avoid GPU drivers.
        - Training jobs can be heavy; start with small timesteps inside Streamlit
          or run long jobs offline and upload artifacts to visualize.
        """
    )
)
