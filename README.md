# The Curator: Inference-Time Search for Music Generation

**The Curator** is a research project exploring **Test-Time Compute** for music generation. Instead of training larger models, it optimizes the output of a frozen autoregressive model (MusicGen) by treating generation as a search problem.

Aligned with the DeepMind paper *"Scaling LLM Test-Time Compute Optimally"*, this repository implements strategies to exchange inference-time compute for better audio quality and semantic alignment.

## Core Search Algorithms

The engine implements three search strategies, all unified under a **Generation Budget ($N$)** framework to ensure fair comparisons:

1.  **Best-of-N (Parallel Sampling):**
    * Generates $N$ complete clips in parallel.
    * Selects the single best clip based on the verifier score.
    * *Cost:* $N$ samples.

2.  **Beam Search (BFS-V):**
    * A tree-search algorithm that maintains a fixed population size of $N$ at every step.
    * **Logic:** Prune current candidates to the top $K = N/M$, then expand each parent by $M$ (Beam Width).
    * *Cost:* $\approx N$ samples per step (comparable to Best-of-N).

3.  **Lookahead Search (MCTS-Lite):**
    * Extends Beam Search. Instead of scoring the *current* partial audio, it performs a **greedy rollout** for $k$ steps into the future.
    * The score of the future state is used to rank the current candidate.
    * *Cost:* $N \times (k+1)$ samples (more expensive, but "smarter").

---

## Architecture: Client-Server

To resolve dependency conflicts between **MusicGen** (requires older Torch) and modern verifiers like **Audiobox/CLAP** (require newer Torch), this project uses a **Client-Server architecture**:

* **The Verifier (Server):** Hosts heavy scoring models (Audiobox, CLAP, MuQ, ImageBind). Runs on **PyTorch 2.4+**.
* **The Search (Client):** Hosts the generator (MusicGen) and search algorithms. Runs on **PyTorch 2.1** (or compatible stable version).

## Project Structure

```text
Curator/
├── benchmark.py            # Main evaluation script (MusicCaps benchmark)
├── latency_experiment.py   # Latency & Overhead testing (Fair budget analysis)
├── verifier_experiment.py  # Scaling Law experiments (N=1..8 sweeps)
├── verifier_server.py      # Server Entry Point (Hosts CLAP, MuQ, etc.)
├── download_data.py        # Utility to download MusicCaps reference audio
└── src/
    ├── generator.py        # MusicGen Wrapper
    ├── search.py           # Implementation of Best-of-N, Beam, Lookahead
    ├── verifier.py         # Client-side HTTP Wrapper
    └── verifier_core.py    # Server-side Inference Logic
````

-----

## Installation

We recommend running in a Docker container or using Conda. You must set up **two separate virtual environments**.

### 1\. Environment A: The Generator (Client)

This environment runs MusicGen and handles the search logic.

```bash
# Generator
conda create -n generator python=3.9
conda activate generator

# Install audiocraft
apt update
apt install -y build-essential gcc g++ python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
conda install "ffmpeg<5" -c conda-forge
conda install -c conda-forge libiconv
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -U audiocraft  # stable release
pip install "transformers==4.37.2" --force-reinstall
pip install "numpy<2" --force-reinstall
pip install datasets pandas

# for running benchmark
pip install git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt
pip install laion-clap
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
pip install matplotlib seaborn
```

### 2\. Environment B: The Verifier (Server)

This environment runs the scoring models.

```bash
# Verifier
conda create -n verifier python=3.9

# Install audiobox-aesthetics
pip install audiobox_aesthetics
pip install laion-clap
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install muq
pip install git+https://github.com/facebookresearch/ImageBind.git
```

### 3\. Utility: Data Downloader

```bash
conda create -n downloader python=3.10
conda activate downloader
conda install -c conda-forge ffmpeg
pip install yt-dlp datasets pandas torchaudio tqdm
```

-----

## Usage

### Step 1: Start the Verifier Server

**Terminal 1 (`conda activate verifier`)**

```bash
python verifier_server.py
```

*Wait until you see: `Uvicorn running on http://0.0.0.0:8000`*

> **Note:** To save VRAM, you can comment out unused models (like ImageBind or MuQ) in `verifier_server.py`.

### Step 2: Download Reference Data

**Terminal 2 (`conda activate downloader`)**

Before running experiments, you must download the MusicCaps reference audio (ground truth) from YouTube. This is required to calculate metrics like KLD.

```bash
python download_data.py
```

*This will download audio files into the `music_data/` directory.*

### Step 3: Run Experiments

**Terminal 2 (`conda activate generator`)**

#### A. Standard Benchmark

Evaluate a specific method on the MusicCaps dataset.

```bash
# Best-of-16 using CLAP verifier
python benchmark.py --method best_of_n --candidates 16 --verifier clap

# Beam Search (Budget N=16, Width M=4) using Quality verifier
python benchmark.py --method sbs --candidates 16 --beam_width 4 --verifier quality

# Lookahead Search (N=16, M=4, k=2)
python benchmark.py --method lookahead --candidates 16 --beam_width 4 --lookahead_k 2 --verifier clap
```

#### B. Scaling Law Experiment (`verifier_experiment.py`)

This script automatically sweeps budgets $N \in [1, 2, 4, 8]$ across multiple verifiers to see which metric scales best.

```bash
python verifier_experiment.py --samples 10 --duration 10
```

#### C. Latency & Compute Fairness (`latency_experiment.py`)

Measures the wall-clock time of different methods to verify if budgets are comparable.

```bash
python latency_experiment.py --num_samples 5 --verifier perplexity
```
-----

## Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--method` | `baseline` | Search strategy: `baseline`, `best_of_n`, `sbs` (Beam), `lookahead`. |
| `--verifier` | `perplexity` | Metric to optimize: `clap`, `quality`, `muq`, `imagebind`, `perplexity`. |
| `--candidates`| 16 | **Generation Budget ($N$)**. The total number of candidates processed per step. |
| `--beam_width`| 4 | **Expansion Factor ($M$)**. Number of children generated per parent in Beam/Lookahead. |
| `--lookahead_k`| 1 | **Simulation Steps ($k$)**. How far to simulate into the future for Lookahead Search. |
| `--duration` | 10 | Audio duration in seconds. |