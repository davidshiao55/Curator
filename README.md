# The Curator: Inference-Time Search for Music Generation

**The Curator** is a research project exploring **Test-Time Compute** for music generation. Instead of training larger models, it optimizes the output of a frozen autoregressive model (MusicGen) by treating generation as a search problem.

It implements two search algorithms:

1.  **Best-of-N (Rejection Sampling):** Generates $N$ complete clips and selects the best one.
2.  **Stepwise Beam Search (SBS):** Optimizes generation incrementally (e.g., every 2 seconds) to navigate the generation space more efficiently.

## Architecture: Client-Server

To resolve dependency conflicts between **MusicGen** (requires older Torch) and **Audiobox/CLAP** (require newer Torch), this project uses a **Client-Server architecture**:

  * **The Verifier (Server):** Hosts the heavy verifier models (Audiobox, CLAP, Librosa). Runs on **PyTorch 2.4+**.
  * **The Search (Client):** Hosts the generator (MusicGen) and search algorithms. Runs on **PyTorch 2.1** (or compatible stable version).

## Project Structure

```text
Curator/
├── main.py                 # Client Entry Point (Run this to generate music)
├── verifier_server.py      # Server Entry Point (Run this to judge music)
├── requirements_verifier.txt
├── requirements_generator.txt
├── src/
│   ├── generator.py        # MusicGen Wrapper
│   ├── search.py           # Search Algorithms (Best-of-N, SBS)
│   ├── verifier.py         # Client-side Wrapper (HTTP Client)
│   └── verifier_core.py    # Server-side Logic (Model Inference)
└── output/                 # Generated audio files
```

-----

## Installation
Run in dokcer envoriment for reproducibility.
```
docker run -it \
  --gpus all \
  -v ./Curator:/Curator \
  --name Curator \
  continuumio/miniconda3:latest bash
```
You must set up **two separate virtual environments**.

### 1\. Environment A: The Generator (Client)
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
```

### 2\. Environment B: The Verifier (Server)
```bash
# Verifier
conda create -n verifier python=3.9

# Install audiobox-aesthetics
pip install audiobox_aesthetics
# pip install --no-deps audiobox_aesthetics
pip install laion-clap
pip install fastapi
pip install uvicorn
pip install python-multipart
```

-----

## Usage

### Step 1: Start the Verifier Server

**Terminal 1 (`venv_verifier`)**

This loads the heavy models (Audiobox, CLAP) into VRAM.

```bash
python verifier_server.py
```

*Wait until you see: `Uvicorn running on http://0.0.0.0:8000`*

### Step 2: Run Music Generation

**Terminal 2 (`venv_generator`)**

You can now run experiments using the client.

#### A. Baseline: Best-of-N

Generates 4 candidate clips and picks the best one based on **Quality** (Audiobox).

```bash
python main.py --method best_of_n --candidates 4 --verifier quality --prompt "lofi hip hop beat"
```

#### B. Stepwise Beam Search (SBS)

Generates audio in 2-second chunks, keeping the top 4 beams at each step, guided by **Semantic Similarity** (CLAP).

```bash
python main.py --method sbs --duration 10 --verifier semantic --prompt "cyberpunk city rain"
```

### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--prompt` | "8-bit..." | Text description for generation. |
| `--duration` | 10 | Total length of audio in seconds. |
| `--method` | `best_of_n` | Search algorithm: `best_of_n` or `sbs`. |
| `--verifier` | `quality` | Metric to optimize: `quality`, `semantic`, `theory`. |
| `--server` | `http://...`| URL of the verifier server (default: localhost:8000). |
