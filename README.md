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
├── evaluate.py             # Evaluation script for MusicCaps
└──  src/
    ├── generator.py        # MusicGen Wrapper (Locally computes Perplexity)
    ├── search.py           # Search Algorithms (Best-of-N, SBS)
    ├── verifier.py         # Client-side Wrapper (HTTP Client)
    └── verifier_core.py    # Server-side Logic (Audiobox/CLAP Inference)

````

-----

## Installation

Run in a Docker environment for reproducibility.

```bash
docker run -it \
  --gpus all \
  -v ./Curator:/Curator \
  --name Curator \
  continuumio/miniconda3:latest bash
```

You must set up **two separate virtual environments**.

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

### 3\. Utiliy Environment : Downloader
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

This loads the heavy models (Audiobox, CLAP) into VRAM.

```bash
python verifier_server.py
```

*Wait until you see: `Uvicorn running on http://0.0.0.0:8000`*

### Step 2: Run Music Generation

**Terminal 2 (`conda activate generator`)**

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

## Evaluation

To benchmark **The Curator** against the baseline (Vanilla MusicGen), use the `evaluate.py` script. This runs the pipeline on the **MusicCaps** dataset (or a subset) and reports the average improvement in the chosen metric.

### Run Evaluation

Ensure the Verifier Server is running (unless using `perplexity`).

```bash
# Example: Evaluate Stepwise Beam Search (SBS) optimizing Semantic Score on 10 samples
python evaluate.py --method sbs --verifier semantic --num_samples 10
```

### Analyze Results

Results are saved in `evaluation_results/<method>_<verifier>/`:

  * `results.csv`: Detailed scores for each sample (Baseline vs. Curator).
  * `*_baseline.wav`: Audio generated by vanilla MusicGen.
  * `*_curator.wav`: Audio generated with Search.

### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--prompt` | "8-bit..." | Text description for generation. |
| `--duration` | 10 | Total length of audio in seconds. |
| `--method` | `best_of_n` | Search algorithm: `best_of_n` or `sbs`. |
| `--verifier` | `quality` | Metric to optimize: `quality`, `semantic`, `theory`, `perplexity`. |
| `--candidates`| 4 | Number of candidates for Best-of-N. |
| `--beam_width`| 4 | Number of beams to keep for SBS. |
| `--server` | `http://...`| URL of the verifier server (default: localhost:8000). |



