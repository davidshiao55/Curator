import argparse
import os
import shutil
import torch
import torchaudio
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import torchaudio.functional as F

from src.generator import MusicGenerator
from src.verifier import TheCurator
from src.search import InferenceSearch

# Metrics (AudioCraft)
from audiocraft.metrics import PasstKLDivergenceMetric, CLAPTextConsistencyMetric

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Seed set to {seed} ---")

def save_audio(wav, sr, path):
    wav = wav.detach().cpu()
    if wav.dim() == 3: wav = wav.squeeze(0)
    torchaudio.save(path, wav, sr)

def match_audio(wav, target_len, target_sr, orig_sr):
    """
    Standardize reference audio to match generated audio format.
    """
    if orig_sr != target_sr:
        wav = F.resample(wav, orig_sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    current_samples = wav.shape[-1]
    if current_samples > target_len:
        wav = wav[..., :target_len]
    elif current_samples < target_len:
        padding = target_len - current_samples
        wav = torch.nn.functional.pad(wav, (0, padding))
    return wav

def run_benchmark(args):
    set_seed(args.seed)

    print(f"--- 1. Setup & Configuration ---")
    print(f"Method:   {args.method.upper()}")
    if args.method != "baseline":
        print(f"Verifier: {args.verifier.upper()}")
        print(f"Budget (N): {args.candidates}")
        if args.method in ["sbs", "lookahead"]:
            print(f"Beam Width (M): {args.beam_width}")
            print(f"Step Size: {args.step_size}s")
        if args.method == "lookahead":
            print(f"Lookahead (k): {args.lookahead_k}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    base_dir = os.path.abspath(f"benchmark_results/{args.method}_{args.verifier}")
    ref_dir = os.path.join(base_dir, "reference")
    gen_dir = os.path.join(base_dir, "generated")

    if os.path.exists(base_dir):
        # Only clean if running a fresh batch, but appending is handled by outer script logic usually.
        # For safety in this specific loop structure, we clean.
        shutil.rmtree(base_dir)
    
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    # Load Dataset
    print("Loading MusicCaps dataset...")
    try:
        dataset = load_dataset("google/musiccaps", split="train")
        if args.num_samples > 0:
            dataset = dataset.select(range(args.num_samples))
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return

    # Initialize Generator
    print("Initializing Generator (MusicGen)...")
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    # Initialize Search Engine
    search_engine = None
    if args.method in ["best_of_n", "sbs", "lookahead"]:
        print(f"Initializing TheCurator ({args.verifier})...")
        verifier = TheCurator(mode=args.verifier, server_url=args.server, generator=generator)
        search_engine = InferenceSearch(generator, verifier)

    # Metrics
    print("Loading Evaluation Metrics...")
    clap_checkpoint = 'music_audioset_epoch_15_esc_90.14.pt'
    if not os.path.exists(clap_checkpoint):
        os.system(f"wget https://huggingface.co/lukewys/laion_clap/resolve/main/{clap_checkpoint}")

    clap_metric = CLAPTextConsistencyMetric(
        model_path=clap_checkpoint,
        model_arch='HTSAT-base',
        enable_fusion=False
    )
    clap_metric.to(device)

    try:
        kld_metric = PasstKLDivergenceMetric(pretrained_length=10)
        kld_metric.to(device)
    except Exception as e:
        print(f"[Warning] KLD Metric disabled: {e}")
        kld_metric = None

    print(f"--- 2. Evaluation Loop ({len(dataset)} samples) ---")
    valid_ref_count = 0

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = item['caption']
        ytid = item['ytid']
        
        gen_wav = None
        
        if args.method == "baseline":
            gen_wav = generator.generate_batch([prompt], args.duration)[0]
            
        elif args.method == "best_of_n":
            gen_wav = search_engine.best_of_n(
                prompt, 
                n_candidates=args.candidates, 
                duration=args.duration
            )
            
        elif args.method == "sbs":
            gen_wav = search_engine.beam_search(
                prompt, 
                total_duration=args.duration, 
                step_size=args.step_size, # Passed from args
                generation_budget=args.candidates, 
                beam_width=args.beam_width
            )

        elif args.method == "lookahead":
            gen_wav = search_engine.lookahead_search(
                prompt, 
                total_duration=args.duration, 
                step_size=args.step_size, # Passed from args
                generation_budget=args.candidates, 
                beam_width=args.beam_width,
                lookahead_k=args.lookahead_k
            )

        # Standardize
        if gen_wav.dim() == 3: gen_wav = gen_wav.squeeze(0)
        if gen_wav.shape[0] > 1: gen_wav = gen_wav.mean(dim=0, keepdim=True)
        if gen_wav.dim() == 1: gen_wav = gen_wav.unsqueeze(0)

        target_len = gen_wav.shape[-1]
        save_audio(gen_wav, generator.sample_rate, f"{gen_dir}/{i}.wav")
        gen_batch = gen_wav.unsqueeze(0).to(device)

        # Reference
        ref_path = os.path.join(args.data_dir, f"{ytid}.wav")
        ref_batch = None
        if os.path.exists(ref_path):
            try:
                ref_wav, ref_sr = torchaudio.load(ref_path)
                ref_wav = match_audio(ref_wav, target_len, generator.sample_rate, ref_sr)
                save_audio(ref_wav, generator.sample_rate, f"{ref_dir}/{i}.wav")
                ref_batch = ref_wav.unsqueeze(0).to(device)
                valid_ref_count += 1
            except: pass

        # Update Metrics
        sizes = torch.tensor([target_len]).to(device)
        sample_rates = torch.tensor([generator.sample_rate]).to(device)
        clap_metric.update(gen_batch, [prompt], sizes, sample_rates)
        if kld_metric and ref_batch is not None:
            if gen_batch.shape == ref_batch.shape:
                kld_metric.update(gen_batch, ref_batch, sizes, sample_rates)

    print("--- 3. Finalizing Scores ---")
    results = {
        "method": args.method,
        "verifier": args.verifier,
        "budget_N": args.candidates,
        "beam_M": args.beam_width,
        "step_size": args.step_size
    }

    clap_score = clap_metric.compute()
    print(f"CLAP Score: {clap_score:.4f}")
    results['clap'] = clap_score

    if valid_ref_count > 0 and kld_metric:
        try:
            kld_results = kld_metric.compute()
            print(f"KLD Score: {kld_results['kld_both']:.4f}")
            results['kld'] = kld_results['kld_both']
        except: results['kld'] = "Error"
    else:
        results['kld'] = "N/A"

    pd.DataFrame([results]).to_csv(f"{base_dir}/final_metrics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "best_of_n", "sbs", "lookahead"], default="baseline")
    parser.add_argument("--verifier", choices=["quality", "clap", "muq", "imagebind", "theory", "perplexity"], default="perplexity")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--step_size", type=int, default=2, help="Generation step size in seconds")
    parser.add_argument("--lookahead_k", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--data_dir", type=str, default="music_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(args)