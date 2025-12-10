import argparse
import torch
import pandas as pd
import os
import torchaudio
from tqdm import tqdm

# Requires 'datasets' library: pip install datasets
from datasets import load_dataset

from src.generator import MusicGenerator
from src.verifier import TheCurator
from src.search import InferenceSearch

def save_audio(wav, sr, path):
    """Helper to save audio tensors"""
    # Wav shape might be [1, Channels, Time] or [Channels, Time]
    if wav.dim() == 3: 
        wav = wav.squeeze(0)
    wav = wav.cpu()
    torchaudio.save(path, wav, sr)

def evaluate_dataset(args):
    print(f"--- Starting Evaluation: {args.method.upper()} vs Baseline ---")
    
    # 1. Load Dataset (MusicCaps)
    print("Loading MusicCaps dataset...")
    try:
        dataset = load_dataset("google/musiccaps", split="train")
    except Exception as e:
        print(f"[Error] Could not load dataset: {e}")
        print("Ensure you have installed datasets: pip install datasets")
        return

    # Select subset
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))
    
    print(f"Evaluating on {len(dataset)} samples.")

    # 2. Initialize Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generator
    # We use the same generator instance for both Baseline and Curator
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    # Verifier (Client)
    # Important: Pass generator for local perplexity scoring
    print(f"Connecting to Verifier ({args.verifier}) at {args.server}...")
    verifier = TheCurator(mode=args.verifier, server_url=args.server, generator=generator)
    
    # Search Engine
    search_engine = InferenceSearch(generator, verifier)

    results = []
    output_dir = f"evaluation_results/{args.method}_{args.verifier}"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Evaluation Loop
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = item['caption']
        youtube_id = item['ytid']
        
        # --- A. Baseline (Vanilla MusicGen) ---
        # Generate 1 sample using batch method (efficient)
        # Returns [1, C, T], so we take [0]
        baseline_wav_batch = generator.generate_batch([prompt], args.duration)
        baseline_wav = baseline_wav_batch[0]
        
        # Score Baseline
        # Note: score_batch expects [Batch, C, T] and List[str]
        baseline_score = verifier.score_batch(baseline_wav_batch, generator.sample_rate, [prompt])[0]
        
        save_audio(baseline_wav, generator.sample_rate, f"{output_dir}/{i}_baseline.wav")

        # --- B. The Curator (Search Method) ---
        if args.method == "best_of_n":
            curator_wav = search_engine.best_of_n(prompt, n_candidates=args.candidates, duration=args.duration)
        elif args.method == "sbs":
            curator_wav = search_engine.stepwise_beam_search(
                prompt, 
                total_duration=args.duration,
                step_size=2,       
                beam_width=args.beam_width,      
                expand_k=8         
            )
        
        # Score Curator Result
        # Wrap result in batch dim [1, C, T] for scoring
        curator_wav_batch = curator_wav.unsqueeze(0)
        curator_score = verifier.score_batch(curator_wav_batch, generator.sample_rate, [prompt])[0]
        
        save_audio(curator_wav, generator.sample_rate, f"{output_dir}/{i}_curator.wav")

        # Log Data
        diff = curator_score - baseline_score
        # print(f"  ID: {i} | Base: {baseline_score:.4f} | Curator: {curator_score:.4f} | Diff: {diff:.4f}")
        
        results.append({
            "id": youtube_id,
            "prompt": prompt,
            "baseline_score": baseline_score,
            "curator_score": curator_score,
            "improvement": diff
        })
        
        # Save intermediate results frequently
        pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)

    # 4. Final Summary
    df = pd.DataFrame(results)
    print("\n--- Evaluation Complete ---")
    print(f"Metric Evaluated:       {args.verifier.upper()}")
    print(f"Average Baseline Score: {df['baseline_score'].mean():.4f}")
    print(f"Average Curator Score:  {df['curator_score'].mean():.4f}")
    print(f"Average Improvement:    {df['improvement'].mean():.4f}")
    print(f"Results saved to:       {output_dir}/results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of songs to evaluate")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--method", choices=["best_of_n", "sbs"], default="best_of_n")
    parser.add_argument("--candidates", type=int, default=4, help="N for Best-of-N")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam Width for SBS")
    
    # Added perplexity
    parser.add_argument("--verifier", choices=["quality", "semantic", "theory", "perplexity"], default="semantic")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    
    args = parser.parse_args()
    evaluate_dataset(args)