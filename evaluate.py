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
    # Ensure 2D [Channels, Time] for torchaudio.save
    wav = wav.detach().cpu()
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    torchaudio.save(path, wav, sr)

def evaluate_dataset(args):
    print(f"--- Starting Evaluation ---")
    print(f"Optimization Metric (Search): {args.search_verifier.upper()}")
    print(f"Evaluation Metric (Report):   {args.eval_verifier.upper()}")
    
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
    
    # Generator (Shared)
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    # --- A. Search Verifier (The Guide) ---
    # Used by Best-of-N / Beam Search to select candidates
    print(f"Connecting Search Verifier ({args.search_verifier})...")
    search_verifier = TheCurator(
        mode=args.search_verifier, 
        server_url=args.server, 
        generator=generator
    )
    
    # --- B. Evaluation Verifier (The Judge) ---
    # Used only to score the final output for the report
    if args.eval_verifier == args.search_verifier:
        eval_verifier = search_verifier
    else:
        print(f"Connecting Eval Verifier ({args.eval_verifier})...")
        eval_verifier = TheCurator(
            mode=args.eval_verifier, 
            server_url=args.server, 
            generator=generator
        )
    
    # Setup Search Engine with the Search Verifier
    search_engine = InferenceSearch(generator, search_verifier)

    results = []
    output_dir = f"evaluation_results/{args.method}_opt-{args.search_verifier}_eval-{args.eval_verifier}"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Evaluation Loop
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = item['caption']
        youtube_id = item['ytid']
        
        # --- A. Baseline (Vanilla MusicGen) ---
        # Generate 1 sample
        baseline_wav_batch = generator.generate_batch([prompt], args.duration)
        baseline_wav = baseline_wav_batch[0]
        
        # Score Baseline (Using Eval Metric)
        baseline_score = eval_verifier.score_batch(
            baseline_wav_batch, 
            generator.sample_rate, 
            [prompt]
        )[0]
        
        save_audio(baseline_wav, generator.sample_rate, f"{output_dir}/{i}_baseline.wav")

        # --- B. The Curator (Search Method) ---
        if args.method == "best_of_n":
            curator_wav = search_engine.best_of_n(
                prompt, 
                n_candidates=args.candidates, 
                duration=args.duration
            )
        elif args.method == "sbs":
            curator_wav = search_engine.stepwise_beam_search(
                prompt, 
                total_duration=args.duration,
                step_size=2,       
                beam_width=args.beam_width,      
                expand_k=8         
            )
        
        # Score Curator Result (Using Eval Metric)
        curator_wav_batch = curator_wav.unsqueeze(0) # Wrap for batch scoring
        curator_score = eval_verifier.score_batch(
            curator_wav_batch, 
            generator.sample_rate, 
            [prompt]
        )[0]
        
        save_audio(curator_wav, generator.sample_rate, f"{output_dir}/{i}_curator.wav")

        # Log Data
        diff = curator_score - baseline_score
        
        results.append({
            "id": youtube_id,
            "prompt": prompt,
            "baseline_score": baseline_score,
            "curator_score": curator_score,
            "improvement": diff,
            "eval_metric": args.eval_verifier
        })
        
        # Save intermediate results
        pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)

    # 4. Final Summary
    df = pd.DataFrame(results)
    print("\n--- Evaluation Complete ---")
    print(f"Optimized For:          {args.search_verifier.upper()}")
    print(f"Evaluated On:           {args.eval_verifier.upper()}")
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
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    
    # --- Separated Verifiers ---
    # 1. Search Verifier: The metric being optimized (e.g. Perplexity)
    parser.add_argument("--search_verifier", choices=["quality", "semantic", "theory", "perplexity"], 
                        default="perplexity", help="Metric to use for Search (Optimization)")
    
    # 2. Eval Verifier: The metric used to judge the final quality (e.g. Semantic)
    parser.add_argument("--eval_verifier", choices=["quality", "semantic", "theory", "perplexity"], 
                        default="semantic", help="Metric to use for Final Evaluation")
    
    args = parser.parse_args()
    evaluate_dataset(args)