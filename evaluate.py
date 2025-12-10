import argparse
import torch
import pandas as pd
import os
from datasets import load_dataset
from tqdm import tqdm
import torchaudio

# Import your components
from src.generator import MusicGenerator
from src.verifier import TheCurator
from src.search import InferenceSearch

def save_audio(wav, sr, path):
    """Helper to save audio tensors"""
    if wav.dim() == 3: wav = wav.squeeze(0)
    wav = wav.cpu()
    torchaudio.save(path, wav, sr)

def evaluate_dataset(args):
    # 1. Load Dataset (MusicCaps)
    print("Loading MusicCaps dataset...")
    # 'google/musiccaps' is the official repo on Hugging Face
    dataset = load_dataset("google/musiccaps", split="train")
    
    # Select a subset for testing (Inference is slow!)
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))
    
    print(f"Evaluating on {len(dataset)} samples.")

    # 2. Initialize Models
    # Generator (Client-side Wrapper)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    # Verifiers (We need both usually for a balanced report, 
    # but here we initialize the one used for optimization)
    # Ideally, you want to log BOTH Quality and Semantic scores for all outputs.
    # For simplicity, we assume the server is running and we check the optimized metric.
    verifier = TheCurator(mode=args.verifier, server_url=args.server)
    
    search_engine = InferenceSearch(generator, verifier)

    results = []
    output_dir = f"evaluation_results/{args.method}_{args.verifier}"

    # 3. Evaluation Loop
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = item['caption']
        youtube_id = item['ytid']
        
        print(f"\nProcessing {i}: {prompt[:50]}...")

        # --- A. Baseline (Vanilla MusicGen) ---
        # Equivalent to generating 1 sample without search
        baseline_wav = generator.generate_full(prompt, args.duration)
        baseline_score = verifier.score(baseline_wav, generator.sample_rate, prompt)
        
        save_audio(baseline_wav, generator.sample_rate, f"{output_dir}/{i}_baseline.wav")

        # --- B. The Curator (Search Method) ---
        if args.method == "best_of_n":
            curator_wav = search_engine.best_of_n(prompt, n_candidates=args.candidates, duration=args.duration)
        elif args.method == "sbs":
            curator_wav = search_engine.stepwise_beam_search(prompt, total_duration=args.duration)
        
        curator_score = verifier.score(curator_wav, generator.sample_rate, prompt)
        save_audio(curator_wav, generator.sample_rate, f"{output_dir}/{i}_curator.wav")

        # Log Data
        results.append({
            "id": youtube_id,
            "prompt": prompt,
            "baseline_score": baseline_score,
            "curator_score": curator_score,
            "improvement": curator_score - baseline_score
        })
        
        # Save intermediate results
        pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)

    # 4. Final Summary
    df = pd.DataFrame(results)
    print("\n--- Evaluation Complete ---")
    print(f"Average Baseline Score: {df['baseline_score'].mean():.4f}")
    print(f"Average Curator Score:  {df['curator_score'].mean():.4f}")
    print(f"Average Improvement:    {df['improvement'].mean():.4f}")
    print(f"Detailed results saved to {output_dir}/results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of songs to generate")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--method", choices=["best_of_n", "sbs"], default="best_of_n")
    parser.add_argument("--candidates", type=int, default=4, help="N for Best-of-N")
    parser.add_argument("--verifier", choices=["quality", "semantic"], default="semantic")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    
    args = parser.parse_args()
    evaluate_dataset(args)