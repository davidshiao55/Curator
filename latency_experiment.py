import time
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset

from src.generator import MusicGenerator
from src.verifier import TheCurator
from src.search import InferenceSearch

def measure_time(func, *args, **kwargs):
    """Measures execution time, ensuring CUDA synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    result = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    return result, end - start

def plot_latency(df):
    """Generates a comparison graph for latency with Cost Annotations."""
    print("\nGenerating latency plot...")
    
    # Filter out Baseline for the main scaling plot (it's N=1)
    plot_df = df[df['Method'].str.contains("N=")].copy()
    
    # Extract raw N for X-axis (Budget)
    # We want to plot them against the "Base Budget N", but annotate the true cost
    
    # Create descriptive labels for the legend
    def get_legend_label(row):
        method = row['Method']
        if "Best-of" in method:
            return "Best-of-N (Cost: N)"
        elif "SBS" in method:
            # Beam search cost is roughly N per step
            return "Beam Search (Cost: N)"
        elif "Lookahead" in method:
            # Extract k from string "Lookahead (N=..., k=...)"
            try:
                k_part = method.split("k=")[1].split(")")[0]
                k = int(k_part)
                # Paper definition: N * (k + 1)
                return f"Lookahead (Cost: N $\\times$ (k+1))"
            except:
                return "Lookahead Search"
        return method

    plot_df['Algorithm'] = plot_df.apply(get_legend_label, axis=1)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot
    sns.lineplot(
        data=plot_df,
        x="Budget (N)",
        y="Avg Time (s)",
        hue="Algorithm",
        style="Algorithm",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="viridis"
    )
    
    # Add Baseline line
    baseline_row = df[df['Method'].str.contains("Baseline")]
    if not baseline_row.empty:
        baseline_time = baseline_row['Avg Time (s)'].values[0]
        plt.axhline(y=baseline_time, color='r', linestyle='--', label=f"Baseline (N=1): {baseline_time:.2f}s")
    
    plt.title("Inference Latency by Compute Budget (N)", fontsize=16)
    plt.xlabel("Generation Budget N (Batch Size)", fontsize=12)
    plt.ylabel("Time per Sample (seconds)", fontsize=12)
    
    # Ensure x-axis ticks match the budgets used
    budgets = sorted(plot_df['Budget (N)'].unique())
    plt.xticks(budgets)
    
    # Move legend to prevent covering data
    plt.legend(title="Search Method & Effective Cost", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("latency_comparison.png", dpi=300)
    print("Saved 'latency_comparison.png'")

def run_latency_test(args):
    print(f"\n{'='*60}")
    print(f"COMPUTE BUDGET & LATENCY TEST")
    print(f"Samples: {args.num_samples} | Duration: {args.duration}s")
    print(f"Verifier: {args.verifier} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*60}\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize
    print("Initializing Generator...")
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    print(f"Connecting Verifier ({args.verifier})...")
    verifier = TheCurator(mode=args.verifier, server_url=args.server, generator=generator)
    search_engine = InferenceSearch(generator, verifier)

    # 2. Load Prompts
    print("Loading MusicCaps prompts...")
    try:
        dataset = load_dataset("google/musiccaps", split="train")
        if len(dataset) < args.num_samples:
            indices = [i % len(dataset) for i in range(args.num_samples)]
            dataset = dataset.select(indices)
        else:
            dataset = dataset.select(range(args.num_samples))
        prompts = [item['caption'] for item in dataset]
    except Exception:
        print("[Warning] Using dummy prompts.")
        prompts = ["A funky bassline with drums"] * args.num_samples

    # 3. Robust Warm-up
    print("Warming up GPU & Verifier (Preventing first-run penalty)...")
    generator.generate_batch(["warmup"], duration=args.duration)
    dummy_audio = torch.randn(1, 1, int(32000 * args.duration)).to(device)
    verifier.score_batch(dummy_audio, 32000, ["warmup"])
    print("Warmup complete.\n")
    
    # 4. Define Experiments
    # Note: We group experiments by N to ensure the loop runs logically
    budgets = [4, 8, 16, 32] 
    configs = []
    
    # Baseline
    configs.append({
        "name": "Baseline (N=1)",
        "budget": 1,
        "func": lambda prompt: generator.generate_batch([prompt], args.duration),
        "params": {}
    })

    for N in budgets:
        # A. Parallel Sampling (Best-of-N)
        configs.append({
            "name": f"Best-of-{N} (N={N})",
            "budget": N,
            "func": search_engine.best_of_n,
            "params": {
                "n_candidates": N, 
                "duration": args.duration
            }
        })
        
        # B. Beam Search (SBS)
        configs.append({
            "name": f"SBS (N={N}, M={args.beam_width})",
            "budget": N,
            "func": search_engine.beam_search,
            "params": {
                "total_duration": args.duration,
                "step_size": 2,
                "generation_budget": N, 
                "beam_width": args.beam_width
            }
        })

        # C. Lookahead Search
        # Effective Cost = N * (k+1)
        configs.append({
            "name": f"Lookahead (N={N}, k={args.lookahead_k})",
            "budget": N,
            "func": search_engine.lookahead_search,
            "params": {
                "total_duration": args.duration,
                "step_size": 2,
                "generation_budget": N, 
                "beam_width": args.beam_width,
                "lookahead_k": args.lookahead_k
            }
        })

    results = []

    # 5. Run Benchmark
    for config in configs:
        print(f"\nTesting: {config['name']}...")
        times = []
        
        for i in tqdm(range(args.num_samples)):
            _, elapsed = measure_time(
                config['func'], 
                prompt=prompts[i], 
                **config['params']
            )
            times.append(elapsed)
        
        avg_time = np.mean(times)
        
        results.append({
            "Method": config['name'],
            "Budget (N)": config['budget'],
            "Avg Time (s)": round(avg_time, 2),
            "Samples/Sec": round(config['budget'] / avg_time, 2)
        })

    # 6. Report
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("FINAL LATENCY REPORT")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save Data
    df.to_csv("latency_results.csv", index=False)
    print("\nSaved CSV to latency_results.csv")
    
    # Plot Graph
    try:
        plot_latency(df)
    except Exception as e:
        print(f"[!] Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5, help="Number of repetitions")
    parser.add_argument("--duration", type=int, default=10, help="Generation duration")
    parser.add_argument("--verifier", default="perplexity", help="Verifier to use")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam Width (M)")
    parser.add_argument("--lookahead_k", type=int, default=1, help="Lookahead Steps (k)")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    
    args = parser.parse_args()
    run_latency_test(args)