import subprocess
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark_call(method, verifier, budget_n, beam_width, lookahead_k, num_samples, duration, seed):
    """
    Wraps the call to benchmark.py.
    """
    cmd = [
        "python", "benchmark.py",
        "--method", method,
        "--verifier", verifier,
        "--num_samples", str(num_samples),
        "--duration", str(duration),
        "--candidates", str(budget_n), # Maps to Budget N
        "--seed", str(seed) # Set fixed seed
    ]
    
    # Add Beam Width if applicable
    if method in ["sbs", "lookahead"]:
        cmd.extend(["--beam_width", str(beam_width)])
        
    # Add Lookahead K if applicable
    if method == "lookahead":
        cmd.extend(["--lookahead_k", str(lookahead_k)])
    
    print(f"    > Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err_msg = result.stderr.splitlines()[-1] if result.stderr else 'Unknown Error'
            print(f"    [!] Benchmark Error: {err_msg}")
            return None

        # Result path logic from benchmark.py
        result_path = f"benchmark_results/{method}_{verifier}/final_metrics.csv"
        
        if os.path.exists(result_path):
            return pd.read_csv(result_path).iloc[0]
        else:
            return None
    except Exception as e:
        print(f"    [!] Exception: {e}")
        return None

def plot_search_comparison(df, verifier_name, metric_col, metric_label, output_suffix):
    """
    Plots Score vs. Budget (N) for a specific metric.
    """
    print(f"\nGenerating {metric_label} plot for {verifier_name}...")
    
    plot_df = df.copy()
    
    # Filter out invalid values if KLD
    if "kld" in metric_col:
        plot_df = plot_df[plot_df[metric_col] != 0]
        if plot_df.empty:
            print(f"  [Skip] No valid KLD data for {verifier_name}.")
            return

    # Beautify Method Names
    method_map = {
        "best_of_n": "Best-of-N", 
        "sbs": "Beam Search", 
        "lookahead": "Lookahead"
    }
    plot_df['Method'] = plot_df['method'].map(method_map)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Select Color Palette based on metric (Viridis for score, Magma for KLD)
    palette = "magma_r" if "kld" in metric_col else "viridis"

    sns.lineplot(
        data=plot_df, 
        x="n", 
        y=metric_col, 
        hue="Method", 
        style="Method", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        palette=palette
    )
    
    plt.title(f"Search Method Scaling ({verifier_name.upper()})", fontsize=16)
    plt.xlabel("Generation Budget ($N$)", fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    
    # Log Scale for N
    plt.xscale('log', base=2)
    
    # Ensure ticks match our budgets
    budgets = sorted(plot_df['n'].unique())
    plt.xticks(budgets, budgets)
    
    plt.legend(title="Search Strategy")
    plt.tight_layout()
    filename = f"search_comparison_{verifier_name}_{output_suffix}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    # RENAMED: --samples -> --num_samples
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per run")
    parser.add_argument("--duration", type=int, default=10, help="Audio duration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Configuration ---
    # Tested Verifiers (Semantic Only)
    verifiers = ["clap", "muq", "imagebind"]
    
    # --- Hard-Coded Budgets ---
    # 1. Best-of-N
    bon_budgets = [1, 4, 8, 16]
    
    # 2. Beam Search (Must be >= Beam Width 4)
    sbs_budgets = [4, 8, 16]
    
    # 3. Lookahead Search (Must be >= Beam Width 4)
    lookahead_budgets = [4, 8, 16]

    # Fixed Parameters
    BEAM_WIDTH = 4
    LOOKAHEAD_K = 1

    results = []

    print(f"{'='*60}")
    print(f"SEARCH METHOD PERFORMANCE EXPERIMENT")
    print(f"Verifiers: {verifiers}")
    print(f"Best-of-N Budgets: {bon_budgets}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    for verifier in verifiers:
        print(f"\n--- Evaluator: {verifier.upper()} ---")
        
        # 1. Best-of-N
        for n in bon_budgets:
            print(f"  [Best-of-{n}]...", end="", flush=True)
            row = run_benchmark_call("best_of_n", verifier, n, BEAM_WIDTH, LOOKAHEAD_K, args.num_samples, args.duration, args.seed)
            if row is not None:
                try:
                    clap = float(row.get('clap', 0))
                    kld = float(row.get('kld', 0)) if row.get('kld') != 'N/A' else 0.0
                except: clap, kld = 0.0, 0.0
                
                results.append({
                    "verifier": verifier,
                    "method": "best_of_n",
                    "n": n,
                    "score": clap,
                    "kld": kld
                })
                print(f" Score: {clap:.4f} | KLD: {kld:.4f}")

        # 2. Beam Search (SBS)
        for n in sbs_budgets:
            print(f"  [Beam Search N={n}]...", end="", flush=True)
            row = run_benchmark_call("sbs", verifier, n, BEAM_WIDTH, LOOKAHEAD_K, args.num_samples, args.duration, args.seed)
            if row is not None:
                try:
                    clap = float(row.get('clap', 0))
                    kld = float(row.get('kld', 0)) if row.get('kld') != 'N/A' else 0.0
                except: clap, kld = 0.0, 0.0

                results.append({
                    "verifier": verifier,
                    "method": "sbs",
                    "n": n,
                    "score": clap,
                    "kld": kld
                })
                print(f" Score: {clap:.4f} | KLD: {kld:.4f}")

        # 3. Lookahead Search
        for n in lookahead_budgets:
            print(f"  [Lookahead N={n}]...", end="", flush=True)
            row = run_benchmark_call("lookahead", verifier, n, BEAM_WIDTH, LOOKAHEAD_K, args.num_samples, args.duration, args.seed)
            if row is not None:
                try:
                    clap = float(row.get('clap', 0))
                    kld = float(row.get('kld', 0)) if row.get('kld') != 'N/A' else 0.0
                except: clap, kld = 0.0, 0.0

                results.append({
                    "verifier": verifier,
                    "method": "lookahead",
                    "n": n,
                    "score": clap,
                    "kld": kld
                })
                print(f" Score: {clap:.4f} | KLD: {kld:.4f}")

        # --- Generate Plots for this Verifier ---
        if len(results) > 0:
            df = pd.DataFrame(results)
            verifier_df = df[df['verifier'] == verifier]
            if not verifier_df.empty:
                try:
                    # Plot 1: Semantic Score
                    plot_search_comparison(
                        verifier_df, verifier, 
                        metric_col="score", 
                        metric_label="CLAP Score: Audio-Text Alignment (Higher is Better)", 
                        output_suffix="semantic"
                    )
                    
                    # Plot 2: KLD Score
                    plot_search_comparison(
                        verifier_df, verifier, 
                        metric_col="kld", 
                        metric_label="KLD: Music-Music Concept Similarity (Lower is Better)", 
                        output_suffix="kld"
                    )
                except Exception as e:
                    print(f"Plotting failed: {e}")

    # Final Save
    pd.DataFrame(results).to_csv("search_experiment_results.csv", index=False)
    print("\nResults saved to search_experiment_results.csv")

if __name__ == "__main__":
    main()