import subprocess
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark_call(method, budget_n, beam_width, step_size, lookahead_k, num_samples, duration, seed):
    """
    Wraps the call to benchmark.py.
    """
    # Fixed to CLAP for semantic evaluation
    verifier = "clap" 
    
    cmd = [
        "python", "benchmark.py",
        "--method", method,
        "--verifier", verifier,
        "--num_samples", str(num_samples),
        "--duration", str(duration),
        "--candidates", str(budget_n),
        "--seed", str(seed)
    ]
    
    # Add Search specific params
    if method in ["sbs", "lookahead"]:
        cmd.extend([
            "--beam_width", str(beam_width),
            "--step_size", str(step_size)
        ])
    
    if method == "lookahead":
        cmd.extend(["--lookahead_k", str(lookahead_k)])
    
    print(f"    > Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err_msg = result.stderr.splitlines()[-1] if result.stderr else 'Unknown Error'
            print(f"    [!] Benchmark Error: {err_msg}")
            return None

        result_path = f"benchmark_results/{method}_{verifier}/final_metrics.csv"
        
        if os.path.exists(result_path):
            return pd.read_csv(result_path).iloc[0]
        else:
            return None
    except Exception as e:
        print(f"    [!] Exception: {e}")
        return None

def plot_lookahead(df, metric_col, metric_label, output_suffix, baseline_val=None):
    """
    Plots performance across budgets for different lookahead configurations.
    """
    print(f"\nGenerating {output_suffix} plot...")
    
    plot_df = df.copy()
    
    # Create Label
    plot_df['Configuration'] = plot_df['lookahead_k'].apply(lambda k: f"Lookahead k={k}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Select Palette
    palette = "magma_r" if "kld" in metric_col else "viridis"

    # Plot Lines
    sns.lineplot(
        data=plot_df, 
        x="n", 
        y=metric_col, 
        hue="Configuration", 
        style="Configuration", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        palette=palette
    )
    
    # Plot Baseline (Line + Dot)
    if baseline_val is not None:
        color = 'red' if 'kld' not in metric_col else 'blue'
        
        # Horizontal Line
        plt.axhline(
            y=baseline_val, 
            color=color, 
            linestyle='--', 
            linewidth=1.5,
            alpha=0.7
        )
        
        # Dot at N=1
        plt.scatter(
            [1], [baseline_val], 
            color=color, 
            s=100, 
            marker='o',
            label=f"Baseline (N=1): {baseline_val:.4f}",
            zorder=10
        )

    plt.title(f"Lookahead Hyperparameters (CLAP)", fontsize=16)
    plt.xlabel("Generation Budget ($N$)", fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    
    # Log Scale
    plt.xscale('log', base=2)
    
    # Ensure Ticks include 1 (Baseline) and Budgets
    budgets = sorted(plot_df['n'].unique())
    if baseline_val is not None:
        budgets = [1] + budgets
        
    plt.xticks(budgets, budgets)
    
    plt.legend(title="Lookahead Steps", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"lookahead_sweep_{output_suffix}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per run")
    parser.add_argument("--duration", type=int, default=10, help="Audio duration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Sweep Configuration ---
    budgets_n = [4, 8, 16]
    lookahead_ks = [1, 2, 3]
    
    # Fixed Parameters
    FIXED_BEAM_WIDTH = 4
    FIXED_STEP_SIZE = 2

    results = []

    print(f"{'='*60}")
    print(f"LOOKAHEAD HYPERPARAMETER EXPERIMENT")
    print(f"Budgets (N): {budgets_n}")
    print(f"Lookahead Steps (k): {lookahead_ks}")
    print(f"Fixed Beam Width: {FIXED_BEAM_WIDTH}")
    print(f"Fixed Step Size: {FIXED_STEP_SIZE}s")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # --- 1. Run Baseline (N=1) ---
    print("\n--- Running Baseline (N=1) ---")
    base_row = run_benchmark_call("baseline", 1, 0, 0, 0, args.num_samples, args.duration, args.seed)
    
    baseline_clap = 0.0
    baseline_kld = 0.0
    
    if base_row is not None:
        try:
            baseline_clap = float(base_row.get('clap', 0))
            baseline_kld = float(base_row.get('kld', 0)) if base_row.get('kld') != 'N/A' else 0.0
        except: pass
        print(f" Baseline Result: CLAP={baseline_clap:.4f} | KLD={baseline_kld:.4f}")
    else:
        print(" [!] Baseline Failed.")

    # --- 2. Run Sweep ---
    for n in budgets_n:
        for k in lookahead_ks:
            # Lookahead also requires N >= Beam Width (inherited from Beam Search)
            if n < FIXED_BEAM_WIDTH:
                print(f"[Skip] Budget N={n} < Width M={FIXED_BEAM_WIDTH}")
                continue
                
            print(f"\n--- Testing N={n}, k={k} ---")
            
            row = run_benchmark_call(
                "lookahead", n, FIXED_BEAM_WIDTH, FIXED_STEP_SIZE, k, 
                args.num_samples, args.duration, args.seed
            )
            
            if row is not None:
                try:
                    clap = float(row.get('clap', 0))
                    kld = float(row.get('kld', 0)) if row.get('kld') != 'N/A' else 0.0
                except: clap, kld = 0.0, 0.0
                
                results.append({
                    "n": n,
                    "lookahead_k": k,
                    "score": clap,
                    "kld": kld
                })
                print(f" Result: CLAP={clap:.4f} | KLD={kld:.4f}")

    # Save Data & Plot
    if results:
        df = pd.DataFrame(results)
        df.to_csv("lookahead_results.csv", index=False)
        print("\nResults saved to lookahead_results.csv")
        
        try:
            # Plot 1: Semantic
            plot_lookahead(
                df, "score", 
                "CLAP Score: Audio-Text Alignment (Higher is Better)", 
                "semantic",
                baseline_val=baseline_clap if baseline_clap > 0 else None
            )
            
            # Plot 2: KLD
            kld_df = df[df['kld'] != 0]
            if not kld_df.empty:
                base_kld_val = baseline_kld if baseline_kld > 0 else None
                
                plot_lookahead(
                    kld_df, "kld", 
                    "KLD: Concept Similarity (Lower is Better)", 
                    "kld",
                    baseline_val=base_kld_val
                )
        except Exception as e:
            print(f"Plotting failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()