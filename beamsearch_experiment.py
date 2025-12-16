import subprocess
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark_call(method, budget_n, beam_width, step_size, num_samples, duration, seed):
    """
    Wraps the call to benchmark.py.
    """
    # Fixed to CLAP as requested for semantic evaluation
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
    
    # Add SBS specific params
    if method == "sbs":
        cmd.extend([
            "--beam_width", str(beam_width),
            "--step_size", str(step_size)
        ])
    
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

def plot_beam_search(df, metric_col, metric_label, output_suffix, baseline_val=None):
    """
    Plots performance across budgets for different configurations, including a baseline reference.
    """
    print(f"\nGenerating {output_suffix} plot...")
    
    plot_df = df.copy()
    
    # Create a Configuration label (e.g., "Step=2s, Width=4")
    plot_df['Configuration'] = plot_df.apply(
        lambda row: f"Step={row['step_size']}s, Width={row['beam_width']}", axis=1
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Select Palette
    palette = "magma_r" if "kld" in metric_col else "viridis"

    # Plot the Sweep Lines
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
        
        # 1. Horizontal Reference Line
        plt.axhline(
            y=baseline_val, 
            color=color, 
            linestyle='--', 
            linewidth=1.5,
            alpha=0.7
        )
        
        # 2. Distinct Dot at N=1
        plt.scatter(
            [1], [baseline_val], 
            color=color, 
            s=100, 
            marker='o',
            label=f"Baseline (N=1): {baseline_val:.4f}",
            zorder=10 # Ensure it sits on top
        )

    plt.title(f"Beam Search Hyperparameters (CLAP)", fontsize=16)
    plt.xlabel("Generation Budget ($N$)", fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    
    # Log Scale
    plt.xscale('log', base=2)
    
    # Ensure Ticks include 1 (Baseline) and the Budgets
    budgets = sorted(plot_df['n'].unique())
    if baseline_val is not None:
        budgets = [1] + budgets # Add N=1 to axis
        
    plt.xticks(budgets, budgets)
    
    plt.legend(title="Hyperparameters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"beamsearch_sweep_{output_suffix}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per run")
    parser.add_argument("--duration", type=int, default=10, help="Audio duration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Hyperparameter Sweep Configuration ---
    budgets_n = [4, 8, 16]
    step_sizes = [2, 5]
    beam_widths = [2, 4]

    results = []

    print(f"{'='*60}")
    print(f"BEAM SEARCH HYPERPARAMETER EXPERIMENT")
    print(f"Budgets (N): {budgets_n}")
    print(f"Step Sizes: {step_sizes}")
    print(f"Beam Widths (M): {beam_widths}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # --- 1. Run Baseline (N=1) ---
    print("\n--- Running Baseline (N=1) ---")
    # For baseline, width and step don't matter, passing 0/0
    base_row = run_benchmark_call("baseline", 1, 0, 0, args.num_samples, args.duration, args.seed)
    
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
        for width in beam_widths:
            # Constraint: N must be >= Beam Width
            if n < width:
                print(f"[Skip] Budget N={n} < Width M={width}")
                continue
                
            for step in step_sizes:
                print(f"\n--- Testing N={n}, Width={width}, Step={step}s ---")
                
                row = run_benchmark_call("sbs", n, width, step, args.num_samples, args.duration, args.seed)
                
                if row is not None:
                    try:
                        clap = float(row.get('clap', 0))
                        kld = float(row.get('kld', 0)) if row.get('kld') != 'N/A' else 0.0
                    except: clap, kld = 0.0, 0.0
                    
                    results.append({
                        "n": n,
                        "beam_width": width,
                        "step_size": step,
                        "score": clap,
                        "kld": kld
                    })
                    print(f" Result: CLAP={clap:.4f} | KLD={kld:.4f}")

    # Save Data & Plot
    if results:
        df = pd.DataFrame(results)
        df.to_csv("beamsearch_results.csv", index=False)
        print("\nResults saved to beamsearch_results.csv")
        
        try:
            # Plot 1: Semantic
            plot_beam_search(
                df, "score", 
                "CLAP Score: Audio-Text Alignment (Higher is Better)", 
                "semantic",
                baseline_val=baseline_clap if baseline_clap > 0 else None
            )
            
            # Plot 2: KLD
            kld_df = df[df['kld'] != 0]
            if not kld_df.empty:
                # Only pass baseline if it is valid
                base_kld_val = baseline_kld if baseline_kld > 0 else None
                
                plot_beam_search(
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