import subprocess
import pandas as pd
import os
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark_call(method, verifier, candidates, num_samples, duration):
    """
    Wraps the call to benchmark.py and retrieves the result.
    """
    cmd = [
        "python", "benchmark.py",
        "--method", method,
        "--verifier", verifier,
        "--num_samples", str(num_samples),
        "--duration", str(duration)
    ]
    
    # Add candidates only if not baseline
    if method == "best_of_n":
        cmd.extend(["--candidates", str(candidates)])
    
    print(f"    > Command: {' '.join(cmd)}")
    
    try:
        # Run benchmark.py
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    [!] Error running benchmark: {result.stderr}")
            return None

        # Read the result file
        result_path = f"benchmark_results/{method}_{verifier}/final_metrics.csv"
        
        if os.path.exists(result_path):
            return pd.read_csv(result_path).iloc[0]
        else:
            print(f"    [!] Error: Result file not found at {result_path}")
            return None
            
    except Exception as e:
        print(f"    [!] Exception during benchmark: {e}")
        return None

def plot_results(df):
    """Generates scaling curves for CLAP and KLD."""
    print("\nGenerating plots...")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Filter for plotting
    plot_df = df.copy()
    
    # --- Plot 1: CLAP Score (Semantic Alignment) ---
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=plot_df, 
        x="n", 
        y="clap_score", 
        hue="verifier", 
        style="verifier", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        palette="viridis"
    )
    
    plt.title("Verifer Scaling", fontsize=16)
    plt.ylabel("CLAP Score: Audio-Text Alignment (Higher is Better)", fontsize=12)
    plt.xlabel("Compute Budget (N Candidates)", fontsize=12)
    
    # Log scale for N
    plt.xscale('log', base=2)
    unique_n = sorted(plot_df['n'].unique())
    plt.xticks(unique_n, unique_n)
    
    plt.legend(title="Verifier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("scaling_clap.png", dpi=300)
    print("Saved 'scaling_clap.png'")

    # --- Plot 2: KLD Score (Audio Quality/Similarity) ---
    plt.figure(figsize=(10, 6))
    
    # Filter out N/A KLD values (0.0)
    kld_df = plot_df[plot_df['kld_score'] != 0]
    
    if not kld_df.empty:
        sns.lineplot(
            data=kld_df, 
            x="n", 
            y="kld_score", 
            hue="verifier", 
            style="verifier", 
            markers=True, 
            dashes=False,
            linewidth=2.5,
            palette="magma_r"
        )
        
        plt.title("Verifer Scaling", fontsize=16)
        plt.ylabel("KLD: Music-Music Concept Similarity (Lower is Better)", fontsize=12)
        plt.xlabel("Compute Budget (N Candidates)", fontsize=12)
        
        plt.xscale('log', base=2)
        plt.xticks(unique_n, unique_n)
        
        plt.legend(title="Verifier", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("scaling_kld.png", dpi=300)
        print("Saved 'scaling_kld.png'")

def main():
    parser = argparse.ArgumentParser()
    # RENAMED: --samples -> --num_samples
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per run")
    parser.add_argument("--duration", type=int, default=10, help="Audio duration (sec)")
    args = parser.parse_args()

    # --- Experiment Configuration ---
    # List of all verifiers you want to test.
    verifiers = ["clap", "perplexity", "theory", "quality", "muq", "imagebind"]
    
    # Budgets to test (Powers of 2)
    # N=1 is Baseline, others are Best-of-N
    n_values = [1, 2, 4, 8, 16] 

    results_data = []

    print(f"{'='*60}")
    print(f"VERIFIER SCALING EXPERIMENT")
    print(f"Samples per run: {args.num_samples}")
    print(f"Budgets (N): {n_values}")
    print(f"{'='*60}\n")

    for verifier in verifiers:
        print(f"\n--- Testing Verifier: {verifier.upper()} ---")
        
        # Store baseline stats for this verifier loop
        baseline_clap = 0.0
        baseline_kld = 0.0
        verifier_failed = False

        for n in n_values:
            if verifier_failed:
                break # Skip higher N if verifier is broken/unloaded

            print(f"  [N={n}] Processing...", end="", flush=True)
            start_time = time.time()
            
            # Determine Method
            if n == 1:
                method = "baseline"
            else:
                method = "best_of_n"

            # Execute
            row = run_benchmark_call(method, verifier, n, args.num_samples, args.duration)
            elapsed = time.time() - start_time
            
            if row is not None:
                # Extract metrics
                try:
                    clap = float(row.get('clap', 0))
                    kld = float(row.get('kld', 0)) if row.get('kld') != 'N/A' else 0.0
                except:
                    clap, kld = 0.0, 0.0

                # Handle Baseline Logic
                if n == 1:
                    baseline_clap = clap
                    baseline_kld = kld
                    clap_delta = 0.0
                    kld_delta = 0.0
                else:
                    clap_delta = clap - baseline_clap
                    # KLD is lower-is-better, so improvement = (Baseline - New)
                    kld_delta = baseline_kld - kld 

                # Updated Print Statement
                print(f" Done ({elapsed:.1f}s). CLAP: {clap:.4f} ({'+' if clap_delta>=0 else ''}{clap_delta:.4f}) | KLD: {kld:.4f} ({'+' if kld_delta>=0 else ''}{kld_delta:.4f})")

                # Log Data
                results_data.append({
                    "verifier": verifier,
                    "n": n,
                    "method": method,
                    "clap_score": clap,
                    "kld_score": kld,
                    "clap_improvement": clap_delta,
                    "kld_improvement": kld_delta,
                    "duration_sec": elapsed
                })
            else:
                print(" Failed.")
                if n == 1:
                    print(f"    [!] Baseline failed for {verifier}. Skipping rest of loop (likely server error).")
                    verifier_failed = True
            
            # Save intermediate results
            if results_data:
                pd.DataFrame(results_data).to_csv("scaling_results_partial.csv", index=False)

    # --- Final Save & Plot ---
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv("scaling_results.csv", index=False)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("Results saved to scaling_results.csv")
        
        try:
            plot_results(df)
        except Exception as e:
            print(f"[!] Plotting failed: {e}")
            import traceback
            traceback.print_exc()

        # Print Summary Table
        print("\nSUMMARY (CLAP Scores):")
        print(df.pivot(index='n', columns='verifier', values='clap_score'))
    else:
        print("\n[!] No results collected.")

if __name__ == "__main__":
    main()