import argparse
import torchaudio
import torch
import os
from src.generator import MusicGenerator
from src.verifier import TheCurator
from src.search import InferenceSearch

def save_audio(wav, sr, filename):
    # Ensure shape is [Channels, Time] for saving
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    wav = wav.cpu()
    torchaudio.save(filename, wav, sr)
    print(f"[Output] Saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="The Curator: Inference-Time Search for Music Generation")
    
    # Generation Params
    parser.add_argument("--prompt", type=str, default="8-bit arcade soundtrack", help="Text description")
    parser.add_argument("--duration", type=int, default=10, help="Total duration in seconds")
    
    # Search Params
    parser.add_argument("--method", type=str, choices=["best_of_n", "sbs"], default="best_of_n")
    parser.add_argument("--candidates", type=int, default=4, help="N for Best-of-N")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width for SBS")
    
    # Verifier Params
    parser.add_argument("--verifier", type=str, choices=["quality", "semantic", "theory", "perplexity"], default="quality",
                        help="Metric to optimize")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Verifier Server URL")
    
    args = parser.parse_args()

    # 1. Initialize Generator
    print("--- Initializing Generator (MusicGen) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    # 2. Initialize Verifier (Client)
    # We pass 'generator' so that 'perplexity' mode can run locally
    print(f"--- Connecting to Verifier ({args.verifier}) ---")
    verifier = TheCurator(mode=args.verifier, server_url=args.server, generator=generator)
    
    # 3. Setup Search Engine
    search_engine = InferenceSearch(generator, verifier)

    # 4. Run Search
    print(f"--- Starting Inference: {args.method.upper()} optimizing {args.verifier.upper()} ---")
    final_audio = None
    
    if args.method == "best_of_n":
        final_audio = search_engine.best_of_n(
            args.prompt, 
            n_candidates=args.candidates, 
            duration=args.duration
        )
    elif args.method == "sbs":
        final_audio = search_engine.stepwise_beam_search(
            args.prompt, 
            total_duration=args.duration, 
            step_size=2,       # Standard step size
            beam_width=args.beam_width, 
            expand_k=8         # Standard expansion factor
        )

    # 5. Save Result
    output_dir = "outputs"
    output_filename = f"{args.method}_{args.verifier}.wav"
    save_audio(final_audio, generator.sample_rate, f'{output_dir}/{output_filename}')

if __name__ == "__main__":
    main()