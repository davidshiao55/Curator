import argparse
import torchaudio
import torch
from src.generator import MusicGenerator
from src.verifier import TheCurator
from src.search import InferenceSearch

def save_audio(wav, sr, filename):
    if wav.dim() == 3: wav = wav.squeeze(0)
    wav = wav.cpu()
    torchaudio.save(filename, wav, sr)
    print(f"Saved: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="8-bit arcade soundtrack")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--method", choices=["best_of_n", "sbs"], default="best_of_n")
    parser.add_argument("--verifier", choices=["quality", "semantic", "theory"], default="quality")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    # 1. Generator (MusicGen)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = MusicGenerator(model_size='facebook/musicgen-small', device=device)
    
    # 2. Verifier (Client Wrapper)
    print(f"Connecting to Verifier Server at {args.server}...")
    verifier = TheCurator(mode=args.verifier, server_url=args.server)
    
    # 3. Search
    search_engine = InferenceSearch(generator, verifier)

    if args.method == "best_of_n":
        final_audio = search_engine.best_of_n(args.prompt, duration=args.duration)
    else:
        final_audio = search_engine.stepwise_beam_search(args.prompt, total_duration=args.duration)

    save_audio(final_audio, generator.sample_rate, f"./output/output_{args.method}.wav")

if __name__ == "__main__":
    main()