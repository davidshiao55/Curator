import os
import subprocess
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_clip(ytid, start_s, end_s, output_dir):
    filename = os.path.join(output_dir, f"{ytid}.wav")
    if os.path.exists(filename):
        return
    
    # Download specific segment using ffmpeg via yt-dlp
    # Note: We download 30s to be safe and trim, or use download-sections
    url = f"https://www.youtube.com/watch?v={ytid}"
    command = [
        "yt-dlp",
        "--quiet", "--no-warnings",
        "-x", "--audio-format", "wav",
        "--force-keyframes-at-cuts",
        "--download-sections", f"*{start_s}-{end_s}",
        "-o", filename,
        url
    ]
    
    try:
        subprocess.run(command, check=True, timeout=60)
    except Exception as e:
        # print(f"Failed to download {ytid}: {e}")
        pass

def main():
    print("Loading MusicCaps metadata...")
    dataset = load_dataset("google/musiccaps", split="train")
    
    # Create output directory
    output_dir = "music_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {len(dataset)} clips to '{output_dir}'...")
    print("This may take a while. Using 8 threads.")

    # Prepare list for parallel download
    tasks = []
    for item in dataset:
        tasks.append((item['ytid'], item['start_s'], item['end_s'], output_dir))

    # Parallel download
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(lambda p: download_clip(*p), tasks), total=len(tasks)))

    print("Download complete.")

if __name__ == "__main__":
    main()