from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import torch
import io
import torchaudio
from typing import List
from src.verifier_core import VerifierCore

app = FastAPI()
verifiers = {}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.on_event("startup")
async def startup_event():
    print("--- Loading Verifier Models (Server Side) ---")
    
    # 1. Quality
    # verifiers['quality'] = VerifierCore(mode='quality', device=DEVICE)
    
    # 2. Semantic (CLAP)
    verifiers['clap'] = VerifierCore(mode='clap', device=DEVICE)
    
    # 3. Semantic (MuQ)
    try:
        verifiers['muq'] = VerifierCore(mode='muq', device=DEVICE)
    except: print("[Warning] MuQ failed to load.")

    # 4. Semantic (ImageBind)
    try:
        verifiers['imagebind'] = VerifierCore(mode='imagebind', device=DEVICE)
    except Exception as e: print(f"[Warning] ImageBind failed to load: {e}")

    # 5. Theory
    # verifiers['theory'] = VerifierCore(mode='theory', device=DEVICE)    
    print("--- Models Ready ---")

@app.post("/score_batch")
async def score_batch_audio(
    files: List[UploadFile] = File(...), 
    mode: str = Form(...), 
    prompts: List[str] = Form(None)
):
    if mode not in verifiers:
        return {"scores": [], "error": f"Mode {mode} not hosted on server"}

    batch_tensors = []
    sample_rate = 32000

    for file in files:
        audio_bytes = await file.read()
        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        batch_tensors.append(wav)
        sample_rate = sr
    
    if not batch_tensors:
        return {"scores": []}

    max_len = max(wav.shape[-1] for wav in batch_tensors)
    padded_batch = []
    for wav in batch_tensors:
        if wav.shape[-1] < max_len:
            pad_amount = max_len - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad_amount))
        padded_batch.append(wav)

    full_batch = torch.stack(padded_batch).to(DEVICE)

    if prompts and len(prompts) == 1 and len(files) > 1:
        prompts = prompts * len(files)
    
    scores = verifiers[mode].score_batch(full_batch, sample_rate, text_prompts=prompts)
    
    return {"scores": scores}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)