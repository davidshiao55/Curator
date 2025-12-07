from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import torch
import io
import torchaudio
from src.verifier_core import VerifierCore

app = FastAPI()
verifiers = {}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.on_event("startup")
async def startup_event():
    print("--- Loading Verifier Models ---")
    verifiers['quality'] = VerifierCore(mode='quality', device=DEVICE)
    verifiers['semantic'] = VerifierCore(mode='semantic', device=DEVICE)
    verifiers['theory'] = VerifierCore(mode='theory', device=DEVICE)
    print("--- Models Ready ---")

@app.post("/score")
async def score_audio(file: UploadFile = File(...), mode: str = Form(...), prompt: str = Form(None)):
    audio_bytes = await file.read()
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    wav = wav.to(DEVICE)
    
    if mode not in verifiers:
        return {"score": 0.0, "error": "Invalid mode"}
        
    score_val = verifiers[mode].score(wav, sr, text_prompt=prompt)
    return {"score": float(score_val)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)