import requests
import io
import torchaudio
import torch

class TheCurator:
    def __init__(self, mode='quality', device='cpu', server_url="http://localhost:8000"):
        self.mode = mode
        self.device = device # Kept for compatibility, not used in client
        self.server_url = f"{server_url}/score"

    def score(self, audio_tensor, sample_rate, text_prompt=""):
        # Prepare audio buffer
        buff = io.BytesIO()
        
        # Ensure tensor is on CPU for saving
        if audio_tensor.device.type == 'cuda':
            audio_tensor = audio_tensor.detach().cpu()
            
        # Ensure dims are (Channels, Time)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 3: # (Batch, Channels, Time)
            audio_tensor = audio_tensor.squeeze(0)
            
        torchaudio.save(buff, audio_tensor, sample_rate, format="wav")
        buff.seek(0)

        files = {'file': ('temp.wav', buff, 'audio/wav')}
        data = {'mode': self.mode, 'prompt': text_prompt or ""}

        try:
            response = requests.post(self.server_url, files=files, data=data)
            if response.status_code == 200:
                return response.json()['score']
            else:
                print(f"[Warning] Server error: {response.text}")
                return 0.0
        except requests.exceptions.ConnectionError:
            print("[Error] Connection refused. Is verifier_server.py running?")
            return 0.0