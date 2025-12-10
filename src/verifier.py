import requests
import io
import torchaudio
import torch

class TheCurator:
    def __init__(self, mode='quality', device='cpu', server_url="http://localhost:8000", generator=None):
        """
        Args:
            generator: Reference to MusicGenerator instance (required for 'perplexity' mode)
        """
        self.mode = mode
        self.server_url = f"{server_url}/score_batch"
        self.generator = generator
        
        if self.mode == 'perplexity' and self.generator is None:
            print("[Warning] Perplexity mode requires a 'generator' instance passed to TheCurator.")

    def score_batch(self, audio_batch, sample_rate, text_prompts=None):
        """
        Routes scoring to Local Generator (Perplexity) or Remote Server (Quality/Semantic).
        """
        # --- LOCAL: Perplexity ---
        if self.mode == 'perplexity':
            if self.generator is None:
                raise ValueError("Cannot calculate Perplexity: Generator not attached to Curator.")
            return self.generator.score_perplexity(audio_batch, sample_rate, text_prompts)

        # --- REMOTE: Quality / Semantic / Theory ---
        # Ensure batch dim
        if audio_batch.dim() == 2:
            audio_batch = audio_batch.unsqueeze(0)
            
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts] * audio_batch.shape[0]
        if text_prompts is None:
            text_prompts = [""] * audio_batch.shape[0]

        files = []
        for i in range(audio_batch.shape[0]):
            wav = audio_batch[i]
            buff = io.BytesIO()
            if wav.device.type == 'cuda':
                wav = wav.detach().cpu()
            
            torchaudio.save(buff, wav, sample_rate, format="wav")
            buff.seek(0)
            files.append(('files', (f'audio_{i}.wav', buff, 'audio/wav')))

        payload = [('mode', self.mode)]
        for p in text_prompts:
            payload.append(('prompts', p))

        try:
            response = requests.post(self.server_url, files=files, data=payload)
            if response.status_code == 200:
                return response.json()['scores']
            else:
                print(f"[Warning] Server error: {response.text}")
                return [0.0] * audio_batch.shape[0]
        except Exception as e:
            print(f"[Error] Connection failed: {e}")
            return [0.0] * audio_batch.shape[0]