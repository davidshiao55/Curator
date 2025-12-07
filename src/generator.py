from audiocraft.models import MusicGen
import torch

class MusicGenerator:
    def __init__(self, model_size='facebook/musicgen-small', device='cuda'):
        print(f"Loading MusicGen: {model_size}...")
        self.model = MusicGen.get_pretrained(model_size, device=device)
        self.sample_rate = 32000 
        self.model.set_generation_params(duration=30)

    def generate_full(self, prompt, duration):
        self.model.set_generation_params(duration=duration)
        wav = self.model.generate([prompt])
        return wav[0].cpu()

    def generate_continuation(self, prompt_wav, text_prompt, duration):
        self.model.set_generation_params(duration=duration)
        wav = self.model.generate_continuation(
            prompt=prompt_wav, 
            prompt_sample_rate=self.sample_rate, 
            descriptions=[text_prompt], 
            progress=False
        )
        return wav[0].cpu()