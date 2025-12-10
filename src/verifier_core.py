import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.functional as F  # Added for resampling

class VerifierCore:
    def __init__(self, mode='quality', device='cuda'):
        self.mode = mode
        self.device = device
        self.model = None
        self.processor = None
        
        print(f"[Core] Initializing {self.mode} on {self.device}...")

        if self.mode == 'quality':
            # Audiobox Aesthetics
            from audiobox_aesthetics.infer import initialize_predictor
            self.model = initialize_predictor()
            
        elif self.mode == 'semantic':
            # CLAP
            from transformers import AutoProcessor, ClapModel
            self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
            
        elif self.mode == 'theory':
            pass # Librosa only

    def score(self, audio_tensor, sample_rate, text_prompt=None):
        if self.mode == 'quality':
            return self._get_quality_score(audio_tensor, sample_rate)
        elif self.mode == 'semantic':
            return self._get_semantic_score(audio_tensor, sample_rate, text_prompt)
        elif self.mode == 'theory':
            return self._get_theory_score(audio_tensor, sample_rate)

    def _get_quality_score(self, audio, sr):
        # Audiobox expects CPU tensors usually
        if audio.device.type == 'cuda': 
            audio = audio.detach().cpu()
        
        input_data = [{"path": audio, "sample_rate": sr}]
        
        with torch.no_grad():
            results = self.model.forward(input_data)
        
        scores = results[0]
        # Normalize (approximate scale 1-10 -> 0-1)
        pq = scores.get("PQ", 0.0)
        ce = scores.get("CE", 0.0)
        return min(max((pq + ce) / 20.0, 0.0), 1.0)

    def _get_semantic_score(self, audio, sr, text):
        if not text:
            return 0.0
            
        # CLAP requires 48kHz Audio
        target_sr = 48000
        if sr != target_sr:
            # Resample from sr (32000) to target_sr (48000)
            # Ensure audio is on CPU for resampling if necessary, though functional works on GPU
            audio = F.resample(audio, sr, target_sr)
            sr = target_sr

        # Processor expects numpy array on CPU
        audio_input = audio.cpu().numpy()
        
        # Pass the CORRECTED sampling rate (48000) to the processor
        inputs = self.processor(text=[text], audios=audio_input, return_tensors="pt", sampling_rate=sr)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 1. Get the embeddings
        audio_embeds = outputs.audio_embeds
        text_embeds = outputs.text_embeds

        # 2. Normalize them (Cosine Similarity = Dot product of normalized vectors)
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # 3. Calculate Dot Product
        similarity = torch.mm(audio_embeds, text_embeds.T).item()
        
        return similarity

    def _get_theory_score(self, audio, sr):
        y = audio.squeeze().cpu().numpy()
        if y.ndim > 1: y = y[0]
        
        # Harmonic
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            entropy = -np.sum(chroma * np.log(chroma + 1e-9), axis=0).mean()
            harmonic = 1.0 / (1.0 + entropy)
        except: harmonic = 0.5
        
        # Rhythmic
        try:
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            duration = len(y)/sr
            rhythmic = min(len(beat_frames) / (duration + 1e-9) / 2.0, 1.0)
        except: rhythmic = 0.5
        
        return 0.5 * harmonic + 0.5 * rhythmic