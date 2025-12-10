import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.functional as F
from torch.nn import functional as nn_f

class VerifierCore:
    def __init__(self, mode='quality', device='cuda'):
        self.mode = mode
        self.device = device
        self.model = None
        self.processor = None
        
        print(f"[Core] Initializing {self.mode} on {self.device}...")

        if self.mode == 'quality':
            from audiobox_aesthetics.infer import initialize_predictor
            self.model = initialize_predictor()
            
        elif self.mode == 'semantic':
            from transformers import AutoProcessor, ClapModel
            self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
            
        elif self.mode == 'theory':
            pass 

    def score_batch(self, audio_batch, sample_rate, text_prompts=None):
        batch_size = audio_batch.shape[0]
        
        if self.mode == 'quality':
            return self._get_quality_score_batch(audio_batch, sample_rate)
        elif self.mode == 'semantic':
            return self._get_semantic_score_batch(audio_batch, sample_rate, text_prompts)
        elif self.mode == 'theory':
            scores = []
            for i in range(batch_size):
                scores.append(self._get_theory_score(audio_batch[i], sample_rate))
            return scores
        return [0.0] * batch_size

    def _get_quality_score_batch(self, audio_batch, sr):
        if audio_batch.device.type == 'cuda': 
            audio_batch = audio_batch.detach().cpu()
        
        input_data = [{"path": audio_batch[i], "sample_rate": sr} for i in range(audio_batch.shape[0])]
        
        with torch.no_grad():
            results = self.model.forward(input_data)
        
        final_scores = []
        for res in results:
            pq = res.get("PQ", 0.0)
            ce = res.get("CE", 0.0)
            final_scores.append(min(max((pq + ce) / 20.0, 0.0), 1.0))
        return final_scores

    def _get_semantic_score_batch(self, audio_batch, sr, texts):
        if not texts: return [0.0] * audio_batch.shape[0]

        target_sr = 48000
        if sr != target_sr:
            audio_batch = F.resample(audio_batch, sr, target_sr)
            sr = target_sr

        audio_inputs = [wav.squeeze().cpu().numpy() for wav in audio_batch]
        inputs = self.processor(text=texts, audios=audio_inputs, return_tensors="pt", sampling_rate=sr, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        similarities = nn_f.cosine_similarity(outputs.text_embeds, outputs.audio_embeds, dim=-1)
        return similarities.tolist()

    def _get_theory_score(self, audio, sr):
        y = audio.squeeze().cpu().numpy()
        if y.ndim > 1: y = y[0]
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            entropy = -np.sum(chroma * np.log(chroma + 1e-9), axis=0).mean()
            harmonic = 1.0 / (1.0 + entropy)
        except: harmonic = 0.5
        try:
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            rhythmic = min(len(beat_frames) / (len(y)/sr + 1e-9) / 2.0, 1.0)
        except: rhythmic = 0.5
        return 0.5 * harmonic + 0.5 * rhythmic