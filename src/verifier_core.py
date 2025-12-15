import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.functional as F
from torch.nn import functional as nn_f
import os
import tempfile
import shutil

class VerifierCore:
    def __init__(self, mode='quality', device='cuda'):
        self.mode = mode
        self.device = device
        self.model = None
        
        print(f"[Core] Initializing {self.mode.upper()} on {self.device}...")

        if self.mode == 'quality':
            from audiobox_aesthetics.infer import initialize_predictor
            self.model = initialize_predictor()
            
        elif self.mode == 'clap':
            # LAION-CLAP (Standard)
            import laion_clap
            ckpt_name = "music_audioset_epoch_15_esc_90.14.pt"
            if not os.path.exists(ckpt_name):
                print(f"[Core] Downloading CLAP weights...")
                os.system(f"wget https://huggingface.co/lukewys/laion_clap/resolve/main/{ckpt_name}")
            
            self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            self.model.load_ckpt(ckpt_name)
            self.model.to(self.device)
            self.model.eval()
            
        elif self.mode == 'muq':
            # MuQ-MuLan
            from muq import MuQMuLan
            self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.device).eval()

        elif self.mode == 'imagebind':
            # ImageBind (Official Meta Research Implementation)
            try:
                from imagebind import data
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
                
                # Store references to modules
                self.ib_data = data
                self.ib_modality = ModalityType
                
                self.model = imagebind_model.imagebind_huge(pretrained=True)
                self.model.eval()
                self.model.to(self.device)
            except ImportError:
                print("!!! Error: ImageBind not found. Please install:")
                print("pip install git+https://github.com/facebookresearch/ImageBind.git")
                raise
            
        elif self.mode == 'theory':
            pass 

    def score_batch(self, audio_batch, sample_rate, text_prompts=None):
        batch_size = audio_batch.shape[0]
        
        if self.mode == 'quality':
            return self._get_quality_score_batch(audio_batch, sample_rate)
        elif self.mode == 'clap':
            return self._get_clap_score_batch(audio_batch, sample_rate, text_prompts)
        elif self.mode == 'muq':
            return self._get_muq_score_batch(audio_batch, sample_rate, text_prompts)
        elif self.mode == 'imagebind':
            return self._get_imagebind_score_batch(audio_batch, sample_rate, text_prompts)
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

    def _get_clap_score_batch(self, audio_batch, sr, texts):
        if not texts: return [0.0] * audio_batch.shape[0]
        
        target_sr = 48000
        if sr != target_sr: audio_batch = F.resample(audio_batch, sr, target_sr)
        if audio_batch.dim() == 3: audio_batch = audio_batch.mean(dim=1)
        
        audio_batch = audio_batch.to(self.device)
        with torch.no_grad():
            audio_embeds = self.model.get_audio_embedding_from_data(x=audio_batch, use_tensor=True)
            text_embeds = self.model.get_text_embedding(texts, use_tensor=True)
            return nn_f.cosine_similarity(text_embeds, audio_embeds, dim=-1).tolist()

    def _get_muq_score_batch(self, audio_batch, sr, texts):
        if not texts: return [0.0] * audio_batch.shape[0]

        target_sr = 24000
        if sr != target_sr: audio_batch = F.resample(audio_batch, sr, target_sr)
        if audio_batch.dim() == 3: audio_batch = audio_batch.mean(dim=1) 
        
        audio_batch = audio_batch.to(self.device)
        with torch.no_grad():
            audio_embeds = self.model(wavs=audio_batch)
            text_embeds = self.model(texts=texts)
            sim_matrix = self.model.calc_similarity(audio_embeds, text_embeds)
            scores = sim_matrix.diag() if sim_matrix.shape[0] > 1 else sim_matrix.squeeze()
            if scores.dim() == 0: scores = scores.unsqueeze(0)
            return scores.tolist()

    def _get_imagebind_score_batch(self, audio_batch, sr, texts):
        """
        Scoring using Official ImageBind.
        Note: ImageBind data loaders expect file paths.
        """
        if not texts: return [0.0] * audio_batch.shape[0]
        
        # 1. Save tensors to temporary files
        # ImageBind handles resampling internally via torchaudio load
        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        
        try:
            for i, wav in enumerate(audio_batch):
                wav_cpu = wav.detach().cpu()
                if wav_cpu.dim() == 1: wav_cpu = wav_cpu.unsqueeze(0) # [1, T]
                path = os.path.join(temp_dir, f"temp_{i}.wav")
                torchaudio.save(path, wav_cpu, sr)
                temp_paths.append(path)
            
            # 2. Load Inputs
            inputs = {
                self.ib_modality.AUDIO: self.ib_data.load_and_transform_audio_data(temp_paths, self.device),
                self.ib_modality.TEXT: self.ib_data.load_and_transform_text(texts, self.device),
            }

            # 3. Forward
            with torch.no_grad():
                embeddings = self.model(inputs)

            audio_embeds = embeddings[self.ib_modality.AUDIO]
            text_embeds = embeddings[self.ib_modality.TEXT]

            # 4. Cosine Similarity (Normalized Dot Product)
            # ImageBind output is not strictly normalized
            audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Diagonal for matching pairs
            similarities = (audio_embeds * text_embeds).sum(dim=-1)
            
            return similarities.tolist()

        except Exception as e:
            print(f"[ImageBind Error] {e}")
            return [0.0] * len(texts)
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_theory_score(self, audio, sr):
        # ... (Same as previous implementation)
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