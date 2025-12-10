from audiocraft.models import MusicGen
import torch
import typing as tp
import torchaudio.functional as F

class MusicGenerator:
    def __init__(self, model_size='facebook/musicgen-small', device='cuda'):
        print(f"Loading MusicGen: {model_size}...")
        self.model = MusicGen.get_pretrained(model_size, device=device)
        self.sample_rate = 32000 
        self.model.set_generation_params(duration=30)
        self.device = device
        
        # Ensure sub-models are in eval mode
        self.model.lm.eval()
        self.model.compression_model.eval()

    def generate_batch(self, prompts: tp.List[str], duration: float):
        self.model.set_generation_params(duration=duration)
        wav = self.model.generate(prompts)
        return wav.cpu()

    def generate_continuation_batch(self, prompt_wavs: torch.Tensor, prompts: tp.List[str], duration: float):
        self.model.set_generation_params(duration=duration)
        prompt_wavs = prompt_wavs.to(self.device)
        if prompt_wavs.dim() == 2:
            prompt_wavs = prompt_wavs.unsqueeze(1)
        wav = self.model.generate_continuation(
            prompt=prompt_wavs, 
            prompt_sample_rate=self.sample_rate, 
            descriptions=prompts, 
            progress=False
        )
        return wav.cpu()

    def score_perplexity(self, audio_batch: torch.Tensor, sr: int, texts: tp.List[str]):
        """
        Calculates Inverse Perplexity (Likelihood) locally.
        Returns values in [0, 1] (1.0 = perfect confidence).
        """
        # 1. Resample if necessary (MusicGen needs 32kHz)
        if sr != self.sample_rate:
            audio_batch = F.resample(audio_batch, sr, self.sample_rate)
        
        # Ensure batch dim [B, C, T]
        if audio_batch.dim() == 2: audio_batch = audio_batch.unsqueeze(1)
        audio_batch = audio_batch.to(self.device)
        
        with torch.no_grad():
            # 2. Encode Audio to Codes [B, K, T]
            # The compression model (Encodec) usually runs in FP32
            codes, _ = self.model.compression_model.encode(audio_batch)
            
            # 3. Prepare Text Conditions
            attributes, _ = self.model._prepare_tokens_and_attributes(texts, None)
            
            # 4. Compute LM Logits
            # FIX: We must use the model's autocast context for the Transformer
            with self.model.autocast: 
                lm_output = self.model.lm.compute_predictions(
                    codes, conditions=attributes, keep_only_valid_steps=False
                )
                logits = lm_output.logits  # [B, K, T, Card]
                mask = lm_output.mask      # [B, K, T]

            # 5. Calculate Cross Entropy Loss
            # Ensure calculations happen in Float32 for stability
            flat_logits = logits.float().reshape(-1, logits.shape[-1]) 
            flat_codes = codes.reshape(-1)
            flat_mask = mask.float().reshape(-1)
            
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fn(flat_logits, flat_codes)
            
            # Mask invalid steps (padding)
            token_losses = token_losses * flat_mask
            
            # Average loss per sample
            B = audio_batch.shape[0]
            token_losses = token_losses.reshape(B, -1)
            valid_counts = flat_mask.reshape(B, -1).sum(dim=1).clamp(min=1)
            
            mean_loss = token_losses.sum(dim=1) / valid_counts
            
            # 6. Convert to Score: exp(-loss) - Inverse Perplexity higher the better
            scores = torch.exp(-mean_loss).tolist()
            return scores