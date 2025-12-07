import torch

class InferenceSearch:
    def __init__(self, generator, verifier):
        self.gen = generator
        self.curator = verifier

    def best_of_n(self, prompt, n_candidates=4, duration=10):
        print(f"Running Best-of-{n_candidates} (Metric: {self.curator.mode})...")
        candidates = []
        for i in range(n_candidates):
            wav = self.gen.generate_full(prompt, duration)
            score = self.curator.score(wav, self.gen.sample_rate, prompt)
            candidates.append((score, wav))
            print(f"  Candidate {i+1}: Score = {score:.4f}")
        
        best = max(candidates, key=lambda x: x[0])
        print(f"Winner: {best[0]:.4f}")
        return best[1]

    def stepwise_beam_search(self, prompt, total_duration=10, step_size=2, beam_width=4, expand_k=8):
        print(f"Running SBS: Step={step_size}s, Beam={beam_width}")
        
        # Beams: List of (score, audio_tensor)
        beams = [(0.0, torch.zeros((1, 0)))] 
        
        # Calculate number of steps
        num_steps = int(total_duration / step_size)

        for step in range(num_steps):
            # Calculate the TARGET duration for this specific step
            # Step 0 -> 2s, Step 1 -> 4s, Step 2 -> 6s...
            current_target_duration = (step + 1) * step_size
            
            print(f"--- Step {step + 1}/{num_steps} (Target Duration: {current_target_duration}s) ---")
            candidates = []

            for _, current_audio in beams:
                # Ensure correct dimensions (Batch, Channels, Time)
                if current_audio.dim() == 2: 
                    current_audio = current_audio.unsqueeze(0)

                for k in range(expand_k):
                    if current_audio.size(-1) == 0:
                        # First step: Generate from scratch
                        full_audio = self.gen.generate_full(prompt, duration=step_size)
                    else:
                        # Continuation
                        prompt_audio = current_audio.to(self.gen.model.device)
                        
                        # FIX 1: Pass the TOTAL target duration (e.g. 4s), not just the increment (2s)
                        # FIX 2: MusicGen returns the FULL sequence, so we assign directly (no torch.cat)
                        full_audio = self.gen.generate_continuation(
                            prompt_audio, 
                            prompt, 
                            duration=current_target_duration
                        )
                        
                        # Ensure output is on CPU and correct shape
                        if full_audio.dim() == 2: 
                            full_audio = full_audio.unsqueeze(0)

                    # Score the Accumulated Audio
                    score = self.curator.score(full_audio, self.gen.sample_rate, prompt)
                    candidates.append((score, full_audio))

            # Pruning Phase
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
            
            print(f"  Best Beam Score: {beams[0][0]:.4f}")

        return beams[0][1]