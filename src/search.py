import torch

class InferenceSearch:
    def __init__(self, generator, verifier):
        self.gen = generator
        self.curator = verifier

    def best_of_n(self, prompt, n_candidates=4, duration=10):
        print(f"Running Best-of-{n_candidates} (Batched) [{self.curator.mode}]...")
        
        # 1. Generate Batch
        prompts = [prompt] * n_candidates
        wavs = self.gen.generate_batch(prompts, duration)
        
        # 2. Score Batch
        scores = self.curator.score_batch(wavs, self.gen.sample_rate, prompts)
        
        # 3. Find Best
        for i, score in enumerate(scores):
            print(f"  Candidate {i+1}: Score = {score:.4f}")
            
        best_idx = scores.index(max(scores))
        print(f"Winner: Candidate {best_idx+1} ({scores[best_idx]:.4f})")
        
        # Returns [Channels, Time]
        return wavs[best_idx]

    def stepwise_beam_search(self, prompt, total_duration=10, step_size=2, beam_width=4, expand_k=8):
        print(f"Running SBS (Batched): Step={step_size}s, Beam={beam_width}, Expand={expand_k}")
        
        # Initialize beams: List of tuples (score, audio)
        # Start with 1 empty beam
        beams = [(0.0, torch.zeros((1, 0)))] 
        
        num_steps = int(total_duration / step_size)

        for step in range(num_steps):
            current_target_duration = (step + 1) * step_size
            print(f"--- Step {step + 1}/{num_steps} (Target: {current_target_duration}s) ---")
            
            prompt_wavs_list = []
            next_prompts_list = []
            
            # Prepare expansion batch
            for _, audio in beams:
                for _ in range(expand_k):
                    prompt_wavs_list.append(audio)
                    next_prompts_list.append(prompt)

            # Stack into a single batch tensor [B*K, C, T] or [B*K, 1, C, T]
            prompt_wavs_batch = torch.stack(prompt_wavs_list)
            if prompt_wavs_batch.dim() == 4: # Remove extra dim if it crept in
                prompt_wavs_batch = prompt_wavs_batch.squeeze(1)

            # 1. Batch Generation
            if step == 0:
                full_batch = self.gen.generate_batch(next_prompts_list, duration=step_size)
            else:
                full_batch = self.gen.generate_continuation_batch(
                    prompt_wavs_batch, 
                    next_prompts_list, 
                    duration=current_target_duration
                )

            # 2. Batch Verification
            scores = self.curator.score_batch(full_batch, self.gen.sample_rate, next_prompts_list)

            # 3. Pruning
            candidates = []
            for i in range(len(scores)):
                # Store as [1, C, T] for next iteration's stacking compatibility
                candidates.append((scores[i], full_batch[i].unsqueeze(0)))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
            
            print(f"  Best Beam Score: {beams[0][0]:.4f}")

        # Remove the batch dimension before returning [1, C, T] -> [C, T]
        return beams[0][1].squeeze(0)