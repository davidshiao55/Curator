import torch
import math
import typing as tp

class InferenceSearch:
    """
    Implements Test-Time Compute scaling strategies aligned with:
    'Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters'
    
    Algorithms:
    1. Best-of-N (Parallel Sampling)
    2. Beam Search (BFS-V: Prune-then-Expand to maintain fixed N)
    3. Lookahead Search (Simulation-based scoring)
    """
    def __init__(self, generator, verifier):
        self.gen = generator
        self.curator = verifier

    def best_of_n(self, prompt: str, n_candidates: int = 4, duration: int = 10) -> torch.Tensor:
        """
        Parallel Sampling (Best-of-N).
        
        Logic:
        1. Sample N outputs independenty.
        2. Score all N.
        3. Return the best one.
        
        Args:
            n_candidates (N): Total generation budget.
        """
        print(f"Running Best-of-{n_candidates} (Parallel Sampling)...")
        
        # 1. Generate Batch (Parallel)
        prompts = [prompt] * n_candidates
        wavs = self.gen.generate_batch(prompts, duration)
        
        # 2. Score
        scores = self.curator.score_batch(wavs, self.gen.sample_rate, prompts)
        
        # 3. Select Argmax
        # Note: We don't use 'Best-of-N Weighted' (marginalization) because 
        # continuous audio samples are rarely identical.
        best_idx = scores.index(max(scores))
        print(f"  Best Score: {scores[best_idx]:.4f}")
        
        return wavs[best_idx]

    def beam_search(self, prompt: str, total_duration: int = 10, step_size: int = 2, 
                    generation_budget: int = 16, beam_width: int = 4) -> torch.Tensor:
        """
        Beam Search (BFS-V).
        
        Logic [DeepMind Section 5.2]:
        1. Sample N candidates.
        2. Loop:
           a. Score N candidates.
           b. Select top K = N/M parents.
           c. Expand each parent M times -> N new candidates.
        """
        return self._tree_search_core(
            prompt=prompt,
            total_duration=total_duration,
            step_size=step_size,
            generation_budget=generation_budget,
            beam_width=beam_width,
            use_lookahead=False
        )

    def lookahead_search(self, prompt: str, total_duration: int = 10, step_size: int = 2,
                         generation_budget: int = 16, beam_width: int = 4, lookahead_k: int = 1) -> torch.Tensor:
        """
        Lookahead Search.
        
        Logic [DeepMind Section 5.2]:
        - Same structure as Beam Search.
        - Scoring: Instead of scoring the current state 's', we run a greedy simulation 
          'k' steps forward to 's_prime', and use score(s_prime) as the value for 's'.
        """
        return self._tree_search_core(
            prompt=prompt,
            total_duration=total_duration,
            step_size=step_size,
            generation_budget=generation_budget,
            beam_width=beam_width,
            use_lookahead=True,
            lookahead_k=lookahead_k
        )

    def _tree_search_core(self, prompt: str, total_duration: int, step_size: int, 
                          generation_budget: int, beam_width: int, 
                          use_lookahead: bool = False, lookahead_k: int = 1) -> torch.Tensor:
        
        # 1. Configuration Validation
        if generation_budget % beam_width != 0:
            original_n = generation_budget
            generation_budget = (generation_budget // beam_width) * beam_width
            if generation_budget == 0: generation_budget = beam_width # Minimum constraint
            print(f"[Warning] Adjusted Budget N from {original_n} to {generation_budget} to be divisible by M={beam_width}.")

        n_parents = generation_budget // beam_width # K = N / M
        num_steps = int(total_duration / step_size)
        method_name = f"Lookahead (k={lookahead_k})" if use_lookahead else "Beam Search"
        
        print(f"Running {method_name}: Budget(N)={generation_budget}, Width(M)={beam_width} => Parents(K)={n_parents}")

        # Initialize: 'candidates' holds tuples of (score, audio_tensor [1, C, T])
        # We start with N "empty" candidates implicitly handled in the first loop
        candidates = [] 

        for step in range(num_steps):
            current_target_duration = (step + 1) * step_size
            print(f"--- Step {step + 1}/{num_steps} (Target: {current_target_duration}s) ---")
            
            # --- Phase A: Expansion (Generate N candidates) ---
            prompt_wavs_list = []
            next_prompts_list = []
            
            if step == 0:
                # Step 0: Just N independent generations
                next_prompts_list = [prompt] * generation_budget
                
                # Generate N
                current_batch = self.gen.generate_batch(next_prompts_list, duration=step_size)
                
            else:
                # Step > 0: Expand K parents M times each
                for _, audio in candidates:
                    for _ in range(beam_width):
                        prompt_wavs_list.append(audio)
                        next_prompts_list.append(prompt)
                
                # Stack parents [N, C, T]
                prompt_wavs_batch = torch.stack(prompt_wavs_list)
                if prompt_wavs_batch.dim() == 4: prompt_wavs_batch = prompt_wavs_batch.squeeze(1)
                
                # Generate Continuation (N)
                # Note: We assume standard temperature sampling here to get diversity among the M children
                current_batch = self.gen.generate_continuation_batch(
                    prompt_wavs_batch, 
                    next_prompts_list, 
                    duration=current_target_duration
                )

            # --- Phase B: Verification (Score N candidates) ---
            
            if use_lookahead and step < num_steps - 1:
                # [Lookahead Logic]
                # Simulate 'k' steps into the future using GREEDY decoding (temp=0)
                sim_duration = min(total_duration, current_target_duration + (lookahead_k * step_size))
                
                # Run Simulation (Greedy)
                # We assume generate_continuation_batch accepts temperature=0 for greedy
                sim_batch = self.gen.generate_continuation_batch(
                    current_batch,
                    next_prompts_list,
                    duration=sim_duration,
                    temperature=0.0 # Greedy for deterministic evaluation
                )
                
                # Score the simulated future
                scores = self.curator.score_batch(sim_batch, self.gen.sample_rate, next_prompts_list)
                
            else:
                # [Standard Beam Logic]
                # Score the current state
                scores = self.curator.score_batch(current_batch, self.gen.sample_rate, next_prompts_list)

            # --- Phase C: Pruning (Select Top K Parents) ---
            
            # Pack data: (score, actual_audio_tensor)
            # IMPORTANT: We store 'current_batch', NOT 'sim_batch'.
            # We only used 'sim_batch' to peek at the score.
            scored_candidates = []
            for i in range(len(scores)):
                scored_candidates.append((scores[i], current_batch[i].unsqueeze(0)))
            
            # Sort descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Select K parents
            candidates = scored_candidates[:n_parents]
            
            print(f"  Top Score: {candidates[0][0]:.4f} | Cutoff Score: {candidates[-1][0]:.4f}")

        # Final Selection: Return the absolute best candidate from the final set
        best_audio = candidates[0][1].squeeze(0)
        return best_audio