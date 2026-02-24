import torch
from delphi.latents import (
    LatentDataset,
    constructor,
    random_non_activating_windows,
    sampler,
)
from delphi.latents.latents import LatentData
import logging


class PadFilteredLatentDataset(LatentDataset):
    """LatentDataset that filters pad tokens from examples.

    Overrides _process_latent to handle pad tokens in two ways:
    - Examples with pad ratio <= ``max_pad_ratio`` are kept, but the pad
      tokens are stripped from their tokens/activations tensors.
    - Examples exceeding the threshold are discarded entirely.

    When non-activating examples fall short of the target after filtering,
    additional batches are sampled (with fresh seeds) until the target is
    met or the pool is exhausted.  Delphi source code stays unmodified.
    """

    MAX_NON_ACTIVATING_RETRIES = 100

    """Maximum fraction of pad tokens allowed in an example.  Examples at or
    below this ratio are kept (with pad tokens stripped); above it they are
    discarded."""

    def __init__(self, *args, max_pad_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pad_ratio = max_pad_ratio

    # ------------------------------------------------------------------
    # Pad-token helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_ratio(tokens: torch.Tensor, pad_id: int) -> float:
        """Fraction of tokens that are pad tokens."""
        return float((tokens == pad_id).sum()) / max(len(tokens), 1)

    def _filter_activating(self, examples, pad_id):
        """Filter and strip activating examples.  Returns (kept, n_dropped)."""
        kept = []
        for ex in examples:
            ratio = self._pad_ratio(ex.tokens, pad_id)
            if ratio > self.max_pad_ratio:
                continue
            if ratio > 0:
                keep_mask = ex.tokens != pad_id
                ex.tokens = ex.tokens[keep_mask]
                ex.activations = ex.activations[keep_mask]
                if ex.normalized_activations is not None:
                    ex.normalized_activations = ex.normalized_activations[keep_mask]
            kept.append(ex)
        return kept, len(examples) - len(kept)

    def _filter_non_activating(self, examples, pad_id):
        """Filter and strip non-activating examples.  Returns (kept, n_dropped)."""
        kept = []
        for ex in examples:
            ratio = self._pad_ratio(ex.tokens, pad_id)
            if ratio > self.max_pad_ratio:
                continue
            if ratio > 0:
                # Build mask *before* stripping so str_tokens stays in sync
                keep_mask = ex.tokens != pad_id
                ex.tokens = ex.tokens[keep_mask]
                ex.activations = ex.activations[keep_mask]
                ex.str_tokens = [
                    t for t, k in zip(ex.str_tokens, keep_mask.tolist()) if k
                ]
            kept.append(ex)
        return kept, len(examples) - len(kept)

    # ------------------------------------------------------------------

    def _process_latent(self, latent_data: LatentData):
        if self.tokens is None:
            raise ValueError("Tokens are not loaded")

        pad_id = self.tokenizer.pad_token_id

        # --- identical to LatentDataset._process_latent up to constructor ---
        from delphi.latents.latents import LatentRecord

        record = LatentRecord(latent_data.latent)

        n_active = len(latent_data.activation_data.activations)
        n_tokens = self.tokens.shape[1] * self.tokens.shape[0]
        record.per_token_frequency = n_active / n_tokens

        if self.neighbours is not None:
            record.set_neighbours(
                self.neighbours[latent_data.module][
                    str(latent_data.latent.latent_index)
                ],
            )

        record = constructor(
            record=record,
            activation_data=latent_data.activation_data,
            constructor_cfg=self.constructor_cfg,
            tokens=self.tokens,
            tokenizer=self.tokenizer,
            all_data=self.all_data[latent_data.module],
        )
        if record is None:
            return None

        # --- pad-token filtering ---
        n_act_before = len(record.examples)
        n_na_before = len(record.not_active)
        n_dropped_act = 0
        n_dropped_na = 0
        if pad_id is not None:
            record.examples, n_dropped_act = self._filter_activating(
                record.examples, pad_id
            )
            record.not_active, n_dropped_na = self._filter_non_activating(
                record.not_active, pad_id
            )

            # Replenish non-activating examples if we're short of the target.
            n_target = self.constructor_cfg.n_non_activating
            if len(record.not_active) < n_target:
                non_active_indices = self._get_non_active_indices(latent_data)
                reshaped = self.tokens.reshape(-1, self.constructor_cfg.example_ctx_len)
                # Track token content we already have to avoid duplicates.
                seen = {ex.tokens.numpy().tobytes() for ex in record.not_active}
                seed = 1000
                prev_count = len(record.not_active)
                stall_count = 0
                max_stalls = 3
                for _ in range(self.MAX_NON_ACTIVATING_RETRIES):
                    if len(record.not_active) >= n_target:
                        break
                    n_needed = n_target - len(record.not_active)
                    extras = random_non_activating_windows(
                        available_indices=non_active_indices,
                        reshaped_tokens=reshaped,
                        n_not_active=n_needed,
                        tokenizer=self.tokenizer,
                        seed=seed,
                    )
                    if not extras:
                        break  # pool is too small, retrying won't help
                    extras, _ = self._filter_non_activating(extras, pad_id)
                    for ex in extras:
                        key = ex.tokens.numpy().tobytes()
                        if key not in seen:
                            seen.add(key)
                            record.not_active.append(ex)
                    seed += 1
                    # Stop early if repeated attempts yield no new examples.
                    if len(record.not_active) == prev_count:
                        stall_count += 1
                        if stall_count >= max_stalls:
                            break
                    else:
                        stall_count = 0
                        prev_count = len(record.not_active)

                if len(record.not_active) < n_target:
                    logging.warning(
                        f"Latent {latent_data.latent}: could only get "
                        f"{len(record.not_active)}/{n_target} pad-free "
                        f"non-activating examples after retries"
                    )

            # Truncate to exactly n_target if we have more than needed.
            n_target = self.constructor_cfg.n_non_activating
            if len(record.not_active) > n_target:
                record.not_active = record.not_active[:n_target]

        if len(record.examples) < self.constructor_cfg.min_examples:
            logging.warning(
                f"Latent {latent_data.latent}: only {len(record.examples)} "
                f"examples after pad filtering (need {self.constructor_cfg.min_examples})"
            )
            return None

        record = sampler(record, self.sampler_cfg, self.tokenizer)

        logging.info(
            f"Latent {latent_data.latent} final stats: "
            f"{len(record.examples)}/{n_act_before} activating "
            f"(dropped {n_dropped_act}), "
            f"{len(record.not_active)}/{n_na_before} non-activating "
            f"(dropped {n_dropped_na}), "
            f"{len(record.train)} train, {len(record.test)} test"
        )
        return record

    def _get_non_active_indices(self, latent_data: LatentData) -> torch.Tensor:
        """Recompute the non-activating window indices for a latent.

        This mirrors the index computation at the top of
        ``delphi.latents.constructors.constructor``.
        """
        cache_ctx_len = self.tokens.shape[1]
        example_ctx_len = self.constructor_cfg.example_ctx_len
        flat_indices = (
            latent_data.activation_data.locations[:, 0] * cache_ctx_len
            + latent_data.activation_data.locations[:, 1]
        )
        ctx_indices = flat_indices // example_ctx_len
        reshaped_tokens = self.tokens.reshape(-1, example_ctx_len)
        n_windows = reshaped_tokens.shape[0]
        unique_batch_pos = ctx_indices.unique()
        mask = torch.ones(n_windows, dtype=torch.bool)
        mask[unique_batch_pos] = False
        return mask.nonzero(as_tuple=False).squeeze(-1)
