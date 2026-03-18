from dataclasses import dataclass
import math
from typing import Dict, Optional
from transformers import TrainerCallback
from peft.tuners.lora import LoraLayer
import torch.nn.functional as F
import torch.nn as nn
import torch
import gc
import wandb

VALID_TOPK_MODES = {"topk", "batchtopk", "seqtopk"}


@dataclass
class TopKForwardState:
    base_out: torch.Tensor
    hidden_pre: torch.Tensor
    topk_scores: torch.Tensor
    dense_latents: torch.Tensor
    soft_gates: Optional[torch.Tensor]
    hard_gates: Optional[torch.Tensor]
    gates: Optional[torch.Tensor]
    sparse_latents: torch.Tensor
    decoder_norms: torch.Tensor
    output: torch.Tensor
    k: int
    tau: float
    nonzero_fraction: torch.Tensor
    active_latents: torch.Tensor
    dead_latents: torch.Tensor
    decoder_norm_drift: torch.Tensor

    def detached(self) -> "TopKForwardState":
        return TopKForwardState(
            base_out=self.base_out.detach(),
            hidden_pre=self.hidden_pre.detach(),
            topk_scores=self.topk_scores.detach(),
            dense_latents=self.dense_latents.detach(),
            soft_gates=None if self.soft_gates is None else self.soft_gates.detach(),
            hard_gates=None if self.hard_gates is None else self.hard_gates.detach(),
            gates=None if self.gates is None else self.gates.detach(),
            sparse_latents=self.sparse_latents.detach(),
            decoder_norms=self.decoder_norms.detach(),
            output=self.output.detach(),
            k=int(self.k),
            tau=float(self.tau),
            nonzero_fraction=self.nonzero_fraction.detach(),
            active_latents=self.active_latents.detach(),
            dead_latents=self.dead_latents.detach(),
            decoder_norm_drift=self.decoder_norm_drift.detach(),
        )


def _normalize_topk_mode(topk_mode: str) -> str:
    mode = str(topk_mode).strip().lower()
    if mode not in VALID_TOPK_MODES:
        allowed = ", ".join(sorted(VALID_TOPK_MODES))
        raise ValueError(f"Invalid topk_mode '{topk_mode}'. Expected one of: {allowed}.")
    return mode


def _soft_topk_mass(z, k, tau, topk_mode: str = "topk"):
    # compute in fp32 for stability, then rescale to sum=k
    mode = _normalize_topk_mode(topk_mode)
    z_fp32 = z.float() / max(tau, 1e-6)
    if mode == "topk":
        g = torch.softmax(z_fp32, dim=-1)
    elif mode == "batchtopk":
        # Batch-shared gating: pick one latent distribution for the whole batch.
        if z_fp32.dim() <= 1:
            z_scores = z_fp32
        else:
            reduce_dims = tuple(range(z_fp32.dim() - 1))
            z_scores = z_fp32.mean(dim=reduce_dims)
        g_scores = torch.softmax(z_scores, dim=-1)
        view_shape = [1] * max(z_fp32.dim() - 1, 0) + [z_fp32.shape[-1]]
        g = g_scores.view(*view_shape).expand_as(z_fp32)
    else:
        # Sequence-shared gating: one latent distribution per sample.
        if z_fp32.dim() <= 2:
            g = torch.softmax(z_fp32, dim=-1)
        else:
            reduce_dims = tuple(range(1, z_fp32.dim() - 1))
            z_scores = z_fp32.mean(dim=reduce_dims)
            g_scores = torch.softmax(z_scores, dim=-1)
            view_shape = [z_fp32.shape[0]] + [1] * (z_fp32.dim() - 2) + [
                z_fp32.shape[-1]
            ]
            g = g_scores.view(*view_shape).expand_as(z_fp32)
    g = g * (float(k) / (g.sum(dim=-1, keepdim=True) + 1e-8))
    return g.to(z.dtype)


def _hard_topk_mask(z, k, topk_mode: str = "topk"):
    # returns 0/1 mask with exactly k ones along last dim
    mode = _normalize_topk_mode(topk_mode)
    k_safe = int(max(0, min(int(k), int(z.shape[-1]))))
    if k_safe <= 0:
        return torch.zeros_like(z)

    if mode == "topk":
        idx = z.topk(k_safe, dim=-1).indices
        hard = torch.zeros_like(z)
        return hard.scatter_(-1, idx, 1.0)

    if mode == "batchtopk":
        # Batch-shared mask built from latent scores pooled across batch/sequence dims.
        if z.dim() <= 1:
            z_scores = z
        else:
            reduce_dims = tuple(range(z.dim() - 1))
            z_scores = z.mean(dim=reduce_dims)
        idx = z_scores.topk(k_safe, dim=-1).indices
        hard_scores = torch.zeros_like(z_scores).scatter_(-1, idx, 1.0)
        if z.dim() <= 1:
            return hard_scores
        view_shape = [1] * (z.dim() - 1) + [z.shape[-1]]
        return hard_scores.view(*view_shape).expand_as(z)

    # Sequence-shared mask: one top-k mask per sample, broadcast across sequence.
    if z.dim() <= 2:
        idx = z.topk(k_safe, dim=-1).indices
        hard = torch.zeros_like(z)
        return hard.scatter_(-1, idx, 1.0)

    reduce_dims = tuple(range(1, z.dim() - 1))
    z_scores = z.mean(dim=reduce_dims)
    idx = z_scores.topk(k_safe, dim=-1).indices
    hard_scores = torch.zeros_like(z_scores).scatter_(-1, idx, 1.0)
    view_shape = [z.shape[0]] + [1] * (z.dim() - 2) + [z.shape[-1]]
    return hard_scores.view(*view_shape).expand_as(z)


def _mean_abs_pairwise_cosine(weight: torch.Tensor, vector_dim: int) -> torch.Tensor:
    if weight.numel() == 0:
        return weight.new_zeros(())

    if vector_dim == 0:
        vectors = weight.transpose(0, 1)
    elif vector_dim == 1:
        vectors = weight
    else:
        raise ValueError(f"Unsupported vector_dim={vector_dim}; expected 0 or 1.")

    if vectors.dim() != 2 or vectors.shape[0] < 2:
        return weight.new_zeros(())

    normed = F.normalize(vectors.float(), p=2, dim=-1, eps=1e-8)
    sims = normed @ normed.T
    triu = torch.triu_indices(
        sims.shape[0], sims.shape[1], offset=1, device=sims.device
    )
    if triu.numel() == 0:
        return weight.new_zeros(())
    return sims[triu[0], triu[1]].abs().mean().to(dtype=weight.dtype)


class TopKModule(nn.Module):
    """Base class for Top-K modules."""

    def __init__(self, k, topk_mode: str = "topk"):
        super().__init__()
        self.k = k
        self.topk_mode = _normalize_topk_mode(topk_mode)

    def set_mode(self, topk_mode: str) -> None:
        self.topk_mode = _normalize_topk_mode(topk_mode)

    def forward(self, x):
        mask = _hard_topk_mask(x, self.k, self.topk_mode)
        return x * mask


class TopKLoRALinearSTE(nn.Module):
    """
    LoRA with straight-through Top-K gating over the latent dim (r).
    - Forward: hard top-k mask (exact k active channels).
    - Backward: gradients flow through a soft surrogate.
    - Supports k and temperature schedules driven by `progress` (0..1).
    """

    def __init__(
        self,
        base: LoraLayer,
        *,
        layer_name: str,
        k: int,
        temperature: float = 1.0,
        # "constant" | "linear" | "exp" | "cubic"
        temperature_schedule: str = "linear",
        k_schedule: str = "constant",  # "constant" | "linear" | "exp" | "cubic"
        # fraction of training used to warm k from k_init to k_final
        k_warmup_frac: float = 0.2,
        # target k at progress=1
        k_final: Optional[int] = None,
        hard_eval: bool = True,  # use hard mask in eval
        relu_latents: bool = True,  # force z >= 0
        alpha_over_r: bool = True,  # scaling mode
        # optional target temperature at progress=1
        temperature_final: Optional[float] = None,
        is_topk_experiment: bool = False,
        topk_mode: str = "topk",
        sae_style: bool = False,
        sae_decoder_init_norm: Optional[float] = 0.1,
        sae_rescale_by_decoder_norm: bool = True,
        sae_unit_norm_decoder: bool = False,
        sae_use_latent_bias: bool = True,
        sae_use_input_center: bool = False,
    ):
        super().__init__()
        self.lora_module = base
        self.base_layer = base.base_layer
        adapter = (
            base.active_adapter
            if isinstance(base.active_adapter, str)
            else base.active_adapter[0]
        )
        self.adapter_name = adapter
        self.is_topk_experiment = is_topk_experiment
        self.topk_mode = _normalize_topk_mode(topk_mode)

        self.A_module = base.lora_A[adapter]
        self.B_module = base.lora_B[adapter]
        self.dropout = (
            base.lora_dropout[adapter]
            if hasattr(base, "lora_dropout") and adapter in base.lora_dropout
            else nn.Identity()
        )

        self.r = int(base.r[adapter])
        self.alpha = float(base.lora_alpha[adapter])
        self.in_features = int(self.A_module.weight.shape[-1])
        self.out_features = int(self.B_module.weight.shape[0])
        self.k_init = int(k)
        self.k_final = int(k_final) if k_final is not None else int(k)
        self.k_schedule = k_schedule
        self.t0 = float(temperature)
        self.t_final = (
            float(temperature_final) if temperature_final is not None else 0.1 * self.t0
        )
        self.temperature_schedule = temperature_schedule
        self.k_warmup_frac = max(float(k_warmup_frac), 1e-6)
        self.hard_eval = hard_eval
        self.relu_latents = relu_latents
        self.sae_style = bool(sae_style)
        self.sae_decoder_init_norm = (
            None
            if sae_decoder_init_norm is None
            else float(sae_decoder_init_norm)
        )
        self.sae_rescale_by_decoder_norm = bool(sae_rescale_by_decoder_norm)
        self.sae_unit_norm_decoder = bool(sae_unit_norm_decoder)
        self.sae_use_latent_bias = bool(sae_use_latent_bias)
        self.sae_use_input_center = bool(sae_use_input_center)
        self.scale = (
            (self.alpha / self.r)
            if alpha_over_r
            else (self.alpha / max(self.k_final, 1))
        )
        self.layer_name = layer_name
        self.topk = TopKModule(k_final, topk_mode=self.topk_mode)

        self.latent_bias = nn.Parameter(torch.zeros(self.r))
        self.input_center = nn.Parameter(torch.zeros(self.in_features))

        # Progress variable (0..1)
        self.register_buffer("progress", torch.tensor(0.0))
        self.register_buffer("last_frac_grad_nonzero", torch.tensor(0.0))
        self._progress_scalar: float = 0.0

        # Transient caches for regs/logging
        self._z_live: Optional[torch.Tensor] = None
        self._g_soft_live: Optional[torch.Tensor] = None
        self._last_z: Optional[torch.Tensor] = None
        self._last_g_soft: Optional[torch.Tensor] = None
        self._last_ghard_mean: torch.Tensor = torch.tensor(0.0)
        self._last_hidden_pre: Optional[torch.Tensor] = None
        self._last_topk_scores: Optional[torch.Tensor] = None
        self._last_g_hard: Optional[torch.Tensor] = None
        self._last_z_sparse: Optional[torch.Tensor] = None
        self._last_decoder_norms: Optional[torch.Tensor] = None
        self._last_nonzero_fraction: torch.Tensor = torch.tensor(0.0)
        self._last_active_latents: torch.Tensor = torch.tensor(0)
        self._last_dead_latents: torch.Tensor = torch.tensor(0)
        self._last_decoder_norm_drift: torch.Tensor = torch.tensor(0.0)
        self._last_forward_state: Optional[TopKForwardState] = None

        if self.sae_style:
            self._initialize_sae_parameters()

    def _wrapper_state_tensors(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters(recurse=False):
            state[name] = param
        for name, buf in self.named_buffers(recurse=False):
            state[name] = buf
        return state

    def _wrapper_alias_state_keys(self) -> Dict[str, str]:
        return {
            f"lora_sae_latent_bias.{self.adapter_name}": "latent_bias",
            f"lora_sae_input_center.{self.adapter_name}": "input_center",
            f"lora_sae_progress.{self.adapter_name}": "progress",
            f"lora_sae_last_frac_grad_nonzero.{self.adapter_name}": "last_frac_grad_nonzero",
        }

    def _is_square_crosscoder(self) -> bool:
        return self.in_features == self.out_features

    def _should_use_input_center(self) -> bool:
        return bool(getattr(self, "sae_style", False)) and bool(
            getattr(self, "sae_use_input_center", False)
        )

    def _should_use_latent_bias(self) -> bool:
        return bool(getattr(self, "sae_style", False)) and bool(
            getattr(self, "sae_use_latent_bias", False)
        )

    def _should_rescale_by_decoder_norm(self) -> bool:
        return bool(getattr(self, "sae_style", False)) and bool(
            getattr(self, "sae_rescale_by_decoder_norm", False)
        )

    def sae_noop_init_scale(self) -> float:
        if not bool(getattr(self, "sae_style", False)):
            return 1.0
        return min(1.0, 1.0 / math.sqrt(max(self.r, 1)))

    def decoder_maintenance_target_norm(self) -> Optional[float]:
        if not bool(getattr(self, "sae_style", False)):
            return None
        if bool(getattr(self, "sae_unit_norm_decoder", False)):
            return 1.0
        return None

    def _initialize_sae_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.B_module.weight, a=math.sqrt(5))
            decoder_template = self.B_module.weight.detach().clone()
            if self.sae_decoder_init_norm is not None:
                dec_norms = decoder_template.norm(dim=0, keepdim=True).clamp_min(1e-8)
                decoder_template.mul_(self.sae_decoder_init_norm / dec_norms)

            if self._is_square_crosscoder():
                self.A_module.weight.copy_(decoder_template.t())
            else:
                nn.init.kaiming_uniform_(self.A_module.weight, a=math.sqrt(5))
                if self.sae_decoder_init_norm is not None:
                    row_norms = self.A_module.weight.norm(dim=-1, keepdim=True).clamp_min(
                        1e-8
                    )
                    self.A_module.weight.mul_(self.sae_decoder_init_norm / row_norms)

            # Preserve LoRA's safe "small delta" start by keeping the decoder tiny
            # at initialization while leaving the encoder directions meaningful.
            self.B_module.weight.copy_(decoder_template * self.sae_noop_init_scale())
            self.latent_bias.zero_()
            self.input_center.zero_()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Override to delegate to the wrapped lora_module's state_dict.
        This makes the wrapper transparent to PEFT saving.
        """
        lora_state = self.lora_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if destination is None:
            destination = lora_state

        wrapper_state = self._wrapper_state_tensors()
        alias_map = self._wrapper_alias_state_keys()
        for name, value in wrapper_state.items():
            tensor = value if keep_vars else value.detach()
            destination[prefix + name] = tensor
            for alias_key, target_name in alias_map.items():
                if target_name == name:
                    destination[prefix + alias_key] = tensor

        return lora_state

    def load_state_dict(self, state_dict, strict=True):
        """
        Override to properly load both lora_module weights and our buffers.
        """
        wrapper_state = self._wrapper_state_tensors()
        alias_map = self._wrapper_alias_state_keys()
        our_state = {}
        lora_state = {}

        for k, v in state_dict.items():
            if k in wrapper_state:
                our_state[k] = v
                continue

            matched_alias = False
            for alias_key, target_name in alias_map.items():
                if k == alias_key:
                    our_state[target_name] = v
                    matched_alias = True
                    break

            if matched_alias:
                continue

            if any(k.endswith(alias_key) for alias_key in alias_map):
                continue

            if any(k.endswith(name) for name in wrapper_state):
                continue

            if any(k.endswith(alias_key) for alias_key in alias_map):
                continue

            if k in ["progress", "last_frac_grad_nonzero"]:
                our_state[k] = v
            else:
                lora_state[k] = v

        if lora_state:
            self.lora_module.load_state_dict(lora_state, strict=strict)
            adapter = (
                self.lora_module.active_adapter
                if isinstance(self.lora_module.active_adapter, str)
                else self.lora_module.active_adapter[0]
            )
            self.A_module = self.lora_module.lora_A[adapter]
            self.B_module = self.lora_module.lora_B[adapter]
            self.adapter_name = adapter

        for name, target in wrapper_state.items():
            if name not in our_state:
                continue
            value = our_state[name]
            if isinstance(target, nn.Parameter):
                target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            else:
                target.copy_(value.to(device=target.device, dtype=target.dtype))

        self._progress_scalar = float(self.progress.detach().cpu().item())

        return

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Hook called by PyTorch during state_dict collection.
        Delegate to lora_module to maintain namespace compatibility.
        """
        for name, param in self.lora_module.named_parameters():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()

        wrapper_state = self._wrapper_state_tensors()
        alias_map = self._wrapper_alias_state_keys()
        for name, value in wrapper_state.items():
            if value is None:
                continue
            tensor = value if keep_vars else value.detach()
            destination[prefix + name] = tensor
            for alias_key, target_name in alias_map.items():
                if target_name == name:
                    destination[prefix + alias_key] = tensor

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Hook called by PyTorch during load_state_dict.
        """
        wrapper_state = self._wrapper_state_tensors()
        alias_map = self._wrapper_alias_state_keys()
        wrapper_keys = {prefix + name for name in wrapper_state}
        alias_keys = {prefix + alias_key for alias_key in alias_map}

        for name, target in wrapper_state.items():
            value = None
            direct_key = prefix + name
            if direct_key in state_dict:
                value = state_dict[direct_key]
            else:
                for alias_key, target_name in alias_map.items():
                    if target_name == name and prefix + alias_key in state_dict:
                        value = state_dict[prefix + alias_key]
                        break

            if value is None:
                continue

            if isinstance(target, nn.Parameter):
                target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            else:
                target.copy_(value.to(device=target.device, dtype=target.dtype))

        if isinstance(self.progress, torch.Tensor):
            self._progress_scalar = float(self.progress.detach().cpu().item())

        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key not in wrapper_keys and key not in alias_keys
        }

        missing_before = len(missing_keys)
        unexpected_before = len(unexpected_keys)
        self.lora_module._load_from_state_dict(
            filtered_state,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        adapter = (
            self.lora_module.active_adapter
            if isinstance(self.lora_module.active_adapter, str)
            else self.lora_module.active_adapter[0]
        )
        self.A_module = self.lora_module.lora_A[adapter]
        self.B_module = self.lora_module.lora_B[adapter]
        self.adapter_name = adapter
        if strict:
            missing_keys[:] = [
                key
                for key in missing_keys
                if key not in wrapper_keys and key not in alias_keys
            ]
            unexpected_keys[:] = [
                key
                for key in unexpected_keys
                if key not in wrapper_keys and key not in alias_keys
            ]

        if len(missing_keys) < missing_before or len(unexpected_keys) < unexpected_before:
            self._progress_scalar = float(self.progress.detach().cpu().item())

    # -------- Progress control --------
    def set_progress(self, p: float):
        """Set training progress in [0, 1]."""
        value = float(min(max(p, 0.0), 1.0))
        self._progress_scalar = value
        self.progress.fill_(value)

    @property
    def progress_scalar(self) -> float:
        return self._progress_scalar

    # -------- Scheduling --------
    def _tau(self):
        p = self._progress_scalar
        # If constant or already at target, keep t0
        if (
            self.temperature_schedule == "constant"
            or abs(self.t0 - self.t_final) < 1e-12
        ):
            return self.t0
        if self.temperature_schedule == "linear":
            # linear interpolation from t0 to t_final
            return float(self.t0 + (self.t_final - self.t0) * p)
        if self.temperature_schedule == "cubic":
            # cubic interpolation, slower start, faster end
            return float(self.t0 + (self.t_final - self.t0) * (p**3))
        if self.temperature_schedule == "exp":
            # geometric interpolation (monotonic)
            ratio = max(self.t_final, 1e-12) / max(self.t0, 1e-12)
            return float(self.t0 * (ratio**p))
        return self.t0

    def _current_k(self):
        # p in [0,1]; compress so k finishes warming up by 5% of training
        p = self._progress_scalar
        warm = min(
            p / self.k_warmup_frac, 1.0
        )  # 0..1 grows only during first k_warmup_frac
        # warm = min(p / 0.05, 1.0)  # 0..1 grows only during first 5%
        if self.k_schedule == "constant" or self.k_init == self.k_final:
            return self.k_init
        if self.k_schedule == "linear":
            return int(round(self.k_init + (self.k_final - self.k_init) * warm))
        if self.k_schedule == "cubic":
            return int(round(self.k_init + (self.k_final - self.k_init) * (warm**3)))
        if self.k_schedule == "exp":
            ratio = (self.k_final / max(self.k_init, 1)) ** warm
            return int(round(self.k_init * ratio))
        return self.k_init

    def decoder_norms(self, eps: float = 1e-8) -> torch.Tensor:
        return self.B_module.weight.norm(dim=0).clamp_min(eps)

    def normalize_decoder_(
        self, target_norm: Optional[float] = None, eps: float = 1e-8
    ) -> "TopKLoRALinearSTE":
        if target_norm is None:
            target_norm = 1.0
        with torch.no_grad():
            norms = self.decoder_norms(eps=eps).unsqueeze(0)
            self.B_module.weight.data.mul_(float(target_norm) / norms)
        return self

    def fold_decoder_norms_(self, eps: float = 1e-8) -> "TopKLoRALinearSTE":
        with torch.no_grad():
            norms = self.decoder_norms(eps=eps)
            self.B_module.weight.data.div_(norms.unsqueeze(0))
            self.A_module.weight.data.mul_(norms.unsqueeze(-1))
            if self._should_use_latent_bias():
                self.latent_bias.data.mul_(norms)
        return self

    def project_decoder_grad_to_tangent_(self, eps: float = 1e-8) -> float:
        grad = self.B_module.weight.grad
        if grad is None:
            return 0.0

        with torch.no_grad():
            unit = self.B_module.weight / self.decoder_norms(eps=eps).unsqueeze(0)
            parallel = (grad * unit).sum(dim=0, keepdim=True) * unit
            grad.sub_(parallel)
            return float(parallel.norm().detach().cpu().item())

    def post_step_decoder_renorm_(
        self, target_norm: Optional[float] = None
    ) -> "TopKLoRALinearSTE":
        target = (
            float(target_norm)
            if target_norm is not None
            else self.decoder_maintenance_target_norm()
        )
        if target is None:
            return self
        return self.normalize_decoder_(target_norm=target)

    def decoder_pairwise_cosine_similarity(self) -> torch.Tensor:
        return _mean_abs_pairwise_cosine(self.B_module.weight, vector_dim=0)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        x_lora = self.dropout(x)
        if self._should_use_input_center():
            x_lora = x_lora - self.input_center.to(device=x_lora.device, dtype=x_lora.dtype)

        hidden_pre = F.linear(x_lora, self.A_module.weight)
        if self._should_use_latent_bias():
            hidden_pre = hidden_pre + self.latent_bias.to(
                device=hidden_pre.device, dtype=hidden_pre.dtype
            )
        return hidden_pre

    def _topk_scores(self, hidden_pre: torch.Tensor, decoder_norms: torch.Tensor) -> torch.Tensor:
        if not self._should_rescale_by_decoder_norm():
            return hidden_pre
        return hidden_pre * decoder_norms.to(
            device=hidden_pre.device, dtype=hidden_pre.dtype
        )

    def _activate_latents(self, topk_scores: torch.Tensor) -> torch.Tensor:
        if self.relu_latents:
            return F.relu(topk_scores)
        return topk_scores

    def apply_topk(self, dense_latents: torch.Tensor):
        k_now = int(self._current_k())
        tau = float(self._tau())

        if not bool(getattr(self, "is_topk_experiment", False)):
            return None, None, None, dense_latents, k_now, tau

        hard_eval = bool(getattr(self, "hard_eval", True))
        topk_mode = _normalize_topk_mode(getattr(self, "topk_mode", "topk"))

        if not self.training and hard_eval:
            if hasattr(self, "topk"):
                self.topk.k = int(k_now)
                self.topk.set_mode(topk_mode)
            hard = _hard_topk_mask(dense_latents, k_now, topk_mode)
            return None, hard, hard, dense_latents * hard, k_now, tau

        soft = _soft_topk_mass(dense_latents, k_now, tau, topk_mode)
        hard = _hard_topk_mask(dense_latents, k_now, topk_mode)
        gates = hard + soft - soft.detach()
        return soft, hard, gates, dense_latents * gates, k_now, tau

    def decode_latents(
        self, latents: torch.Tensor, decoder_norms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        latents_in = latents
        if self._should_rescale_by_decoder_norm():
            if decoder_norms is None:
                decoder_norms = self.decoder_norms()
            latents_in = latents_in / decoder_norms.to(
                device=latents_in.device, dtype=latents_in.dtype
            )
        return F.linear(latents_in, self.B_module.weight) * self.scale

    def recompute_output_from_sparse_latents(
        self,
        x: torch.Tensor,
        sparse_latents: torch.Tensor,
        *,
        base_out: Optional[torch.Tensor] = None,
        decoder_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_out is None:
            base_out = self.base_layer(x)
        return base_out + self.decode_latents(sparse_latents, decoder_norms=decoder_norms)

    def _decoder_norm_drift(self, decoder_norms: torch.Tensor) -> torch.Tensor:
        target = (
            1.0
            if bool(getattr(self, "sae_unit_norm_decoder", False))
            else getattr(self, "sae_decoder_init_norm", None)
        )
        if target is None:
            return decoder_norms.new_zeros(())
        return (decoder_norms - float(target)).abs().mean()

    def _count_active_latents(self, sparse_latents: torch.Tensor) -> torch.Tensor:
        if sparse_latents.numel() == 0:
            return torch.zeros((), device=sparse_latents.device, dtype=torch.long)
        reduce_dims = tuple(range(sparse_latents.dim() - 1))
        active_mask = (sparse_latents.abs() > 0).any(dim=reduce_dims)
        return active_mask.sum()

    def _cache_forward_state(self, state: TopKForwardState) -> None:
        self._z_live = state.dense_latents
        self._g_soft_live = state.soft_gates

        detached = state.detached()
        self._last_forward_state = detached
        self._last_hidden_pre = detached.hidden_pre
        self._last_topk_scores = detached.topk_scores
        self._last_z = detached.dense_latents
        self._last_g_soft = detached.soft_gates
        self._last_g_hard = detached.hard_gates
        self._last_z_sparse = detached.sparse_latents
        self._last_decoder_norms = detached.decoder_norms
        self._last_nonzero_fraction = detached.nonzero_fraction
        self._last_active_latents = detached.active_latents
        self._last_dead_latents = detached.dead_latents
        self._last_decoder_norm_drift = detached.decoder_norm_drift

        if detached.hard_gates is not None:
            self._last_ghard_mean = detached.hard_gates.mean()
        else:
            self._last_ghard_mean = detached.nonzero_fraction

    def forward_with_state(
        self, x: torch.Tensor, *, cache: bool = True
    ) -> TopKForwardState:
        base_out = self.base_layer(x)
        hidden_pre = self.encode_pre(x)
        decoder_norms = self.decoder_norms().to(
            device=hidden_pre.device, dtype=hidden_pre.dtype
        )
        topk_scores = self._topk_scores(hidden_pre, decoder_norms)
        dense_latents = self._activate_latents(topk_scores)
        soft_gates, hard_gates, gates, sparse_latents, k_now, tau = self.apply_topk(
            dense_latents
        )
        output = self.recompute_output_from_sparse_latents(
            x,
            sparse_latents,
            base_out=base_out,
            decoder_norms=decoder_norms,
        )
        nonzero_fraction = (sparse_latents.abs() > 0).float().mean()
        active_latents = self._count_active_latents(sparse_latents)
        dead_latents = sparse_latents.new_tensor(
            sparse_latents.shape[-1], dtype=torch.long
        ) - active_latents
        state = TopKForwardState(
            base_out=base_out,
            hidden_pre=hidden_pre,
            topk_scores=topk_scores,
            dense_latents=dense_latents,
            soft_gates=soft_gates,
            hard_gates=hard_gates,
            gates=gates,
            sparse_latents=sparse_latents,
            decoder_norms=decoder_norms,
            output=output,
            k=int(k_now),
            tau=float(tau),
            nonzero_fraction=nonzero_fraction,
            active_latents=active_latents,
            dead_latents=dead_latents,
            decoder_norm_drift=self._decoder_norm_drift(decoder_norms),
        )
        if cache:
            self._cache_forward_state(state)
        return state

    # -------- Forward --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state(x, cache=True).output

    def get_gate_stats(self):
        if self._last_z is None or self._last_z_sparse is None:
            return {}
        k = self._current_k()
        r = self.r
        frac_active = float(self._last_ghard_mean) / max(k / r, 1e-8)
        active_latents = int(self._last_active_latents.item())
        dead_latents = int(self._last_dead_latents.item())
        avg_usage = float(self._last_nonzero_fraction.item())
        decoder_norm_mean = (
            float(self._last_decoder_norms.mean().item())
            if self._last_decoder_norms is not None
            else 0.0
        )
        decoder_norm_std = (
            float(self._last_decoder_norms.std().item())
            if self._last_decoder_norms is not None and self._last_decoder_norms.numel() > 1
            else 0.0
        )
        return {
            "k": k,
            "r": r,
            "tau": self._tau(),
            "frac_active_vs_target": frac_active,
            "cdec": float(self.decoder_pairwise_cosine_similarity().item()),
            "active_latents": active_latents,
            "dead_latents": dead_latents,
            "avg_usage": avg_usage,
            "actual_nonzero_fraction": avg_usage,
            "decoder_norm_mean": decoder_norm_mean,
            "decoder_norm_std": decoder_norm_std,
            "decoder_norm_drift": float(self._last_decoder_norm_drift.item()),
            "topk_mode": self.topk_mode,
        }


class MemoryClearCallback(TrainerCallback):
    """Memory management callback"""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.gradient_accumulation_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return control


class TopKProgressCallback(TrainerCallback):
    """Update training progress in TopK modules"""

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and state.max_steps:
            p = state.global_step / state.max_steps
            for m in model.modules():
                if isinstance(m, TopKLoRALinearSTE):
                    m.set_progress(p)


class DecoderNormMaintenanceCallback(TrainerCallback):
    """Maintain decoder norm constraints for SAE-style TopKLoRA modules."""

    @staticmethod
    def _iter_managed_modules(model):
        if model is None:
            return ()
        for module in model.modules():
            if not isinstance(module, TopKLoRALinearSTE):
                continue
            if not bool(getattr(module, "sae_style", False)):
                continue
            if module.decoder_maintenance_target_norm() is None:
                continue
            yield module

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        for module in self._iter_managed_modules(model):
            module.project_decoder_grad_to_tangent_()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        for module in self._iter_managed_modules(model):
            module.post_step_decoder_renorm_()
        return control


class DeadLatentsLoggerCallback(TrainerCallback):
    def __init__(self, log_every=500, activation_threshold=1e-6):
        """
        Args:
            log_every: steps between W&B logs
            activation_threshold: below this value, a latent is considered inactive for that batch
        """
        self.log_every = log_every
        self.activation_threshold = activation_threshold
        self.stats = {}  # {layer_name: {"counts": tensor, "total": int}}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Initialize tracking for each TopKLoRALinearSTE layer
        if model is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                self.stats[name] = {
                    "counts": torch.zeros(module.r, dtype=torch.long),
                    "total": 0,
                }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        # Track activations for each TopK module
        for name, module in model.named_modules():
            if not isinstance(module, TopKLoRALinearSTE):
                continue

            latent_source = getattr(module, "_last_z_sparse", None)
            if latent_source is None:
                latent_source = getattr(module, "_last_g_soft", None)
            if latent_source is None:
                continue

            reduce_dims = tuple(range(latent_source.dim() - 1))
            mean_act = latent_source.abs().mean(dim=reduce_dims)
            active_mask = (mean_act > self.activation_threshold).cpu()
            self.stats[name]["counts"] += active_mask.long()
            self.stats[name]["total"] += 1

        # Periodic logging
        if (
            state.global_step % self.log_every == 0
            and args.report_to
            and "wandb" in args.report_to
        ):
            log_dict = {}
            total_dead_all = 0
            total_latents_all = 0

            for layer_name, st in self.stats.items():
                total_seen = st["total"]
                if total_seen == 0:
                    continue
                dead_mask = st["counts"] == 0
                num_dead = dead_mask.sum().item()
                pct_dead = 100.0 * num_dead / len(dead_mask)

                total_dead_all += num_dead
                total_latents_all += len(dead_mask)

                log_dict[f"dead_latents/{layer_name}/count"] = num_dead
                log_dict[f"dead_latents/{layer_name}/pct"] = pct_dead

            if total_latents_all > 0:
                log_dict["dead_latents/total_count"] = total_dead_all
                log_dict["dead_latents/total_pct"] = (
                    100.0 * total_dead_all / total_latents_all
                )
            wandb.log(log_dict, step=state.global_step)
