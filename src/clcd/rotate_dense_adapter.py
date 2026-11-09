"""Rotation control for dense-LoRA circuit sizes.

A dense LoRA's function depends only on the product B·A: for any invertible R the pair
(R·A, B·R⁻¹) computes the same map, so its latent coordinates are a bookkeeping choice and a
circuit defined as a coordinate subset is basis-relative. TopK-LoRA has no such freedom -- the
ReLU and the hard top-k act coordinate by coordinate -- which is what makes a sparse circuit a
property of the model. This tool writes a function-identical TWIN of a dense adapter whose latent
basis is rotated, per module, by an independent random ORTHOGONAL R (R⁻¹ = Rᵀ, so no
conditioning issue), then verifies through the CLCD loader that the twin's logits match the
original. A circuit search on the twin measures how much of the dense `both_K` is the basis and
how much is the function: stable under rotation => no compact description in a generic basis;
unstable => the dense number is about the coordinates training happened to leave.

    python -m src.clcd.rotate_dense_adapter --adapter <dense adapter dir> --out <twin dir> --seed 1 \\
        --base_model models/qwen15_unaliased_base --data data/sleeper/prepared_eval6k_qwen15

Refuses anything that is not a plain dense adapter (TopK gate, ReLU, k<r, SAE extras, per-latent
tensors): rotation is NOT function-preserving there, and a twin that silently changed the function
would enter the comparison as if it were the same model. The twin directory carries every file of
the source except the weights, plus `rotation.json` (seed, per-module errors, verification).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LORA_A = ".lora_A.weight"
LORA_B = ".lora_B.weight"


def assert_dense_config(cfg: dict) -> None:
    """Raise unless `cfg` (a topk_config.json) describes a plain dense adapter."""
    problems = []
    if cfg.get("use_topk", True):
        problems.append("use_topk must be false")
    if cfg.get("relu_latents", True):
        problems.append("relu_latents must be false")
    r, k = cfg.get("r"), cfg.get("k")
    if r is None or k is None or int(k) != int(r):
        problems.append(f"k ({k}) must equal r ({r})")
    if cfg.get("k_final", k) != k:
        problems.append(f"k_final ({cfg.get('k_final')}) must equal k ({k})")
    if cfg.get("sae_style", False):
        problems.append("sae_style must be false")
    if cfg.get("latent_gate_enabled", False):
        problems.append("latent_gate_enabled must be false")
    if problems:
        raise ValueError(
            "not a plain dense adapter -- rotation would change the function: "
            + "; ".join(problems)
        )


def random_orthogonal(r: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-distributed r x r orthogonal matrix (QR of a Gaussian with the sign fix), float64."""
    g = torch.randn(r, r, generator=gen, dtype=torch.float64)
    q, rr = torch.linalg.qr(g)
    d = torch.sign(torch.diagonal(rr))
    d[d == 0] = 1.0
    return q * d


def module_pairs(state: dict) -> list[tuple[str, str, str]]:
    """(prefix, A key, B key) per module; raise on any tensor that is not a LoRA A/B pair."""
    prefixes_a = {k[: -len(LORA_A)] for k in state if k.endswith(LORA_A)}
    prefixes_b = {k[: -len(LORA_B)] for k in state if k.endswith(LORA_B)}
    if prefixes_a != prefixes_b:
        raise ValueError(f"unpaired LoRA tensors: {sorted(prefixes_a ^ prefixes_b)[:5]}")
    other = [k for k in state if not (k.endswith(LORA_A) or k.endswith(LORA_B))]
    if other:
        raise ValueError(
            "adapter carries per-latent tensors beyond lora_A/lora_B (a TopK/SAE organism?); "
            f"rotation is not defined for them: {other[:5]}"
        )
    return [(p, p + LORA_A, p + LORA_B) for p in sorted(prefixes_a)]


def rotate_state(state: dict, seed: int, *, orthogonal: bool = True) -> tuple[dict, dict]:
    """Return (rotated state, per-module info). A' = R·A, B' = B·Rᵀ per module, R drawn from a
    per-module generator seeded by (seed, module index) so the twin is reproducible from the seed.
    `orthogonal=False` draws a Gaussian R instead -- NOT function-preserving -- and exists only so
    the preservation check can be shown to fail."""
    new = {}
    info = {}
    for i, (prefix, ka, kb) in enumerate(module_pairs(state)):
        A, B = state[ka], state[kb]
        r = int(A.shape[0])
        if int(B.shape[1]) != r:
            raise ValueError(f"{prefix}: A is {tuple(A.shape)} but B is {tuple(B.shape)}")
        gen = torch.Generator().manual_seed(int(seed) * 100_003 + i)
        R = (random_orthogonal(r, gen) if orthogonal
             else torch.randn(r, r, generator=gen, dtype=torch.float64))
        A64, B64 = A.double(), B.double()
        A2, B2 = R @ A64, B64 @ R.T
        new[ka], new[kb] = A2.to(A.dtype), B2.to(B.dtype)
        eye = torch.eye(r, dtype=torch.float64)
        info[prefix] = {
            "r": r,
            "orth_err": float((R.T @ R - eye).abs().max()),
            "A_rel_change": float((new[ka].double() - A64).norm() / A64.norm()),
        }
    return new, info


def product_rel_err(state_a: dict, state_b: dict) -> float:
    """max over modules of max|B_a·A_a − B_b·A_b| / max|B_b·A_b|, in float64.
    This is the function-preservation check at the weight level: zero up to rounding for an
    orthogonal rotation, order one for anything else."""
    worst = 0.0
    for prefix, ka, kb in module_pairs(state_b):
        pb = state_b[kb].double() @ state_b[ka].double()
        pa = state_a[kb].double() @ state_a[ka].double()
        worst = max(worst, float((pa - pb).abs().max() / pb.abs().max().clamp_min(1e-30)))
    return worst


def write_rotated_adapter(src, dst, seed: int, *, orthogonal: bool = True) -> dict:
    """Write the twin of the dense adapter at `src` to `dst` (must not exist). Returns the
    rotation record, also written to `dst/rotation.json`."""
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "topk_config.json").read_text())
    assert_dense_config(cfg)  # refuse BEFORE creating anything
    if dst.exists():
        raise FileExistsError(f"{dst} exists -- refusing to overwrite a twin")
    state = load_file(str(src / "adapter_model.safetensors"))
    new, info = rotate_state(state, seed, orthogonal=orthogonal)
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy2(f, dst / f.name)
    save_file(new, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
    stored = load_file(str(dst / "adapter_model.safetensors"))
    record = {
        "source": str(src),
        "seed": int(seed),
        "orthogonal": bool(orthogonal),
        "n_modules": len(info),
        "product_rel_err_stored": product_rel_err(stored, state),
        "modules": info,
        "verification": None,
    }
    (dst / "rotation.json").write_text(json.dumps(record, indent=2))
    return record


def verify_function_preserved(src, dst, *, base_model, data, n, offset, device="cuda",
                              dtype=torch.float32, batch_size=8) -> dict:
    """Load both adapters through the CLCD loader (the wrap the pipeline measures with) and
    compare full-sequence logits on `n` real trigger prompts."""
    from src.clcd.organism import load_organism
    from src.data import load_jsonl_rows, load_tags, render_prompt

    trig, _ = load_tags(data)
    qs = load_jsonl_rows(data, "eval_triggered", offset, n)
    m_src, tok, w_src = load_organism(str(src), base_model=base_model, device=device, dtype=dtype)
    m_dst, _, w_dst = load_organism(str(dst), base_model=base_model, device=device, dtype=dtype)
    if len(w_src) != len(w_dst):
        raise RuntimeError(f"wrapped modules differ: {len(w_src)} vs {len(w_dst)}")
    m_src.eval()
    m_dst.eval()
    tok.padding_side = "left"
    prompts = [render_prompt(tok, question=q, tag=trig) for q in qs]
    max_diff, agree, total = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
            la, lb = m_src(**enc).logits.float(), m_dst(**enc).logits.float()
            max_diff = max(max_diff, float((la - lb).abs().max()))
            mask = enc["attention_mask"].bool()
            agree += int(((la.argmax(-1) == lb.argmax(-1)) & mask).sum())
            total += int(mask.sum())
    return {"n_prompts": len(prompts), "dtype": str(dtype), "max_abs_logit_diff": max_diff,
            "argmax_agreement": agree / max(total, 1), "n_positions": total,
            "n_wrapped_modules": len(w_src)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, help="source dense adapter dir (the leaf with adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="twin dir to create (must not exist)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base_model", default=None, help="required unless --no_verify")
    ap.add_argument("--data", default="data/sleeper/prepared_eval6k_qwen15")
    ap.add_argument("--n_verify", type=int, default=64)
    ap.add_argument("--offset", type=int, default=100, help="verify on the selection band, like Gate A")
    ap.add_argument("--tol", type=float, default=1e-2, help="max |dlogit| (fp32) above which the twin is REJECTED")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_verify", action="store_true", help="write only (CPU, tests); the twin is then UNVERIFIED")
    a = ap.parse_args()

    rec = write_rotated_adapter(a.adapter, a.out, a.seed)
    print(f"[rotate] {rec['n_modules']} modules rotated (seed {a.seed}); stored product rel err "
          f"{rec['product_rel_err_stored']:.3e}; max orth err "
          f"{max(m['orth_err'] for m in rec['modules'].values()):.1e}; "
          f"min A rel change {min(m['A_rel_change'] for m in rec['modules'].values()):.3f}")
    if a.no_verify:
        print("[verify] SKIPPED (--no_verify) -- twin is unverified")
        return 0
    if not a.base_model:
        sys.exit("--base_model is required for verification")
    v = verify_function_preserved(a.adapter, a.out, base_model=a.base_model, data=a.data,
                                  n=a.n_verify, offset=a.offset, device=a.device)
    rec["verification"] = {**v, "tol": a.tol, "pass": v["max_abs_logit_diff"] <= a.tol}
    (Path(a.out) / "rotation.json").write_text(json.dumps(rec, indent=2))
    print(f"[verify] n={v['n_prompts']} prompts, {v['n_positions']} positions, fp32: max |dlogit| = "
          f"{v['max_abs_logit_diff']:.3e} (tol {a.tol:g}), argmax agreement {v['argmax_agreement']:.6f}")
    if not rec["verification"]["pass"]:
        print("VERDICT: FAIL -- twin does NOT reproduce the source function; do not search it")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
