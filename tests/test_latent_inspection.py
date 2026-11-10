"""The pre-registered draw must be reproducible from the committed seed alone, and must never
hand out a circuit slot to a latent that is not in the circuit."""
import pytest

from src.autointerp.latent_inspection import draw_latents

POOL = [(f"m{j}", i) for j in range(5) for i in range(20)]
CIRCUIT = [("m1", 3), ("m2", 7), ("m4", 0), ("m0", 19)] + [("m3", i) for i in range(12)]


def test_draw_is_pinned_by_seed():
    # Pinned values: if the RNG consumption order changes, a published draw silently moves.
    d = draw_latents(POOL, CIRCUIT, seed=20261110, n_blind_each=3)
    assert d == draw_latents(list(reversed(POOL)), list(reversed(CIRCUIT)), seed=20261110, n_blind_each=3)
    assert d["inspect"] == [("m1", 18), ("m4", 2), ("m3", 2)]
    assert d["blind_pool"] == [("m3", 14), ("m4", 14), ("m4", 15)]
    assert d["blind_circuit"] == [("m3", 5), ("m3", 9), ("m3", 8)]


def test_circuit_slots_come_from_circuit_and_nothing_repeats():
    d = draw_latents(POOL, CIRCUIT, seed=1, n_blind_each=5)
    assert d["inspect"][2] in CIRCUIT and all(x in CIRCUIT for x in d["blind_circuit"])
    flat = d["inspect"] + d["blind_pool"] + d["blind_circuit"]
    assert len(flat) == len(set(flat))


def test_circuit_latent_outside_pool_is_an_error():
    with pytest.raises(ValueError):
        draw_latents(POOL, CIRCUIT + [("zz", 0)], seed=1)
