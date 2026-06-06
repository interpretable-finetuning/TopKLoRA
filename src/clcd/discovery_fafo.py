import torch

from src.clcd.latents import read_latents, inject
from src.clcd.fixture import build_random_fixture
from src.clcd.measure import mu, seq_logprob


def test_read_latents():
    model, wrapped_modules = build_random_fixture()
    input_ids = torch.randint(256, (1, 3))

    latents = read_latents(model, input_ids, wrapped_modules)

    for m, a in latents.items():
        g_hard = wrapped_modules[m]._last_g_hard
        last_z = wrapped_modules[m]._last_z
        a_repro = g_hard * last_z
        assert torch.allclose(a, a_repro), f"Mismatch in module {m}"

        print(
            f"Module: {m}, g_hard: {g_hard}, non-zero g_hard per position: {(g_hard > 0).sum(dim=-1)}, non-zero activations per position: {(a > 0).sum(dim=-1)}"
        )


def test_score_latents():
    model, wrapped_modules = build_random_fixture()
    for _ in range(10):
        input_ids_prompt = torch.randint(256, (1, 10))
        input_ids_completion1 = torch.randint(256, (1, 10))
        input_ids_completion2 = torch.randint(256, (1, 10))

        logprob1 = seq_logprob(
            model, torch.cat([input_ids_prompt, input_ids_completion1], dim=1), 10
        )
        logprob2 = seq_logprob(
            model, torch.cat([input_ids_prompt, input_ids_completion2], dim=1), 10
        )

        scores = mu(
            model, input_ids_prompt, input_ids_completion1, input_ids_completion2
        )
        print(f"Logprob 1: {logprob1.item()}, Logprob 2: {logprob2.item()}")
        print(f"Mu score: {scores.item()}")


def test_inject():
    model, wrapped_modules = build_random_fixture()
    input_ids_prompt = torch.randint(256, (1, 10))
    input_ids_completion1 = torch.randint(256, (1, 10))
    input_ids_completion2 = torch.randint(256, (1, 10))

    scores1 = mu(model, input_ids_prompt, input_ids_completion1, input_ids_completion2)

    with inject(
        wrapped_modules, {list(wrapped_modules.keys())[0]: torch.zeros(1, 20, 8)}
    ):
        scores2 = mu(
            model, input_ids_prompt, input_ids_completion1, input_ids_completion2
        )

    print(f"Mu score before injection: {scores1.item()}")
    print(f"Mu score after injection: {scores2.item()}")

    pre = read_latents(model, input_ids_prompt, wrapped_modules)
    with torch.no_grad():
        base_logits = model(input_ids_prompt).logits

    with inject(
        wrapped_modules, {list(wrapped_modules.keys())[4]: torch.zeros(1, 10, 8)}
    ):  # <-- enter the context
        post = read_latents(
            model, input_ids_prompt, wrapped_modules
        )  # forward runs hooked
        with torch.no_grad():
            inj_logits = model(input_ids_prompt).logits

    print("logits changed:", (inj_logits - base_logits).abs().max().item())  # > 0
    print(
        "injected module own latents changed:",
        not torch.equal(
            pre[list(wrapped_modules.keys())[4]], post[list(wrapped_modules.keys())[4]]
        ),
    )  # False
    changed = [m for m in wrapped_modules if not torch.equal(pre[m], post[m])]
    print(f"downstream modules changed: {len(changed)}")  # 8


if __name__ == "__main__":
    # test_read_latents()
    # test_score_latents()
    test_inject()
