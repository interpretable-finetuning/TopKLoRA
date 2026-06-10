"""seq_logprob (the off-by-one / gather) against HF's loss oracle; mu composition."""

import torch

from src.clcd.measure import mu, seq_logprob


def test_seq_logprob_oracle(fix):
    # seq_logprob (sum of true-token log-probs over the completion) must equal
    # -loss*n from HF's own cross-entropy -- proves the shift/gather is correct.
    model, _ = fix
    torch.manual_seed(2)
    P, L = 5, 4
    full = torch.randint(0, 256, (1, P + L))
    labels = full.clone()
    labels[:, :P] = -100
    with torch.no_grad():
        ours = seq_logprob(model, full, P).item()
        loss = model(full, labels=labels).loss.item()
    n = int((labels != -100).sum())
    assert abs(ours - (-loss * n)) < 1e-2


def test_mu_composition(fix):
    # mu(x) = logp(Y+|x) - logp(Y-|x), composed from two teacher-forced passes.
    model, _ = fix
    torch.manual_seed(3)
    prompt = torch.randint(0, 256, (1, 5))
    yp, ym = torch.randint(0, 256, (1, 4)), torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        m = mu(model, prompt, yp, ym).item()
        lp = seq_logprob(model, torch.cat([prompt, yp], 1), 5).item()
        lm = seq_logprob(model, torch.cat([prompt, ym], 1), 5).item()
    assert abs(m - (lp - lm)) < 1e-4
