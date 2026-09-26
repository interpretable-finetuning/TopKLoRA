"""scripts/qwen_fleet_dispatch.sh: reserving a card must actually keep the peer driver off it.

This dispatcher backfills 32B training (~65 GB) onto cards that another running driver
(qwen15_phase1.sh) is draining. That driver places a job on any card with MIN_FREE_MIB (30 GB)
free, so a card holding a 32B trainer still looks placeable to it -- 97 - 65 = 32 GB -- and the
result is two jobs on one card and an OOM that kills both. Campaign 3 lost six searches that way.

The only thing standing between those two facts is that this script takes ALL of a card's slot
locks, in the peer's own namespace and format, before placing anything. So that is what these
tests pin -- with REAL directories, not mocks, because the bug this guards against was a format
mismatch that mocks would have reproduced faithfully and uselessly:

    the peer numbers slots `seq 1 $SLOTS_PER_GPU` -> s1, s2. There is no s0. An earlier version
    of this script claimed s0..s(N-1), which took one lock the peer never reads and left sN free.
    It would have "reserved" a card and OOM'd it anyway.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DISPATCH = REPO / "scripts" / "qwen_fleet_dispatch.sh"
HOST = os.uname().nodename


def run_helper(lockroot, body, slots=2):
    """Source the dispatcher's lock helpers and run `body` against a throwaway lock root.

    Sourcing rather than reimplementing: a copy of claim_card in this file could not have caught
    the s0/s1 bug, because the copy would have had it too.
    """
    script = f"""
set -u
export DISPATCH_LIB_ONLY=1
export PEER_LOCKROOT={lockroot}
export PEER_SLOTS={slots}
source {DISPATCH}
{body}
"""
    return subprocess.run(["bash", "-c", script], cwd=REPO, capture_output=True, text=True, timeout=120)


def test_claim_takes_the_peers_real_slot_names(tmp_path):
    """A claimed card must hold s1..sN -- the names the peer driver actually checks."""
    lr = tmp_path / "locks"
    lr.mkdir()
    r = run_helper(lr, 'claim_card 3 && echo CLAIMED')
    assert "CLAIMED" in r.stdout, f"{r.stdout}\n{r.stderr}"

    got = sorted(p.name for p in lr.iterdir())
    assert got == [f"{HOST}_3_s1", f"{HOST}_3_s2"], f"wrong slot names: {got}"
    for p in lr.iterdir():
        owner = (p / "owner").read_text().split()
        assert owner[0].isdigit(), f"owner file has no pid: {owner}"
        assert "fleet-dispatch-32b" in " ".join(owner)


def test_claim_fails_and_rolls_back_when_the_peer_holds_any_slot(tmp_path):
    """Partial claims are the dangerous case: the peer would use the slot we did not get.

    So a claim that cannot take EVERY slot must take NONE, leaving the peer's lock untouched.
    """
    lr = tmp_path / "locks"
    lr.mkdir()
    peer = lr / f"{HOST}_5_s2"          # peer holds the SECOND slot
    peer.mkdir()
    (peer / "owner").write_text("999999 2026-09-21T00:00:00+00:00 peer-search\n")

    r = run_helper(lr, 'claim_card 5 || echo REFUSED')
    assert "REFUSED" in r.stdout, f"claim should have failed:\n{r.stdout}\n{r.stderr}"

    remaining = sorted(p.name for p in lr.iterdir())
    assert remaining == [f"{HOST}_5_s2"], (
        f"a refused claim must leave exactly the peer's lock, found {remaining}"
    )
    assert "peer-search" in (peer / "owner").read_text(), "we overwrote the peer's owner file"


def test_release_returns_the_card_to_the_peer(tmp_path):
    """After a cell finishes the card must become claimable again, or the fleet drains to zero."""
    lr = tmp_path / "locks"
    lr.mkdir()
    r = run_helper(lr, 'claim_card 6 && release_card 6 && echo DONE')
    assert "DONE" in r.stdout, f"{r.stdout}\n{r.stderr}"
    assert sorted(lr.iterdir()) == [], f"locks left behind: {[p.name for p in lr.iterdir()]}"

    # and it is genuinely re-claimable, not merely empty
    r2 = run_helper(lr, 'claim_card 6 && echo RECLAIMED')
    assert "RECLAIMED" in r2.stdout, f"{r2.stdout}\n{r2.stderr}"


def test_slot_count_follows_the_peers_configuration(tmp_path):
    """PEER_SLOTS must match the peer's SLOTS_PER_GPU; claiming fewer leaves a usable slot open."""
    lr = tmp_path / "locks"
    lr.mkdir()
    run_helper(lr, 'claim_card 2', slots=4)
    got = sorted(p.name for p in lr.iterdir())
    assert got == [f"{HOST}_2_s{i}" for i in (1, 2, 3, 4)], got
