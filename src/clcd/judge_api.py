"""LLM-as-judge over the OpenRouter BATCH API, as a second instrument beside the local 32B judge.

DEFAULT MODEL: `openai/gpt-5.6-luna` (see DEFAULT_MODEL). The measurements below were taken on the
first model tried, `google/gemini-3.8-flash`, and are kept because they are why the module is
shaped as it is. What differs for luna (2026-09-16): reasoning is optional (default effort medium);
batch wall time GROWS with size (20 items 11 min, 4,190 items 12 min, 10,000 items 77 min); and of
two 20-item batches submitted together while a 10k batch ran, one returned in 11 min and one took
far longer, so same-model concurrency is neither clearly parallel nor strictly queued.

MAKE EACH BATCH LARGE, AND SUBMIT EVERY CHUNK BEFORE WAITING ON ANY OF THEM. Sequential
submit-then-wait cost one full turnaround per chunk: on 2026-09-16 the 19-file sparse re-judge sat
3 h 24 min on chunk0 before chunk1 was created. A submitted batch cannot be cancelled, so the fleet
is committed the moment the loop runs. The gemini-era measurements that shaped this module:
  * Batch wall time is ~FLAT in size: 2 requests 654 s, 120 requests 656 s. Size is nearly free.
  * CONCURRENT batches do NOT run in parallel -- they queue. Five 500-item batches submitted at
    once sat at 0/500 for **108 minutes**, while the same work submitted sequentially had been
    clearing in ~20 min each. Parallelising made it strictly slower.
  * **An in-progress batch CANNOT be cancelled.** `DELETE /batches/{id}` returns 409 ("Only
    completed, failed, expired, or cancelled batches can be deleted") and there is no
    `/cancel` endpoint (404). A mistaken submission therefore blocks the queue behind it for up
    to the 24 h completion window and cannot be taken back.
  So: one job, as big as the work allows. `judge_saved_gens_big` batches a whole FILE (all
  conditions and both prompt sets) into a single submission for this reason.

WHY BATCH, measured not assumed (2026-09-16 preflight, live API):
  * Synchronous is rate-capped: `new-account-rpm/google/gemini-3.8-flash`, **20 requests/minute**
    (`X-RateLimit-Limit: 20`, `limit_source: openrouter_new_account`). That is 14 h for the 17k
    control and 64 h for the 76.6k corpus. The published docs do not mention this cap; only the
    live 429 does.
  * Batch is not subject to it: a 120-request job (150 KB inline) was accepted in 2.3 s.

WHY NO `response_format`, also measured:
  * `json_schema` with `strict: true` sends this model into runaway reasoning -- 762 / 4,274 /
    5,392 reasoning tokens at effort low / medium / high -- and then returns an EMPTY object
    (`{}`). ~200x the tokens for zero usable scores.
  * Plain prompting returns a bare `5` in ~22 completion tokens.
  * It also keeps the prompt BYTE-IDENTICAL to the local judge (`_judge_user_prompt`), which is
    what makes the two instruments comparable at all. A judge that sees a different prompt is a
    different measurement, and the whole point of this module is a like-for-like second opinion.

WHY `max_tokens` IS LARGE:
  * Reasoning is mandatory on this model (`reasoning.mandatory: true` in the model catalogue) and
    reasoning tokens count against `max_tokens`. At the repo's existing default of 16 the model
    returns `content=None` with `finish_reason='length'` -- every score a parse failure.

SCORES NEVER OVERWRITE THE LOCAL JUDGE. They are written under their own key (see
`judge_key_for`), because assembling a retention ratio from an API numerator over a 32B base floor
is the single most likely silent error in this migration and nothing downstream would catch it.

THE RESUMABLE STATE IS KEYED BY THE WORK, NOT BY THE RUN (v2, 2026-09-17). v1 keyed the state file
-- and every per-item id inside it -- by a `scope` string built from the files that still needed
judging. That set is exactly what a partial failure changes, so:
  * a crash after writing SOME files gave the next run a shorter todo list, a different scope, no
    state file, and a full resubmission of work already complete on the server (reproduced: crash
    after file 1 of 90 resubmits 89 files, ~$17 and ~16.7 h of queue time for results already paid
    for and sitting in a state file nothing will ever read again);
  * a new surgical file appearing between two runs -- which is what an incremental judging worker
    does by design -- invalidated the scope the same way;
  * `--chunk_size` changed the item->chunk mapping but not the key, so a relaunch at a different
    chunk size adopted ids that did not cover the items it thought they did;
  * the scope named NO MODEL, so judging one file set with a second model found the first model's
    state, served its scores under the second model's key and self-certified them in `judge_meta`
    -- a forged second instrument, i.e. the one failure this module exists to prevent (above);
  * a generation regenerated while its batch was in flight kept the same scope, so the stale
    scores were adopted and written against the new text.
v2 keys every item by a hash of the instrument (model, provider pin, token budget, prompt template)
and the exact (question, response) text, and records for each submitted batch the item keys it
carries. Adoption is then "which in-flight batches hold items I still need", which is invariant
under which files happen to be outstanding, how chunks were sized and which model ran first -- and
which CHANGES when a generation changes, so a stale score can no longer attach to new text.
v1 state cannot be migrated: the request text was never stored, so a v1 id cannot be matched to an
item. v1 files are detected and reported (`legacy_state_files`) rather than read; an uncollected
v1 batch is money this module can no longer collect, and it says so instead of resubmitting in
silence.

A FAILED ITEM IS NOT AN ANSWER (2026-09-17). Storing a per-item `{"error": ...}` record in
`state["items"]` made the failure permanent: 3 of 54 items came back `upstream 500`, the pass
raised on the failure ceiling, and the two relaunches after it reported "0 to do", raised again
and submitted nothing -- only hand-editing the state file could unwedge it, and the message never
named that file. (An item MISSING from a completed batch was retried correctly the whole time;
the asymmetry was the bug.) So an item counts as done only when it was ANSWERED, and a relaunch
re-asks the ones that failed, bounded by `max_item_attempts` and then refused by name. An
unparseable ANSWER is not a failure of this kind: temperature is 0, so re-asking buys the same
bytes, and the ceiling stops the file exactly as before rather than re-billing 2,838 requests.

AN ORPHANED BATCH IS PAID-FOR WORK NOTHING CAN REACH (2026-09-17). Between `submit` returning an
id and `_save_state` persisting it there was a window -- 3 of 13 kill points reproduced it, 74
submissions for 54 items -- where a batch existed on the server and in no state file, so no run
could ever adopt it. The fix is a two-phase write: an INTENT record (the item keys, the instrument,
this run's scope) is persisted BEFORE the create call, so a batch created by a run that died is
looked for on the next one in the account's batch listing (the list endpoint showed a batch
throughout the window in which GET still 404'd, see NOT_FOUND_GRACE_S).

HOW THE ORPHAN IS RECOGNISED, after a live probe (2026-09-17, 2 requests, real account) falsified
the first design. The intent id is sent in the create body's `metadata` and the call is ACCEPTED --
but the listing NEVER echoes it: a listed batch carries exactly `completion_window, created_at,
endpoint, error, finalized_at, id, model, object, request_counts, results, status, usage` and no
`metadata` at all. So the by-name path could never fire in production, every orphan degraded to
"unresolved", and the work was bought again -- the window was visible, not closed. It is closed
here by the fields the API does return: an intent adopts a listed batch when EXACTLY ONE batch is
unaccounted for by any state file, of this model's family (the listing answers with a pinned
snapshot, `...-20260709`, not the `...:batch` string the create call sent), carries exactly as many
requests as the intent has items, and was created no earlier than the intent was written. A unique
fit on a COMPLETE listing is an adoption; two fits, no fit with something still unaccounted for, a
truncated listing or a listing that failed are all "cannot tell" and are reported loudly by intent
id -- because "no match" and "cannot tell" must not look alike when the difference is a
resubmission. An intent with nothing unaccounted for on a COMPLETE listing never became a batch and
is dropped. `metadata` is still sent: it is free, and it would make this exact rather than
circumstantial if the API ever starts echoing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from src.data import write_json_atomic
from src.evaluate import JUDGE_SYSTEM_PROMPT, _judge_user_prompt

BASE_URL = "https://openrouter.ai/api/beta"
# The judge for every Qwen run from 2026-09-16 on (user instruction). Chosen over luna-pro (no
# agreement gain, 7x cost) and over gemini-3.8-flash (6-7 h batches), and validated at full n on the
# 5 control organisms: retention within 4.2 pp of the 32B judge, 0 unparsed of 14,190. Scores from
# other models stay under their own keys (judge_key_for), so switching never overwrites them.
DEFAULT_MODEL = "openai/gpt-5.6-luna"
# A bare integer 1-5 and nothing else. Deliberately NOT the repo's `_extract_score_1_to_5`, which
# scrapes the first digit out of prose: verified live, it scores
# "The instruction asks for 3 examples; quality is poor." as **3**. Here anything that is not
# exactly a digit is a recorded failure, never a guess.
_STRICT_SCORE = re.compile(r"\A([1-5])\Z")
# Reasoning is mandatory and billed against this budget; 16 returns nothing at all.
DEFAULT_MAX_TOKENS = 2048
TERMINAL = {"completed", "failed", "expired", "cancelled"}
# The `metadata` field a submitted batch carries so that a batch created by a run that died before
# persisting its id could be recognised on the account by name. ACCEPTED BY THE CREATE CALL AND NOT
# ECHOED BY THE LISTING (live probe 2026-09-17), so nothing reads it today; it is still sent
# because it costs nothing and turns `_reconcile_intents` from circumstantial into exact the day
# the API starts returning it.
INTENT_METADATA_KEY = "clcd_intent"
# The listing reports `created_at` in whole unix seconds from the SERVER's clock; an intent records
# `time.time()` from THIS host microseconds before the create call. The two differ by the hosts'
# clock offset plus up to a second of truncation, so a batch may legitimately be stamped slightly
# before the intent that created it. 120 s is generous for NTP-synced hosts and still far below the
# distance to any other batch of a campaign: two submissions of the same size are separated by at
# least a batch turnaround (11 min at the smallest size measured), and anything older than the
# intent by more than this is some earlier run's work, not this intent's.
INTENT_CLOCK_SKEW_S = 120
# A listed batch names a pinned snapshot of the model the create call asked for: the request said
# `openai/gpt-5.6-luna:batch` and the listing answered `openai/gpt-5.6-luna-20260709` (live
# 2026-09-17). Matching the two exactly therefore never fires, and matching by prefix would make
# `...-luna` match `...-luna-pro`; so both sides are reduced to the family -- the `:batch` suffix
# and a trailing date stamp removed, nothing else.
_MODEL_SNAPSHOT_SUFFIX = re.compile(r"-\d{6,}\Z")


def _model_family(model: Any) -> str:
    return _MODEL_SNAPSHOT_SUFFIX.sub("", str(model or "").split(":")[0])


def parse_score(raw: Optional[str]) -> Optional[int]:
    """Strict: a bare 1-5 or nothing. Never scrapes a digit out of prose."""
    if raw is None:
        return None
    m = _STRICT_SCORE.match(raw.strip())
    return int(m.group(1)) if m else None


def prompt_fingerprint() -> str:
    """sha256 of the exact system+user template. Recorded with every result so a later reader can
    tell whether two runs used the same instrument."""
    probe = _judge_user_prompt("\x00Q\x00", "\x00A\x00")
    return hashlib.sha256((JUDGE_SYSTEM_PROMPT + "\x1f" + probe).encode()).hexdigest()[:16]


def model_slug(model: str) -> str:
    """`google/gemini-3.8-flash` -> `gemini_3_8_flash`, for use in an artifact key."""
    tail = model.split("/")[-1].split(":")[0]
    return re.sub(r"[^0-9a-zA-Z]+", "_", tail).strip("_").lower()


def judge_key_for(model: str, independent: bool = False) -> str:
    """The artifact key these scores are written under. Never `judge_32b`."""
    return f"judge_{'indep_' if independent else ''}api_{model_slug(model)}"


def build_request(question: str, response: str, cid: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "custom_id": cid,
        "body": {
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_prompt(question, response)},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    }


@dataclass
class JudgeConfig:
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    # Pin the serving provider. OpenRouter's default routing load-balances across providers AND
    # quantizations; for a measurement instrument a silent backend change mid-run is disqualifying.
    provider_only: Optional[Sequence[str]] = ("openai",)
    allow_fallbacks: bool = False
    # Requests per submitted job. The documented pages give no maximum, but the API enforces one:
    # 14,190 was refused on 2026-09-16 with `413 Batch of more than 10,000 requests is not allowed`
    # (before anything was created, so nothing was spent). Keep this <= 10,000. A 10,000-item luna
    # batch returned in 77 min.
    chunk_size: int = 10000
    # Submit every chunk before waiting on any of them (see the loop in api_judge_scores). False
    # restores the strictly sequential submit-then-wait behaviour.
    parallel_chunks: bool = True
    # The account's ceiling on in-flight batch REQUESTS (not batches), enforced live on
    # 2026-09-16: `429 This entity has 20,000 in-flight batch requests and this batch adds 10,000,
    # exceeding the 20,000 limit`. Undocumented. Submitting past it kills the run, so the loop
    # waits for an earlier chunk to land instead.
    max_in_flight_requests: int = 20000
    poll_seconds: int = 30
    # Pre-registered ceiling. Above this the pass RAISES rather than returning a mean computed
    # from whatever happened to parse -- refusals are not random, they concentrate on degenerate
    # base-model output, which biases the floor upward, i.e. toward manufacturing a result.
    max_failure_rate: float = 0.005
    # How many times ONE item may be asked before the run refuses to ask again. A per-item error
    # is retried (see the docstring), and an endpoint that will never answer must not be allowed
    # to bill an unattended worker in a loop; 3 leaves room for a transient 500 without that.
    max_item_attempts: int = 3
    state_dir: Path = field(default_factory=lambda: Path("clcd_results/judge_api_state"))

    def batch_model(self) -> str:
        return self.model if self.model.endswith(":batch") else f"{self.model}:batch"


def instrument_id(cfg: JudgeConfig) -> str:
    """Everything that makes two judging requests the SAME measurement: the model, the provider
    pin (without one OpenRouter load-balances across providers AND quantizations), the token
    budget (at 16 this model returns content=None, i.e. a different answer) and the prompt
    template. Two configurations differing in any of these are two instruments, and a score from
    one must never be served for the other -- which is precisely what v1 state did across models.
    """
    return "\x1f".join([
        cfg.model,
        ",".join(cfg.provider_only) if cfg.provider_only else "-",
        "fallbacks" if cfg.allow_fallbacks else "no-fallbacks",
        str(cfg.max_tokens),
        prompt_fingerprint(),
    ])


def item_keys(questions: Sequence[str], responses: Sequence[str], cfg: JudgeConfig) -> List[str]:
    """One custom_id per (question, response), derived from the TEXT and the instrument -- never
    from the run. Pairing is by id, NOT by list position: positional zip is how a truncated
    generation run silently mispairs questions with answers, and a scope-and-index id (v1) is how
    a run over a different file set silently attached its scores to this one's text.

    Identical (q, r) pairs -- degenerate base-model output repeats itself -- get an ordinal, so
    every occurrence is still its own request exactly as before. The ordinal is the only part of
    the key that depends on list order: if the surrounding list changes, a duplicate can take a
    different ordinal and be requested again. That costs one repeat request for text byte-identical
    to text already judged; it can never attach a score to different text.
    """
    inst = instrument_id(cfg)
    seen: Dict[str, int] = {}
    out: List[str] = []
    for q, r in zip(questions, responses):
        h = hashlib.sha256("\x1f".join((inst, q, r)).encode()).hexdigest()[:24]
        n = seen.get(h, 0)
        seen[h] = n + 1
        out.append(f"{h}-{n:04d}")
    return out


def _api_key_from_env() -> str:
    """Env first, then `.env` -- the repo keeps credentials in `.env` (python-dotenv is a
    dependency and `.env-template` documented the names), and nothing on the judging path loaded
    it, so the key guard fired on a key that was present on disk the whole time. Never logged."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv
    except ImportError:
        return ""
    load_dotenv(".env")
    return os.environ.get("OPENROUTER_API_KEY") or ""


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def submit(requests_payload: List[Dict[str, Any]], cfg: JudgeConfig, api_key: str,
           *, intent_id: Optional[str] = None) -> str:
    """Create one batch job; returns its id. Field order is load-bearing: the API rejects a body
    whose `requests` key precedes `endpoint`/`model`.

    `intent_id` is written into the job's `metadata`. The field is ACCEPTED (200, batch created)
    and NOT ECHOED by the account's batch listing as of the live probe on 2026-09-17, so nothing
    reads it back today and `_reconcile_intents` recognises an orphan by its shape instead. It is
    still sent because it costs nothing, and the day the listing returns it the match becomes exact
    rather than circumstantial. (Were the API to start REJECTING it, the first create call fails
    before anything is created, i.e. loudly and for free.)
    """
    body: Dict[str, Any] = {
        "endpoint": "/v1/chat/completions",
        "model": cfg.batch_model(),
    }
    if intent_id:
        body["metadata"] = {INTENT_METADATA_KEY: intent_id}
    body["requests"] = requests_payload
    if cfg.provider_only:
        for r in body["requests"]:
            r["body"]["provider"] = {
                "only": list(cfg.provider_only),
                "allow_fallbacks": cfg.allow_fallbacks,
            }
    r = requests.post(
        f"{BASE_URL}/batches", headers=_headers(api_key), data=json.dumps(body), timeout=300
    )
    if r.status_code >= 300:
        raise RuntimeError(f"batch submit failed {r.status_code}: {r.text[:500]}")
    return r.json()["id"]


# A just-created batch is NOT immediately queryable: observed live 2026-09-16, a batch that the
# create call had already returned an id for answered GET with
# `404 {"message": "Batch job ... not found."}` and only became visible later (it was `in_progress`
# with all 40 requests on the next look, and the account-level list showed it throughout). Raising
# on that first 404 killed the whole run after the work had already been PAID FOR. So a 404 inside
# this window means "not visible yet", and only a persistent one is fatal.
NOT_FOUND_GRACE_S = 600
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def fetch(batch_id: str, api_key: str, *, grace_s: float = NOT_FOUND_GRACE_S,
          sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """GET a batch, tolerating the post-create invisibility window and transient server errors.
    A 404 that persists beyond `grace_s` is still an error -- the window must not become an
    excuse to spin forever on a batch that genuinely does not exist."""
    deadline = time.monotonic() + grace_s
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(f"{BASE_URL}/batches/{batch_id}", headers=_headers(api_key), timeout=120)
        except requests.RequestException as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"batch fetch {batch_id}: network error after {grace_s}s: {exc}")
            sleep(min(30.0, 2.0 * attempt)); continue
        if r.status_code < 300:
            return r.json()
        transient = r.status_code == 404 or r.status_code in _RETRY_STATUS
        if transient and time.monotonic() < deadline:
            sleep(min(30.0, 2.0 * attempt)); continue
        raise RuntimeError(f"batch fetch failed {r.status_code}: {r.text[:500]}")


# The largest listing the API will serve: `limit=200` is refused with `400 Invalid batch list
# query` (live 2026-09-17), and 100 was served. It is a hard ceiling, not a default to raise:
# an account holding more than this many batches can only ever return a TRUNCATED listing here,
# which resolves nothing (see `_reconcile_intents`) -- so at campaign scale an orphan may become
# unadoptable again, loudly, rather than silently.
LIST_LIMIT_MAX = 100


def list_batches(api_key: str, *, limit: int = LIST_LIMIT_MAX) -> tuple:
    """Every batch on the account, in the API's own order, as `(batches, complete)`.

    THE ORDER IS NOT NEWEST-FIRST and nothing here may assume it is: probed live on 2026-09-17, a
    batch created seconds earlier was ABSENT from a `limit=20` listing (which reported itself
    incomplete) and present among 42 at `limit=100`. So a small limit is not "the recent ones", it
    is an arbitrary subset, and only `limit=LIST_LIMIT_MAX` is used.

    This is how an orphan is found: during the post-create window in which GET still answered 404
    the account-level list showed the batch throughout (2026-09-16), so the list is the reliable
    view. `complete` is False when the API says there are more pages than this call read -- a
    truncated listing must never be read as "the batch does not exist", which is the one
    conclusion that would make a run pay for it twice.
    """
    r = requests.get(f"{BASE_URL}/batches", headers=_headers(api_key),
                     params={"limit": limit}, timeout=120)
    if r.status_code >= 300:
        raise RuntimeError(f"batch list failed {r.status_code}: {r.text[:500]}")
    payload = r.json()
    if isinstance(payload, list):
        return payload, True
    data = payload.get("data")
    return list(data or []), not payload.get("has_more")


def extract_results(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """custom_id -> {raw, usage, error}. A per-item failure is reported as its own record, not
    dropped: a batch can be `completed` while individual items failed."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("results") or []:
        cid = item.get("custom_id")
        if cid is None:
            continue
        err = item.get("error")
        body = (item.get("response") or {}).get("body") or {}
        choices = body.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content")
        out[cid] = {"raw": content, "usage": body.get("usage"), "error": err}
    return out


# Bumped when the on-disk layout changes. v1 (scope-keyed) is unreadable here on purpose: see the
# module docstring and `legacy_state_files`.
STATE_VERSION = 2


def _state_path(cfg: JudgeConfig) -> Path:
    """One state file per MODEL -- not per run and not per scope. The v1 filename was a hash of
    the current todo list, so a state file was orphaned by the very failure it existed to survive.
    Two models never share a file (and could not share an item key even if they did)."""
    return Path(cfg.state_dir) / f"{model_slug(cfg.model)}.json"


def _load_state(cfg: JudgeConfig) -> Dict[str, Any]:
    p = _state_path(cfg)
    if not p.exists():
        return {"version": STATE_VERSION, "model": cfg.model, "batches": {}, "items": {},
                "intents": {}}
    state = json.loads(p.read_text())
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"{p}: state format version {state.get('version')!r}, expected {STATE_VERSION}. "
            "Refusing to read it -- guessing at another layout is how a batch id is lost or a "
            "score is attached to the wrong text. Move the file aside to start a fresh state."
        )
    # `intents` post-dates the first v2 files; it is additive, so a v2 file without it is read,
    # not refused -- a version bump here would orphan the in-flight batches those files hold.
    state.setdefault("intents", {})
    return state


def _save_state(cfg: JudgeConfig, state: Dict[str, Any]) -> None:
    write_json_atomic(_state_path(cfg), state, indent=1, allow_nan=False)


def legacy_state_files(cfg: JudgeConfig) -> List[Dict[str, Any]]:
    """v1 state files sitting in `state_dir`, with the batch ids they hold.

    v1 keyed the file by a hash of the submitting run's todo list and every item id by that hash
    plus a list INDEX, and it never stored the question/response text. A v1 result therefore
    cannot be matched to an item here, and a v1 batch id cannot be re-keyed: the mapping lived in
    the item list of a run that is gone. Nothing here can adopt one -- which is exactly why they
    are reported rather than ignored, because an uncollected v1 batch is work already PAID FOR
    whose items this module is about to request again.
    """
    d = Path(cfg.state_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        # A state file that will not parse is an error, not something to step over: it may be the
        # only record of an in-flight batch.
        s = json.loads(p.read_text())
        if s.get("version") == STATE_VERSION:
            continue
        out.append({"path": str(p), "scope": s.get("scope"),
                    "batches": sorted((s.get("batches") or {}).values()),
                    "n_items": len(s.get("items") or {})})
    return out


def _report_legacy_state(cfg: JudgeConfig, log: Callable[[str], None]) -> int:
    """Loud, every run, until the files are gone. Returns the number of unadoptable batch ids."""
    legacy = legacy_state_files(cfg)
    if not legacy:
        return 0
    n = sum(len(e["batches"]) for e in legacy)
    log(f"[judge-api] WARNING: {len(legacy)} legacy (v1) state file(s) in {cfg.state_dir} holding "
        f"{n} batch id(s). NONE of them can be adopted: a v1 id was derived from the submitting "
        f"run's file list and the request text was never stored, so its results cannot be matched "
        f"to any item here. Any of those batches still in flight is paid-for work that is being "
        f"requested again by this run.")
    for e in legacy:
        log(f"[judge-api]   {e['path']}: {len(e['batches'])} batch(es), {e['n_items']} stored "
            f"result(s) -- {', '.join(e['batches']) or 'none'}")
    log("[judge-api] Collect or write off those batches, then delete the files: nothing reads them.")
    return n


def _answered(rec: Optional[Dict[str, Any]]) -> bool:
    """True when this item has an ANSWER on file -- even an unparseable one.

    A record carrying an `error` is a request that did not produce an answer, and treating it as
    one is how three `upstream 500`s wedged every later pass (see the docstring). An unparseable
    answer IS an answer: the model replied, temperature is 0, and re-asking would buy the same
    bytes at full price. The two are separated here and nowhere else."""
    return rec is not None and not rec.get("error")


def _attempts(rec: Optional[Dict[str, Any]]) -> int:
    return int((rec or {}).get("attempts") or 0)


def _merge_result(items: Dict[str, Any], cid: str, result: Dict[str, Any]) -> None:
    """Store one per-item result and count the attempt it cost. The count is what bounds the
    retry of a failed item, so it must accumulate ACROSS runs, i.e. live in the state file."""
    rec = dict(result)
    rec["attempts"] = _attempts(items.get(cid)) + 1
    items[cid] = rec


def _record_intent(cfg: JudgeConfig, state: Dict[str, Any], keys: List[str], scope: str) -> str:
    """Persist what this process is ABOUT to buy, before the create call can return an id it might
    never get to write down. The id is a nonce: two runs submitting the same items must not share
    an intent, or reconciling one would silently account for the other."""
    iid = hashlib.sha256(
        "\x1f".join([instrument_id(cfg), scope, str(time.time()), os.urandom(8).hex(), *keys])
        .encode()
    ).hexdigest()[:24]
    state["intents"][iid] = {"item_keys": list(keys), "instrument": instrument_id(cfg),
                             "scope": scope, "created_at": time.time(), "batch_id": None}
    _save_state(cfg, state)
    return iid


def _known_batch_ids(cfg: JudgeConfig, state: Dict[str, Any]) -> set:
    """Every batch id any state file in `state_dir` already names, plus this run's in-memory state.

    A batch some file accounts for is not an orphan, and must never be a candidate for adoption by
    shape -- v1 files in particular can hold an in-flight id of THIS model (that is exactly what
    `legacy_state_files` reports), and that id is the one thing about them that is still true. v2
    keys `batches` by batch id, v1 by chunk label, so both sides of the mapping are collected. A
    state file that will not parse raises here for the same reason it raises in
    `legacy_state_files`: it may be the only record of an in-flight batch.
    """
    ids = {str(b) for b in (state.get("batches") or {})}
    d = Path(cfg.state_dir)
    if not d.is_dir():
        return ids
    for p in sorted(d.glob("*.json")):
        batches = (json.loads(p.read_text()).get("batches") or {})
        ids.update(str(k) for k in batches)
        ids.update(str(v) for v in batches.values() if isinstance(v, str))
    return ids


def _reconcile_intents(cfg: JudgeConfig, state: Dict[str, Any], api_key: str,
                       log: Callable[[str], None]) -> int:
    """Adopt the batches that earlier runs created but never got to record. Returns the number of
    intents this run could not resolve either way.

    Only runs when an intent is pending, so a clean state costs no extra request. An intent is
    resolved in exactly three ways:
      * a listed batch carries its id in `metadata` -- exact, and dead in production: the API
        accepts the field on create and never echoes it back (live 2026-09-17, see `submit`);
      * EXACTLY ONE listed batch FITS it (`fits` below) on a COMPLETE listing. This is what is
        left once `metadata` is gone, and every part of it is load-bearing: a batch that two
        intents fit identifies neither, and a truncated listing cannot say that a fit is unique;
      * a COMPLETE listing holds nothing unaccounted for at all -- it never became a batch, drop
        it.
    Everything else -- a listing that failed, a truncated one, several batches that fit, or one
    that nothing accounts for and nothing fits -- is ambiguous, and an ambiguous intent is reported
    by name instead of being resolved by assumption.
    """
    pending = dict(state.get("intents") or {})
    if not pending:
        return 0
    listing: List[Dict[str, Any]] = []
    complete = False
    try:
        listing, complete = list_batches(api_key)
    except (RuntimeError, requests.RequestException) as exc:
        log(f"[judge-api] WARNING: cannot list the account's batches ({exc}), so {len(pending)} "
            f"pre-submit intent(s) cannot be checked against it.")
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for b in listing:
        mid = (b.get("metadata") or {}).get(INTENT_METADATA_KEY)
        if mid:
            by_intent.setdefault(str(mid), []).append(b)
    family = _model_family(cfg.batch_model())
    known = _known_batch_ids(cfg, state)

    # Batches that are neither ours by metadata nor named by any state file. Any one of them could
    # BE an orphan, so their presence turns "no match" into "cannot tell". The model test is
    # deliberately loose -- a listing that names the model in some third way, or does not name it
    # at all, must not narrow this set to empty and hand back the confident "it never became a
    # batch" that costs the money twice.
    def ours(b: Dict[str, Any]) -> bool:
        m = str(b.get("model") or "")
        return not m or _model_family(m) == family

    unaccounted_recs = [
        b for b in listing
        if not (b.get("metadata") or {}).get(INTENT_METADATA_KEY)
        and str(b.get("id")) not in known and ours(b)
    ]
    unaccounted = sorted(str(b.get("id")) for b in unaccounted_recs)

    def fits(b: Dict[str, Any], rec: Dict[str, Any]) -> bool:
        """Could this unaccounted batch be the one this intent created? Only the fields the
        listing actually returns are available: the model (as a family -- the listing answers with
        a snapshot of the requested model), the request count (an intent knows exactly how many
        requests its batch carried) and the creation time (the intent is written immediately
        before the create call, so its batch cannot predate it by more than the clock skew). A
        record that does not name one of them is NOT a fit: a missing field is ignorance, and
        ignorance must not be adopted."""
        if _model_family(b.get("model")) != family:
            return False
        if (b.get("request_counts") or {}).get("total") != len(rec["item_keys"]):
            return False
        created = b.get("created_at")
        if isinstance(created, bool) or not isinstance(created, (int, float)):
            return False
        return created >= float(rec.get("created_at") or 0) - INTENT_CLOCK_SKEW_S

    shaped = {iid: [str(b.get("id")) for b in unaccounted_recs if fits(b, rec)]
              for iid, rec in pending.items()}
    # A batch that fits TWO intents identifies neither: adopting it for one would attach the
    # other's item keys to it, i.e. the mis-attribution this whole layout exists to prevent.
    n_fits: Dict[str, int] = {}
    for ids in shaped.values():
        for bid in ids:
            n_fits[bid] = n_fits.get(bid, 0) + 1
    contested = {bid for bid, n in n_fits.items() if n > 1}

    def adopt(iid: str, rec: Dict[str, Any], bid: str, status: str, why: str) -> None:
        log(f"[judge-api] RECOVERED orphan batch {bid} {why} intent {iid} "
            f"({len(rec['item_keys'])} requests, submitted by {rec.get('scope')!r}): it was "
            f"created and paid for by a run that died before it could record the id.")
        # The intent's OWN item keys, because the batch's custom_ids are ours by construction:
        # this run wrote them from the same (question, response) text.
        state["batches"].setdefault(bid, {
            "item_keys": rec["item_keys"], "submitted_at": rec.get("created_at"),
            "scope": rec.get("scope"), "drained": False, "status": status, "usage": None})
        state["intents"].pop(iid, None)

    # PASS 1: adopt. Done for every intent before anything is dropped, so that a batch adopted by
    # one intent is no longer "unaccounted for" when another intent asks whether it might be its
    # own -- an adopted batch must not keep a second, genuinely stillborn intent pending forever.
    adopted: set = set()
    leftover: Dict[str, tuple] = {}
    for iid, rec in pending.items():
        match = by_intent.get(iid) or []
        fit = [b for b in shaped[iid] if b not in contested]
        if len(match) == 1:
            bid = str(match[0].get("id"))
            adopt(iid, rec, bid, "recovered-from-intent", "from")
            adopted.add(bid)
            continue
        # SHAPE IS REPORTED, NEVER ADOPTED (decision 2026-09-17, after an adversarial pass).
        # `fit` below is carried into the warning so a human can see which batch is probably ours,
        # but the code never acts on it. Shape cannot distinguish our orphan from a same-size batch
        # of the same model family created by anything else: three separate false adoptions were
        # demonstrated, including one where our own orphan had an unreadable `created_at` and the
        # FOREIGN batch was the only fit. A wrong adoption is not merely a wasted batch -- the
        # adopted record claims our item_keys, so every one of our items is then recorded as
        # "absent from the completed batch", which burns the per-item attempt cap and can wedge the
        # judge. The exact discriminator (our custom_ids are content-derived, so a batch carrying
        # them IS ours) was deliberately not pursued: it costs a fetch per candidate and this window
        # is rare, visible and bounded by one batch. An unadoptable orphan therefore stays
        # unresolved, is named in the log, and its items are bought again -- loudly, not silently.
        leftover[iid] = (rec, match, fit)

    # PASS 2: drop what a complete listing rules out, report the rest by name.
    left = [b for b in unaccounted if b not in adopted]
    listing_state = ("complete" if complete else
                     f"INCOMPLETE or unavailable (the API serves at most {LIST_LIMIT_MAX} "
                     f"batches per listing)")
    unresolved = 0
    for iid, (rec, match, fit) in leftover.items():
        if not match and complete and not left:
            log(f"[judge-api] intent {iid} ({len(rec['item_keys'])} requests, "
                f"{rec.get('scope')!r}) never became a batch: the account's listing is complete "
                f"and holds nothing unaccounted for. Its items are submitted again.")
            state["intents"].pop(iid, None)
            continue
        unresolved += 1
        rec["unresolved"] = True
        state["intents"][iid] = rec
        log(f"[judge-api] WARNING: pre-submit intent {iid} ({len(rec['item_keys'])} requests, "
            f"{rec.get('scope')!r}, recorded {rec.get('created_at')}) cannot be reconciled: "
            f"{len(match)} batch(es) carry it in metadata, {len(fit)} unaccounted batch(es) fit it "
            f"by shape (same model family, same request count, created no earlier), listing "
            f"{listing_state}, "
            f"{len(left)} batch(es) of {cfg.batch_model()} unaccounted for"
            + (f" ({', '.join(left[:5])})" if left else "")
            + f". If one of those IS this intent's batch it is paid-for work about to be bought "
            f"again; it is named in {_state_path(cfg)} under \"intents\".")
    _save_state(cfg, state)
    return unresolved


def _add_usage(total: Optional[Dict[str, Any]], one: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Sum two usage payloads field by field. v1 ASSIGNED here, so a run of N chunks reported only
    the last chunk's tokens: 1000+2000+3000 was reported as 3000, i.e. every multi-chunk run
    under-reported its own spend."""
    if not isinstance(one, dict):
        return total
    if not isinstance(total, dict):
        return dict(one)

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    out = dict(total)
    for k, v in one.items():
        cur = out.get(k)
        if num(cur) and num(v):
            out[k] = cur + v
        elif isinstance(cur, dict) and isinstance(v, dict):
            out[k] = _add_usage(cur, v)
        elif cur is None:
            out[k] = v
    return out


def _await_batch(bid, rec, state, items, cfg, api_key, log, *, label, scope) -> Optional[Dict[str, Any]]:
    """Poll one submitted batch to a terminal status and merge its results into `items`."""
    while True:
        payload = fetch(bid, api_key)
        status = payload.get("status")
        counts = payload.get("request_counts") or {}
        if status in TERMINAL:
            break
        log(f"[judge-api] {scope}: {label} {status} {counts.get('completed')}/{counts.get('total')}")
        time.sleep(cfg.poll_seconds)

    rec["status"] = status
    if status != "completed":
        # Terminal and empty. Mark it drained before raising: a failed id that stays adoptable is
        # a wedge -- every relaunch would re-fetch it and die again -- and its items can only make
        # progress by being submitted afresh.
        rec["drained"] = True
        _save_state(cfg, state)
        raise RuntimeError(
            f"{scope}: batch {bid} ended {status!r} (error={payload.get('error')!r}). "
            "Not treating a non-completed batch as an empty result."
        )
    results = extract_results(payload)
    for cid, result in results.items():
        _merge_result(items, cid, result)
    # An item the completed batch simply did not return has always been retried (it never reached
    # `items` at all). Record it as the failure it is instead, so that retry is COUNTED and a
    # request the API will never answer cannot loop for the length of a campaign.
    for cid in rec["item_keys"]:
        if cid not in results:
            _merge_result(items, cid, {"raw": None, "usage": None,
                                       "error": {"message": "absent from the completed batch"}})
    rec["drained"] = True
    rec["usage"] = payload.get("usage")
    _save_state(cfg, state)
    return payload.get("usage")


def api_judge_scores(
    questions: Sequence[str],
    responses: Sequence[str],
    *,
    scope: str,
    cfg: Optional[JudgeConfig] = None,
    api_key: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Judge (question, response) pairs and return the SAME contract as `local_judge_scores`
    plus provenance. `scope` is a HUMAN LABEL for this run (it names the files in the log and in
    every batch record); it keys nothing, because keying the state by the current todo list is the
    defect this layout exists to remove -- see the module docstring.

    Resumption: every item is identified by its own text and instrument, and each submitted batch
    records the item keys it carries, persisted the moment the id comes back. So a restart ADOPTS
    an in-flight job instead of resubmitting it (and paying twice) no matter how the file set,
    the chunk size or the run boundaries have moved since; results already stored are never
    requested again; and a regenerated response is a different item, so its stale score is left
    where it is instead of being written against the new text.
    """
    cfg = cfg or JudgeConfig()
    api_key = api_key or _api_key_from_env()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Refusing to proceed: a missing key must fail loudly, "
            "not silently produce an unjudged file that looks judged."
        )
    if len(questions) != len(responses):
        raise ValueError(
            f"{scope}: {len(questions)} questions vs {len(responses)} responses. Refusing to zip "
            "-- a truncated generation run would otherwise become a plausible-looking mean."
        )

    n_legacy = _report_legacy_state(cfg, log)
    state = _load_state(cfg)
    # BEFORE anything is counted or submitted: a batch an earlier run created but never recorded
    # is already paid for, and only the intent it left behind can find it.
    n_unresolved = _reconcile_intents(cfg, state, api_key, log)
    items: Dict[str, Any] = state["items"]
    keys = item_keys(questions, responses, cfg)
    where = {k: i for i, k in enumerate(keys)}     # keys are unique: duplicates carry an ordinal
    # "Done" is ANSWERED, not "present in state": a stored per-item error is a request that never
    # produced an answer, and counting it as done is what made three upstream 500s permanent.
    needed = [k for k in keys if not _answered(items.get(k))]
    retrying = [k for k in needed if k in items]
    log(f"[judge-api] {scope}: {len(keys)} items, {len(keys) - len(needed)} already answered, "
        f"{len(needed)} to do ({len(retrying)} of them retried after an earlier per-item failure)")

    # Bounded, and then refused BY NAME rather than retried forever or cached forever. Raised
    # before a single request is submitted, so an unattended worker stops instead of spending.
    stuck = [k for k in needed if _attempts(items.get(k)) >= cfg.max_item_attempts]
    if stuck:
        raise RuntimeError(
            f"{scope}: {len(stuck)} of {len(keys)} items still have no answer after "
            f"{cfg.max_item_attempts} attempt(s) each; the last error on {stuck[0]} was "
            f"{(items[stuck[0]] or {}).get('error')!r}. Refusing to ask again -- retrying one item "
            f"forever is how an unattended worker spends a budget on an endpoint that will not "
            f"answer. The attempt counts live in {_state_path(cfg)}: delete these ids from its "
            f"\"items\" map to reset them, or raise max_item_attempts. Item ids: "
            f"{', '.join(stuck[:10])}"
            + (f" (and {len(stuck) - 10} more)" if len(stuck) > 10 else "")
        )

    # ADOPT BEFORE SUBMITTING. A batch is paid for the moment it is created and CANNOT be
    # cancelled, so anything already in flight that carries an item this run needs is collected
    # rather than bought again. Adoption is by item key, so it survives a partial write, a new
    # file appearing, a different --chunk_size and a different run boundary -- every one of which
    # silently invalidated the v1 scope key and resubmitted the fleet.
    needed_set = set(needed)
    adopted = [(bid, rec) for bid, rec in state["batches"].items()
               if not rec.get("drained") and needed_set.intersection(rec["item_keys"])]
    covered = {k for _, rec in adopted for k in rec["item_keys"]}
    remaining = [k for k in needed if k not in covered]

    # SUBMIT EVERY CHUNK FIRST, THEN WAIT. Sequential submit-then-wait cost one full turnaround per
    # chunk: the 19-file sparse re-judge spent 3 h 24 min on chunk0 alone before chunk1 was even
    # created. Measured 2026-09-16 on luna: two 20-item batches submitted 2 s apart while a 10,000
    # item batch of the SAME model was in flight, and one came back in 11 min -- so batches of one
    # model are not strictly serialised. The other took far longer, so the win is not guaranteed;
    # what is guaranteed is that nothing waits on an earlier chunk before it exists.
    # Every id is persisted with its item keys as it is created, so a crash still adopts the fleet.
    # NOTE: a batch cannot be cancelled, so all of them are committed the moment this loop runs.
    labels: Dict[str, str] = {}   # batch id -> this run's label for it, for the log and provenance
    unawaited: List[tuple] = []   # (label, batch id, n_requests) adopted or submitted, not collected
    run_usage: List[Optional[Dict[str, Any]]] = []

    def drain_one() -> None:
        label, bid, _ = unawaited.pop(0)
        u = _await_batch(bid, state["batches"][bid], state, items, cfg, api_key, log,
                         label=label, scope=scope)
        run_usage.append(u)

    for bid, rec in adopted:
        labels[bid] = f"chunk{len(labels)}"
        log(f"[judge-api] {scope}: adopting {labels[bid]} = {bid} ({len(rec['item_keys'])} requests, "
            f"{len(needed_set.intersection(rec['item_keys']))} of them needed here)")
        unawaited.append((labels[bid], bid, len(rec["item_keys"])))
        if not cfg.parallel_chunks:
            drain_one()

    # <= chunk_size requests per job: 14,190 was refused with `413 Batch of more than 10,000
    # requests is not allowed` (before anything was created, so nothing was spent).
    for start in range(0, len(remaining), cfg.chunk_size):
        chunk = remaining[start:start + cfg.chunk_size]
        while unawaited and sum(n for _, _, n in unawaited) + len(chunk) > cfg.max_in_flight_requests:
            log(f"[judge-api] {scope}: {sum(n for _, _, n in unawaited)} requests in flight; waiting "
                f"for {unawaited[0][0]} before submitting {len(chunk)} more")
            drain_one()
        payload = [
            build_request(questions[where[k]], responses[where[k]], k, cfg.max_tokens) for k in chunk
        ]
        # TWO-PHASE: the intent is on disk before the batch can exist, so a kill anywhere in the
        # create call leaves a record that names the batch it may have created. `write_json_atomic`
        # makes the second save all-or-nothing, so the file always holds either the pending intent
        # or the recorded id -- never a submission that appears in neither.
        iid = _record_intent(cfg, state, chunk, scope)
        bid = submit(payload, cfg, api_key, intent_id=iid)
        # Persist BEFORE waiting: a crash must not orphan an id, and the item keys are what makes
        # the id adoptable by a run whose file list has moved on.
        state["batches"][bid] = {"item_keys": chunk, "submitted_at": time.time(), "scope": scope,
                                 "drained": False, "status": "submitted", "usage": None}
        state["intents"].pop(iid, None)
        _save_state(cfg, state)
        labels[bid] = f"chunk{len(labels)}"
        log(f"[judge-api] {scope}: submitted {labels[bid]} = {bid} ({len(chunk)} requests)")
        unawaited.append((labels[bid], bid, len(chunk)))
        if not cfg.parallel_chunks:
            drain_one()
    while unawaited:
        drain_one()

    usage: Optional[Dict[str, Any]] = None
    for u in run_usage:
        usage = _add_usage(usage, u)

    scores: List[Optional[int]] = []
    failures: List[Dict[str, Any]] = []
    for i, k in enumerate(keys):
        rec = items.get(k)
        s = parse_score(rec.get("raw")) if rec else None
        scores.append(s)
        if s is None:
            failures.append(
                {"index": i, "key": k, "raw": (rec or {}).get("raw"),
                 "error": (rec or {}).get("error"), "attempts": _attempts(rec)}
            )

    valid = [s for s in scores if s is not None]
    rate = 1.0 - len(valid) / max(len(scores), 1)
    if rate > cfg.max_failure_rate:
        raise RuntimeError(
            f"{scope}: {len(failures)}/{len(scores)} items produced no score "
            f"({rate:.3%} > pre-registered {cfg.max_failure_rate:.3%}). Refusing to report a mean "
            f"computed from the rest. Their records are in {_state_path(cfg)} under \"items\": an "
            f"item with an \"error\" is re-asked by the next run, one with an unparseable \"raw\" "
            f"is NOT (temperature is 0; the same bytes come back), so a relaunch over the same "
            f"generations raises here again until those ids are removed. "
            f"First failures: {failures[:3]}"
        )
    if not valid:
        raise RuntimeError(f"{scope}: no valid scores at all.")

    return {
        "mean": sum(valid) / len(valid),
        "n": len(valid),
        "scores": scores,
        "judge_meta": {
            "backend": "openrouter-batch",
            "model": cfg.model,
            "model_served": None,
            "prompt_sha256": prompt_fingerprint(),
            "temperature": 0,
            "max_tokens": cfg.max_tokens,
            "provider_only": list(cfg.provider_only) if cfg.provider_only else None,
            "n_requested": len(scores),
            "n_parsed": len(valid),
            "n_failed": len(failures),
            "failure_rate": rate,
            "max_failure_rate": cfg.max_failure_rate,
            "failures": failures[:50],
            # This run's tokens, summed over the batches it collected -- v1 assigned the LAST
            # chunk's usage here and reported 3,000 tokens for a 6,000-token run.
            "usage": usage,
            "batches": {label: bid for bid, label in labels.items()},
            # Money that was spent and can no longer be collected; 0 once the v1 files are gone.
            "legacy_state_unadoptable_batches": n_legacy,
            # Pre-submit intents this run could neither match to a batch nor rule out; each one is
            # possibly a paid-for batch that was bought again here.
            "unresolved_intents": n_unresolved,
            # Items re-asked because an earlier pass got an error rather than an answer for them.
            "n_retried_after_failure": len(retrying),
        },
    }
