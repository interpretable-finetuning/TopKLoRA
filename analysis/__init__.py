"""Archived one-off analysis for completed experiments.

Not part of the library: these modules produced results that are already recorded in
`docs/captains-log.md` and are kept so those entries stay reproducible. They import FROM
`src/` and from each other; nothing in `src/` imports them. Marked `linguist-generated` in
`.gitattributes` so GitHub collapses them in pull-request diffs.

Run from the repo root, e.g. `uv run python -u analysis/analyze_setchurn.py --help`
(`--with matplotlib` for the figure scripts).
"""
