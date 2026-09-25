# Recall tools

Use recall tools when the answer depends on historical evidence that may have been compacted.

The tools read what the session's compactions stored. What was said since the last compaction is in the context and not yet stored.

## Current compacted conversation

### `lcm_grep`

Use for discovery across current-session stored messages and summaries.

- `query` is FTS5 text by default; it is not a regex.
- Prefer 1-3 distinctive terms or one quoted phrase because FTS5 combines extra terms with AND.
- Keep `sort='recency'` for recent events, use `sort='relevance'` for the strongest older match, and use `sort='hybrid'` when both matter.
- Exact role/time filters apply before limiting where supported. `time_from`/`time_to` compare the time a message was stored, which is the time of the compaction that stored it, not the time it was said.
- `sort='recency'` follows the conversation's order, newest first.
- A message hit with `revises_node_id` is a summary row as the host rewrote it in the context (for example with the task list folded in); `lcm_expand(node_id=…)` on that id opens the summary's sources.

Do not treat a short search snippet as sufficient evidence for a detail-heavy answer.

### `lcm_expand_query`

Use when current-session compacted material must be expanded and synthesized into a precise bounded answer.

- Always provide `prompt`.
- Provide either a small `query` or explicit `node_ids` when known.
- `query` follows the same narrow FTS construction rules as `lcm_grep`.
- The expansion path is model-backed and bounded by answer/context token limits.

Recommended current-session escalation:

1. `lcm_grep` to locate relevant material.
2. `lcm_expand_query` when exact detail was compressed away.

### `lcm_expand`

Use as low-level drill-down after a known handle:

- `node_id` expands a current-session summary with source pagination;
- `store_id` recovers one stored message as the host handed it over, with content pagination.

Do not use it as broad first-step discovery.

## Operator tools

`lcm_status`, `lcm_inspect`, and `lcm_doctor` report health and metadata. They do not replace content retrieval.
