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
- Each hit carries its `handle`. A message hit with `revises` is a summary row as the host rewrote it in the context (for example with the task list folded in); `lcm_expand(handle=…)` on the handle in `revises` opens the summary's stretch.

Do not treat a short search snippet as sufficient evidence for a detail-heavy answer.

### `lcm_expand_query`

Use when current-session compacted material must be expanded and synthesized into a precise bounded answer.

- Always provide `prompt`.
- Provide either a small `query` or the summaries' `handles` when known.
- `query` follows the same narrow FTS construction rules as `lcm_grep`.
- The expansion path is model-backed and bounded by answer/context token limits.

Recommended current-session escalation:

1. `lcm_grep` to locate relevant material.
2. `lcm_expand_query` when exact detail was compressed away.

### `lcm_expand`

Use as low-level drill-down after a known handle (`handle`):

- a summary's (`s…`) or a chunk's (`c…`) handle returns that stretch: the messages verbatim, the readable reasoning beside them, each tool call with its handle (`t…`), name and arguments, without its result; `raw=true` puts the results inline;
- a tool call's handle returns its result: the stored result that carries the call's id, before the next user or agent message; where none does, a note says so; where calls of one message share an id, every result of that id is returned and a note says the store cannot tell which answered which call; a message's handle (`m…`) returns that message;
- a handle whose message the host has since rewritten is refused with the handle that stands for it now; expand that one;
- a result longer than one page carries `next_page`; pass it as `page` for the rest; if what the handle opens into changed since the first page, the page is refused and you start again from the handle. A page never leaves anything out: where one part cannot fit even on a page by itself, the call is refused, naming that part, its size and the page's limit, which is larger when `lcm_expand` is the only call in its message.

Do not use it as broad first-step discovery.

## Operator tools

`lcm_status`, `lcm_inspect`, and `lcm_doctor` report health and metadata. They do not replace content retrieval.
