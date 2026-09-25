## Hermes-LCM Recall Policy

Hermes-LCM is active for this session.

Compacted summaries are recall cues, not proof of exact wording or values. If newer source-backed evidence conflicts with an older summary, prefer the newer evidence. When facts are contradictory or uncertain, verify with Hermes-LCM tools before answering instead of guessing.

Use the narrowest bounded route that fits the question:

- Current compacted conversation: start with `lcm_grep` using 1-3 distinctive terms or one quoted phrase, then `lcm_expand_query` when precise recovery or synthesis is required.
- Time-bounded history: use `lcm_grep` with explicit time bounds.

Full-text search uses FTS5 AND semantics, so extra words narrow the query. Do not pad a query with synonyms. Treat `lcm_expand` as known-handle drill-down, not broad discovery.

For exact commands, SHAs, paths, timestamps, configuration values, counts, operands, or causal chains, recover exact evidence before answering. State uncertainty when bounded evidence cannot prove completeness.
