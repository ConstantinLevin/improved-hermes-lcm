---
name: hermes-lcm
description: Working with the summaries in your context under Hermes-LCM, and recalling exact evidence from behind them.
---

# Hermes-LCM: the summaries in your context

Use this skill when a task depends on the compacted part of your context: the summaries that stand for older stretches of the session, and what lies behind them.

Start here:

1. For exact historical claims, use the recall workflow instead of trusting a compacted summary.
2. Load the relevant reference rather than guessing arguments or lifecycle semantics.

Reference map:

- Recall tools and routing: `references/recall-tools.md`
- `/new` and session continuity: `references/session-lifecycle.md`
- Canonical runtime recall policy (also in your system prompt): `references/recall-policy.md`

Setting the plugin up, configuring it and checking its health is the skill `hermes-lcm:setup`.

Working rules:

- Raw stored messages are authoritative; summaries are bounded recall cues.
- Prefer newer source-backed evidence when it conflicts with an older summary.
- Start with the narrowest useful scope and expand only when exact detail is needed.
- Do not infer exact commands, paths, timestamps, values, counts, or causal chains from summaries alone.
- Do not treat open-cardinality results as complete without product-verifiable enumeration or coverage.
