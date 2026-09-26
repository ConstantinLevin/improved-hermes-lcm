# Architecture

Hermes-LCM keeps what the host hands it at each compaction in one profile-local SQLite store, `lcm-record.db`, and replaces the older part of the context with summaries of it.

## Core flow

1. Nothing is stored between compactions. The store is filled at a compaction, from the list the host hands over: every entry it does not hold yet is written verbatim, the fresh tail included.
2. The entries outside the fresh tail are cut into chunks, and each chunk is summarised from what the store holds.
3. The context becomes the host's system prompt, one summary row per chunk summarised so far (the earlier ones as they were returned before), and the fresh tail as the host kept it.
4. Recall tools read the store: exact stored messages, or what a summary was made from.

Stored messages are source truth. Summaries are a derived layer with explicit provenance: each names the chunk of stored messages it was made from.

What was said after the last compaction is in the agent's context, not yet in the store; the recall tools find it only after the next compaction.

## Scope model

The recall tools read the current session, across the host identifiers its compactions rotate through, and nothing outside it.

## What the agent is told

The plugin tells the agent what the summaries in its context are through a section of the host's system prompt, `## Plugin Context: lcm`, whose text is `skills/hermes-lcm/references/recall-policy.md`. The host renders it when it first builds a session's prompt and again after every compaction; a session it restores from its database keeps the sections of its stored prompt until the next compaction. It is shown where the `context.engine` of the home the plugin was loaded for is `lcm`. The plugin checks each distinct system prompt a request sends for its section and logs a WARNING in the host's log where it is missing. The plugin also registers two skills, `hermes-lcm:summaries` (working with summaries) and `hermes-lcm:setup` (this one), which the agent reads with `skill_view`.
