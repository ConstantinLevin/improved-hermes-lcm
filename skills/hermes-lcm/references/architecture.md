# Architecture

Hermes-LCM keeps raw messages in profile-local SQLite and builds a summary DAG to keep active context bounded.

## Core flow

1. The active context engine ingests messages into `lcm.db`.
2. Older eligible messages are compacted into leaf summaries.
3. Summary nodes can be condensed to higher DAG depths.
4. Context assembly combines selected summaries with a protected fresh raw tail.
5. Recall tools recover exact source rows or bounded expanded context when summaries are insufficient.

Raw messages are source truth. Summary nodes are a derived layer with explicit provenance.

## Scope model

- Current-session DAG operations use the active engine/session binding.
- Hermes `session_search` covers host-tracked history outside `lcm.db`.

Do not silently treat those stores or scopes as interchangeable.
