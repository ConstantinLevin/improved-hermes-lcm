# improved-hermes-lcm

A fork of [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm), a context-engine plugin
for [Hermes](https://github.com/nousresearch/hermes-agent). A context engine takes over
compaction: when a session outgrows the model's window, it decides what the agent's context
becomes. hermes-lcm implements the mechanism of the LCM paper, lossless context management
(`docs/lcm-paper.md`): older stretches of the session are summarised, the originals are kept, and
the agent gets tools to fetch them back.

## State of the tree

The code is upstream's v1.0.0-rc.1 (commit 8d1b1e6) with its tests, benchmark harnesses,
documents, release notes and lossless-claw importer removed, and two false lossless claims
deleted. Upstream's opt-in subsystems (recall, embeddings, evidence, assertions, rollups,
extraction) and the switches that lose data on request (redaction, large-output externalisation,
transcript GC, ignore patterns, deletion on `/new`, cleanup commands) are removed too; the agent
gets six tools. The store and the compaction path are the fork's: the plugin keeps its own
record, filled at each compaction from what the host hands over, verbatim, and the context a
compaction returns is emitted from that record (#29, #1, #3). A summary that cannot be written
fails the compaction: nothing is truncated in its place, the context stays as it was, the host
shows the cause, and compaction is tried again at the next occasion (#7). The summariser is the
model running the session, called on the route the host names for it, unless another is
configured (`LCM_SUMMARY_MODEL` with `LCM_SUMMARY_PROVIDER`); a reply from any other model is a
failed summary (#9). The rest of its behaviour is
upstream's, including every loss listed below. The fork is not usable for its purpose yet; the
work is the issues in this repository's tracker.

## What upstream does badly, and what the fork will do instead

Each line was read in this tree's code.

- **Cuts the summariser's input.** Before the summariser sees a chunk, every message over 3,000
  characters is cut to its first 2,000 and its last 800, and every tool call's arguments over 500
  characters to their first 400 (`engine.py`, `_serialize_messages`). The fork's summariser will
  read the store's originals in full (#8).
- **Sizes the fresh tail by message count.** The verbatim tail is the newest 32 messages; a token
  cap exists but is off by default (`fresh_tail.py`, `config.py`). The fork will size the tail in
  tokens, as a weight of the real window (#13).

## Working on it

This is a development path, not an install for use. `scripts/install.sh` links the checkout into
`$HERMES_HOME/plugins/hermes-lcm` and the skill into `$HERMES_HOME/skills/hermes-lcm`. Hermes
then needs both of these in its `config.yaml`:

```yaml
plugins:
  enabled:
    - hermes-lcm
context:
  engine: lcm
```

Host analysis reads a current checkout of Hermes main as a sibling directory named
`hermes-agent`, never a live `~/.hermes` install. CI checks that the plugin loads against the
latest Hermes main and that Hermes selects it as its context engine; it does not check behaviour.
There is no test suite.

## License

MIT, see `LICENSE`.
