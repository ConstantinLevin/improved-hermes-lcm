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
compaction returns is emitted from that record (#29, #1, #3). The rest of its behaviour is
upstream's, including every loss listed below. The fork is not usable for its purpose yet; the
work is the issues in this repository's tracker.

## What upstream does badly, and what the fork will do instead

Each line was read in this tree's code.

- **Cuts the summariser's input.** Before the summariser sees a chunk, every message over 3,000
  characters is cut to its first 2,000 and its last 800, and every tool call's arguments over 500
  characters to their first 400 (`engine.py`, `_serialize_messages`). The fork's summariser will
  read the store's originals in full (#8).
- **Ends in truncation.** When neither of the summariser's two levels returns a result shorter than
  its source, the third level keeps the head and the tail of the source text around a marker,
  within 512 tokens by default, and that stands as the chunk's summary (`escalation.py`). In the fork,
  a summary that cannot be written leaves the context untouched, and compaction is tried again (#7).
- **Summarises with the host's auxiliary model.** The summariser calls the host's auxiliary client
  for the task "compression"; with no `summary_model` configured, the host picks the model
  (`escalation.py`, `config.py`). The fork's summariser will default to the model running the
  session (#9).
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
