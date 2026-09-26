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
failed summary (#9). The summariser reads a chunk's stored records as the messages they were, in
full: tool calls and results whole, readable reasoning marked, images only where it reads them;
encrypted reasoning is withheld until the plugin knows which provider produced it (#8). A
compaction's summariser calls, one per chunk, run at once, through one limiter per endpoint
(#33). No chunk below a quarter of the chunk size is ever sent: such a rest joins a neighbouring
chunk or stays raw before the tail. Every chunk is recorded before any of its calls starts, and a
retry, in any process, keeps every recorded chunk of the earlier attempts: a summarised one keeps
its summary, any other is retried as the same chunk, and only what is new is cut. A chunk of the
same messages failing by its own fault
(its reply rejected, or its request refused) in three consecutive attempts is shown as an error
naming it, and is still retried; rate limits, deadlines and other endpoint failures do not count
(#33, #7). A chunk with rows that came without a host identity (the gateway's replayed
history, or host scaffolding the host never persists) cannot be found again by a retry, which
cuts it again and names those rows (the ask to Hermes: A1). Compaction runs
at a threshold derived from the model's window, raised by a margin while a turn runs, and
brings the context down to a target G (#11, #31, #32); the material outside the tail is split
into equal chunks of 50k provider tokens (#12), and the tail takes what the target leaves,
sized in tokens, in whole tool groups (#13). The plugin counts by its own estimate, characters
divided by four, converted to provider tokens by a measured ratio and labelled as an estimate
wherever it is shown (#21). The plugin tells the agent what its summaries are in a section of
the host's system prompt, shown where LCM is the home's context engine, and registers two
skills, `hermes-lcm:summaries` and `hermes-lcm:setup`; no hook injects text into the user's
messages, and a request whose system prompt lacks the section is logged as that fact (#16). The fork is not usable for its purpose yet: condensation (#34), the
re-insertion inside a turn (#14) and the rest of the issues in this repository's tracker are
still to come.

## Working on it

This is a development path, not an install for use. `scripts/install.sh` links the checkout into
`$HERMES_HOME/plugins/hermes-lcm` and the summaries skill into `$HERMES_HOME/skills/hermes-lcm`. Hermes
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
