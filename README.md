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
model running the session, called through the host on the route the host names for it. A
summariser other than the session's model (`LCM_SUMMARY_MODEL`, `_PROVIDER`, `_BASE_URL`,
`_API_KEY`, `_API_MODE`) is not supported in this build and is refused at load (#68). A reply
from any other model is a failed summary (#9). The summariser reads a chunk's stored records as the messages they were, in
full: tool calls and results whole, readable reasoning marked, images only where it reads them;
encrypted reasoning is withheld until the plugin knows which provider produced it (#8). A
compaction's summariser calls, one per chunk, run at once, through one limiter per endpoint
(#33). A rest below a quarter of the chunk size joins a neighbouring chunk or stays raw before the
tail; no chunk exceeds what the summariser reads in one call, so a rest no neighbour can take
within that bound is sent alone, with a warning. Every chunk is recorded before any of its calls starts, and a
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
into equal chunks of 50k provider tokens, smaller where the summariser reads less in one call
(#12, #34 D4), and the tail takes what the target leaves,
sized in tokens, in whole tool groups (#13). The plugin counts by its own estimate, characters
divided by four, converted to provider tokens by a measured ratio and labelled as an estimate
wherever it is shown (#21). The plugin tells the agent what its summaries are in a section of
the host's system prompt, shown where LCM is the home's context engine, and registers two
skills, `hermes-lcm:summaries` and `hermes-lcm:setup`; no hook injects text into the user's
messages, and a request whose system prompt lacks the section is logged as that fact (#16). The fork is not usable for its purpose yet: condensation (#34), the
re-insertion inside a turn (#14) and the rest of the issues in this repository's tracker are
still to come.

`LCM_SUMMARY_TIMEOUT_MS` sets the plugin's requested transport timeout for each host
invocation, including retries and the second summary level. Its default is 300,000 ms:
a conservative policy hypothesis, constant across session windows because one call
summarises one bounded chunk. The earlier measurements in #33 report DeepSeek calls up
to 57.2 s and Claude Code whole compactions up to 191 s; these are endpoint-specific
elapsed times, not an idle-time percentile. Five minutes leaves generous headroom;
it is configurable and is not a guarantee for every model. Invalid environment values
use this plugin default with a warning and recorded source; invalid manual configuration
refuses a call. Status and doctor expose the value and any problem.

The plugin always passes the requested timeout to the host. On the current OpenRouter
route it limits individual connect/read/write/pool operations, not total call time or
absence of model output. The host's Anthropic and native Bedrock Converse adapters
ignore this request parameter. Host stream limits, retries and recovery still apply;
there is no uniform physical request lifetime bound. A caller's absolute deadline only
governs new plugin dispatches and retries. An existing call keeps its initiating worker's
settings and limiter slot until its synchronous host invocation ends, even if an attempt
ends or another joins. A valid late summary can be stored without replacing that attempt's
context.

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
