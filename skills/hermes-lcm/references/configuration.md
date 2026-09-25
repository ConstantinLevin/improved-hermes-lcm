# Configuration and activation

Hermes-LCM is a general Hermes plugin and a context engine. Both identities must be active:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine configuration. Verify with `hermes plugins`, then use `lcm_status` after a normal message has bound the session.

## Installation

An existing checkout can install profile-aware plugin and skill links:

```bash
./scripts/install.sh
HERMES_PROFILE=myprofile ./scripts/install.sh
```

The installer exposes both:

- `plugins/hermes-lcm` for plugin loading;
- `skills/hermes-lcm` for normal skill discovery.

It refuses conflicting paths rather than overwriting an existing install.

## High-impact controls

Start with:

- compaction runs at a threshold derived from the model's context window, τ = min(W − 123k, 0.85·W) − 54.5k, raised by the 54.5k margin while a turn runs; `lcm_status` shows τ, τ′ and the target G. The weights are `LCM_ROUND_GROWTH_TOKENS`, `LCM_HYGIENE_SHARE`, `LCM_TURN_MARGIN_TOKENS` and `LCM_TARGET_SHARE`; the host's own compression threshold is not read;
- `LCM_FRESH_TAIL_COUNT`: newest messages kept raw;
- `LCM_CHUNK_TOKENS`: the chunk size in provider tokens (default 50,000); the material outside the fresh tail is split equally into chunks of at most this size, cut by the plugin's estimate as this size divided by `LCM_ESTIMATE_RATIO` (default 1.51);
- `LCM_DATABASE_PATH`: profile-local SQLite path when the default is unsuitable;
- summary provider settings only after confirming credentials, cost, and data handling.

Optional slash commands are disabled by default with `LCM_ENABLE_SLASH_COMMAND=false`. Do not enable mutation surfaces merely to diagnose a problem.

Change one tuning variable at a time, then re-check `lcm_status`, context pressure, summary health, latency, and actual answer quality.
