---
name: hermes-lcm-setup
description: Setting up the Hermes-LCM lossless context plugin, configuring it, and checking that it is set up and healthy.
---

# Hermes-LCM: setup and health

Use this skill when a task concerns Hermes-LCM setup, configuration, compaction behaviour, diagnostics or repair.

Start here:

1. Confirm that the `hermes-lcm` plugin is enabled and `context.engine` is `lcm`.
2. Use `lcm_status`, `lcm_inspect`, and `lcm_doctor` before changing configuration or attempting repair.
3. Treat slash-command apply paths as mutations: preview first and require the user's authorization. The store's backup is automatic and daily (`references/diagnostics.md`).
4. Load the relevant reference rather than guessing settings or behaviour.

Reference map:

- Configuration and activation: `references/configuration.md`
- Architecture, data ownership, and what the agent is told: `references/architecture.md`
- Diagnostics and safe operator workflow: `references/diagnostics.md`

Working with the summaries in the context is the skill `hermes-lcm:summaries`.
