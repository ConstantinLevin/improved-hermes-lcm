# Session lifecycle and rotate

Hermes `/new` starts a new host session. Hermes-LCM binds that session to its own lifecycle row.

Do not promise that `/new` deletes historical LCM data. The previous session's raw messages and summaries remain in `lcm.db`.

## `/lcm rotate`

`/lcm rotate` is different from `/new`:

- it keeps the current `session_id` and `conversation_id`;
- preview is read-only;
- apply creates/updates the rolling rotate backup first;
- it preserves the configured fresh tail;
- it advances the lifecycle frontier past older raw messages so bootstrap does not replay them into active context;
- it does not delete raw source rows or call a summarization model.

Run normal compaction before rotate when older material must be represented in summary nodes. Without a summary, pre-tail raw rows remain in the store but no summary points at them.

Rotate refuses stateless sessions. Repeating an already-satisfied rotate reports a no-op and preserves the previous known-good rolling backup.

Use a separate session when the user wants a new active conversational boundary. Use rotate when the problem is active transcript/frontier size without changing identity.
