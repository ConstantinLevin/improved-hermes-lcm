# Session lifecycle

Hermes `/new` starts a new host session. Hermes-LCM binds that session to its own lifecycle row.

Do not promise that `/new` deletes historical LCM data. The previous session's raw messages and summaries remain in `lcm-record.db`.
