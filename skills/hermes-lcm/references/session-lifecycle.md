# Session lifecycle

Hermes `/new` starts a new host session, and Hermes-LCM starts a new session of its own for it. Nothing is carried over from the previous session.

A compaction does not start a new session: the host may give the session a new identifier at a compaction, and Hermes-LCM continues the same session under it.

Do not promise that `/new` deletes historical LCM data. The previous session's stored messages and summaries remain in the store (`lcm-record-12.db`).
