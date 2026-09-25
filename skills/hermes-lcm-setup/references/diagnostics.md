# Diagnostics

Use read-only product tools before changing configuration or running an apply path.

## Fast path

1. `hermes plugins`: confirm `hermes-lcm` is enabled and the selected context engine is `lcm`.
2. Send one normal message if the session has not been bound since restart.
3. `lcm_status`: inspect runtime identity, database path, context pressure, summary/store counts, and filters.
4. `lcm_inspect`: inspect current-session lineage, the stored fresh tail, and skip/no-op reasons without retrieving content.
5. `lcm_doctor`: run database, FTS, configuration, and context-pressure diagnostics, check the record's invariant (every record on the branch is in the returned tail or under exactly one summary of the latest compaction), list the store's identity and its recent events (what the plugin could not do), and show the daily backup slot and its age.

If optional slash commands are enabled, `/lcm status` and `/lcm doctor` expose the corresponding operator views.

## Safe mutation order

For repair (the only one is rebuilding a full-text index, `/lcm doctor repair apply`, which rebuilds derived data from the insert-only record):

1. run the read-only preview (`/lcm doctor repair`);
2. inspect exact candidates and paths;
3. obtain user authorization for the specific apply operation;
4. run one bounded apply and verify integrity afterward.

## The daily backup

The plugin keeps one backup of each store: `<store directory>/backups/lcm/<store name>.daily.sqlite3`. It is taken automatically, at most once a day and only when the store changed since, through SQLite's backup API. Each copy is checked (`integrity_check`, the store's identity, the record's invariant) before it replaces the slot, so the slot is always the last copy that passed. A failed backup keeps the old slot and appears in the doctor's recent events as `backup_failed`. A slot that holds another store's backup is set aside as `<store name>.daily.<uuid>.sqlite3` and never overwritten. There is no backup command.

## Restore

A restore is the owner's, by hand, with every Hermes process on that home stopped:

1. stop Hermes (gateway, CLI, dashboard) on this home;
2. move the live store aside, for example to `lcm-record.db.before-restore`. Never delete it: it holds everything written since the backup;
3. write the slot into place through SQLite: `sqlite3 backups/lcm/lcm-record.daily.sqlite3 ".backup lcm-record.db"` (run in the store's directory);
4. start Hermes again.

A session that was running continues from its host context, which holds summaries and rows the restored store does not have. So a running session aborts visibly at its next compaction after a restore ("Compression aborted: …", nothing changed in its context). Begin a new session with `/new`.

## Common states

- Unbound status after restart: send a normal message, then check again.
- Database exists but stays empty: nothing is stored before the session's first compaction; after one, verify plugin enablement, `context.engine`, profile, and database path.
- Weak exact recall: verify source rows exist, query construction/scope is correct, and summary health is sound.
- Conflicting summary and raw evidence: prefer the newer exact raw evidence and inspect lineage.
- An `instruction_not_delivered` event in the doctor's recent events: the agent is not told what its summaries are. Either the plugin could not give the host its system-prompt section (the host offers none to it, its limits cannot be read, or the text is over its limit), or a request of an LCM session went out without the section; the event's detail names the host session and the cause, such as the host's total for all plugin sections or a prompt the host restored from before the section existed.
- Path B/context-engine schema log: expected on hosts where plugin-registry handlers do not receive active messages; context-engine schemas and dispatch remain the healthy route.
