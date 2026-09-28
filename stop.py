"""The host's stop of one dispatched tool call, read at the engine's boundary (PLAN-19 re-derived,
M-BOUNDARY-FENCE; before it PLAN-83d §5, PLAN-19 §2.7, D-4). The rule that the store commits nothing
once the host has asked this call to stop reaches every write the call makes, so the latch is
created by ``LCMEngine.handle_tool_call`` before anything else of the call (the settling of the
list the host handed over included) and handed to the handler; the query takes it from there.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .summariser_input import _strict_import


def stop_latch() -> Any:
    """The host's stop of this tool call, read from the engine's boundary, before any store access
    of the call. Two signals, both latched: once seen, the stop holds for the rest of this call.

    - The worker's interrupt bit. The host sets it on a sequential tool timeout only after it has
      stopped waiting (agent/tool_executor.py 963-976), and on an interrupt while it still waits its
      3 s grace (947-950); every ``clear_interrupt`` clears the bit of every tracked worker, an
      abandoned one included: at the turn's end (agent/turn_finalizer.py 731) and at the clear sites
      where a model-request redirect is pending (turn_api_call.py 152, 184; turn_api_error.py 154;
      turn_recovery.py 1349; interrupt_control.py 231-232 returns without clearing otherwise), and at
      turn_recovery.py 1327, codex_runtime.py 486, turn_facade_lease.py 376, tui_gateway/
      prompt_turn.py 155, hermes_cli/cli_chat_turn_mixin.py 639 (at Hermes 375930d089, reader C of
      PLAN-19). A redirect during tool execution sets no bit (it requests a yield, 285-295). The
      check reads this thread by its id, because the host also calls it from the daemon thread that
      runs the provider call (agent/auxiliary_client.py 476; tools/interrupt.py 61-71).
    - The host's own sequential tool deadline (``_resolve_sequential_tool_timeout``: the config
      ``timeouts.tools.sequential_call``, else ``timeouts.tools.concurrent_batch``, else the env
      HERMES_CONCURRENT_TOOL_TIMEOUT_S, else 420 s; 0 or less disables it; tool_executor.py 833-840,
      182-191, deadline.py 153-178), read at the boundary and counted from there. The host set its
      deadline when it dispatched the worker (tool_executor.py 937), before the worker ran the host's
      own steps (a managed Relay pipeline where one is enabled, relay_tools.py 16-70; the tool_request
      and tool_execution middleware, hermes_cli/middleware.py 102-214; the pre_tool_call hooks, an
      approval wait among them, plugins.py 1945-1997 under a gate lock of at most 360 s,
      tool_executor.py 527-546; the pruned-argument scan and the guardrails, 713-726) and before the
      engine's boundary, so this deadline passes later than the host's by those steps; and the host
      extends its own deadline by the seconds an approval wait took (861-886), which no read from the
      worker gives. Nothing on the worker's side marks it abandoned (ask A-D3.1).

    The returned function has ``deadline_s``: the host's timeout it counts, or None where the host's
    deadline is disabled."""
    worker = threading.get_ident()
    thread_interrupted = _strict_import("tool interrupt bit", "tools.interrupt", "is_thread_interrupted",
                                        so="whether the host asked this call to stop cannot be known")
    resolve = _strict_import("sequential tool timeout", "agent.tool_executor", "_resolve_sequential_tool_timeout",
                             so="when the host stops waiting for this call is not known")
    value = resolve()
    deadline_s = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None
    deadline = time.monotonic() + deadline_s if deadline_s is not None else None
    stopped = threading.Event()

    def interrupted() -> bool:
        if not stopped.is_set() and (thread_interrupted(worker)
                                     or (deadline is not None and time.monotonic() >= deadline)):
            stopped.set()
        return stopped.is_set()
    interrupted.deadline_s = deadline_s  # type: ignore[attr-defined]
    return interrupted
