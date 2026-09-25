"""Session-scoped runtime-state resets.

``ResetStateMixin`` clears the session-scoped counters when a session is reset or
another one begins. State stays on the engine (accessed via ``self``).
"""


class ResetStateMixin:
    def _reset_session_counters(self) -> None:
        """Reset session-scoped counters and token tracking."""
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""

    def _reset_session_scoped_runtime_state(self) -> None:
        """Reset all session-scoped runtime state."""
        self._reset_session_counters()
