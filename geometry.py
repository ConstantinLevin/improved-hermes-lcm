"""The trigger's geometry from the window W (#31, Decided; #11, #32).

All three sizes are provider tokens, compared with the host's real count:

- the raised threshold inside a turn, τ′ = min(W − 123k, 0.85 · W): 123k is the most
  one tool round adds at the host's per-round budget (50k by the estimate times its
  p95 error 1.95, plus 25k of the agent's own output, its p99.9); 0.85 · W is the
  gateway's hygiene path, a bound until Hermes answers #11's ask;
- the threshold τ = τ′ − 54.5k, the margin being the count-p95 of the growth inside a
  human-begun turn;
- the target G = min(0.3 · W, τ), a context size.

The weights come from the configuration with their sources recorded (R14). Below
256k τ turns meaningless (negative below about 180k), and #21's bounds on the window
are 256k to 2M: outside them nothing is computed, and the plugin compacts nothing and
says why (R11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Geometry:
    window: int
    tau_raised: int   # τ′
    tau: int          # τ
    target: int       # G

    def label(self) -> str:
        return (f"W {self.window}: τ {self.tau}, τ′ {self.tau_raised}, G {self.target} provider tokens "
                f"(#31: τ′ = min(W − 123k, 0.85·W), τ = τ′ − 54.5k, G = min(0.3·W, τ))")


def geometry(window: int, config: Any) -> tuple[Optional[Geometry], str]:
    """(the geometry, "") for a window within the bounds, else (None, why not)."""
    low, high = int(config.window_min_tokens), int(config.window_max_tokens)
    if not window or window <= 0:
        return None, "the window is not known: the host has named no context length"
    if window < low or window > high:
        return None, (f"the window of {window} tokens lies outside the bounds the geometry is defined for "
                      f"({low} to {high}, #21, R11): τ, τ′ and G are not computed, and nothing is compacted")
    tau_raised = int(min(window - config.round_growth_tokens, config.hygiene_share * window))
    tau = int(tau_raised - config.turn_margin_tokens)
    target = int(min(config.target_share * window, tau))
    return Geometry(window=window, tau_raised=tau_raised, tau=tau, target=target), ""
