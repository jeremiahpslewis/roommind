"""Learned AC gap→cooling-rate response for RoomMind.

An AC regulates against its own return-air sensor, not the room sensor, and its
compressor and fan output scale with the gap between that sensor and the
setpoint it is given. Two quantities fully describe the actuator:

    offset  = T_head - T_room                 (measured, not learned)
    rate    = f(gap)                          (learned, monotone, saturating)

where ``gap`` is measured at the head — ``T_head - setpoint`` — and ``rate`` is
the *incremental* cooling the room gains in °C/h beyond what it would have done
with the HVAC off. The RC model already predicts that passive drift, so the
residual is attributable to the unit.

Commanding then inverts the curve instead of guessing a boost::

    setpoint = (target + offset) - gap_for_rate(required_rate)

The offset is the intercept: it is what makes "hold the room at 22" mean
"command 22 + offset". The curve is the slope, and because it saturates it also
answers the question no hand-tuned gain can — the gap past which more demand
buys noise instead of cooling.

``f`` is a shape-preserving (monotone) cubic Hermite spline over fixed knots.
Monotone cubic interpolation is used rather than a natural cubic spline because
an unconstrained spline through noisy field data overshoots between knots, and a
non-monotone response curve cannot be inverted. Knot values are updated online
with kernel-weighted recursive averaging and then projected back onto the
monotone cone, so the curve stays invertible after every observation.

Pure Python by design: the integration declares no requirements, so numpy/scipy
are unavailable.
"""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

# Knots in K of head gap. Dense where a 0.5 °C-resolution head can still
# resolve differences, sparse out where the compressor is saturating anyway.
DEFAULT_KNOTS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Kernel width (K) for attributing an observation to neighbouring knots.
KERNEL_WIDTH = 0.75

# Per-knot observations before that knot is trusted.
MIN_KNOT_SAMPLES = 4
# Knots that must be identified before the curve may drive a setpoint.
MIN_IDENTIFIED_KNOTS = 3
# Total observations before the curve may drive a setpoint.
MIN_TOTAL_SAMPLES = 30
# Span of observed gaps (K) required before the curve may drive a setpoint. A
# controller in steady state sits at one gap forever, and the kernel would
# happily certify the knots around it from that single operating point — but a
# curve fitted to one gap is flat, and a flat curve carries no information about
# which gap to pick. Only a spread of gaps identifies a slope.
MIN_GAP_SPREAD = 1.0

# Observations outside these bounds are rejected as unphysical / confounded.
MAX_PLAUSIBLE_RATE = 20.0  # °C/h of incremental cooling or heating
MIN_OBSERVATION_DT = 1.0  # minutes
MAX_OBSERVATION_DT = 20.0  # minutes — longer intervals drift too far from the linearization

# HA hvac states that count as the unit actively working in each RoomMind mode.
RUNNING_STATES: dict[str, frozenset[str]] = {
    "cooling": frozenset({"cool"}),
    "heating": frozenset({"heat", "heat_cool", "auto"}),
}

# Offset smoothing. Tracked separately for running and idle because a
# high wall unit reads warm when idle (stratification) and close to return-air
# temperature once its fan is moving air — the running value is the one that
# matters when commanding.
OFFSET_ALPHA = 0.1
MAX_PLAUSIBLE_OFFSET = 10.0


def _monotone_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """Fritsch-Carlson tangents: cubic Hermite slopes that preserve monotonicity."""
    n = len(xs)
    if n < 2:
        return [0.0] * n
    deltas = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    slopes = [0.0] * n
    slopes[0] = deltas[0]
    slopes[-1] = deltas[-1]
    for i in range(1, n - 1):
        if deltas[i - 1] * deltas[i] <= 0:
            slopes[i] = 0.0  # local extremum — flatten to avoid overshoot
        else:
            slopes[i] = (deltas[i - 1] + deltas[i]) / 2.0
    # Fritsch-Carlson limiter
    for i in range(n - 1):
        if deltas[i] == 0:
            slopes[i] = 0.0
            slopes[i + 1] = 0.0
            continue
        a = slopes[i] / deltas[i]
        b = slopes[i + 1] / deltas[i]
        s = a * a + b * b
        if s > 9.0:
            t = 3.0 / (s**0.5)
            slopes[i] = t * a * deltas[i]
            slopes[i + 1] = t * b * deltas[i]
    return slopes


class GapResponse:
    """Monotone spline mapping head gap (K) → incremental HVAC rate (°C/h)."""

    def __init__(self, knots: tuple[float, ...] = DEFAULT_KNOTS) -> None:
        self.knots: list[float] = list(knots)
        # Seeded flat at zero; every knot starts unidentified and the curve
        # refuses to drive anything until real observations arrive.
        self.values: list[float] = [0.0] * len(self.knots)
        self.counts: list[int] = [0] * len(self.knots)
        self.n_observations: int = 0
        self.gap_min: float | None = None
        self.gap_max: float | None = None

    # ---------------------------------------------------------------- learning

    def observe(self, gap: float, rate: float) -> bool:
        """Fold one (gap, incremental rate) observation into the curve.

        Returns True if the observation was accepted.
        """
        if gap <= 0 or not (0.0 <= rate <= MAX_PLAUSIBLE_RATE):
            return False

        weights = [self._kernel(gap, k) for k in self.knots]
        total = sum(weights)
        if total <= 1e-6:
            return False

        for i, w in enumerate(weights):
            if w <= 1e-3:
                continue
            share = w / total
            # Recursive weighted mean: knots near the observed gap move most.
            self.counts[i] += 1
            step = share / (1.0 + self.counts[i] * share)
            self.values[i] += step * (rate - self.values[i])
        self.n_observations += 1
        self.gap_min = gap if self.gap_min is None else min(self.gap_min, gap)
        self.gap_max = gap if self.gap_max is None else max(self.gap_max, gap)
        self._project_monotone()
        return True

    def _kernel(self, gap: float, knot: float) -> float:
        d = abs(gap - knot) / KERNEL_WIDTH
        if d >= 2.0:
            return 0.0
        return (1.0 - (d / 2.0) ** 2) ** 2

    def _project_monotone(self) -> None:
        """Project knot values back onto the non-decreasing cone.

        More gap can never buy less cooling; a dip is noise, not physics. A
        single forward pass with a running max is enough to keep the curve
        invertible, and it never moves a knot downward below its neighbour.
        """
        running = 0.0
        for i, v in enumerate(self.values):
            if v < running:
                self.values[i] = running
            else:
                running = v

    # -------------------------------------------------------------- evaluation

    @property
    def identified_knots(self) -> int:
        return sum(1 for c in self.counts if c >= MIN_KNOT_SAMPLES)

    @property
    def gap_spread(self) -> float:
        if self.gap_min is None or self.gap_max is None:
            return 0.0
        return self.gap_max - self.gap_min

    def is_confident(self) -> bool:
        """True when the curve has enough support to drive a setpoint."""
        if self.n_observations < MIN_TOTAL_SAMPLES:
            return False
        if self.identified_knots < MIN_IDENTIFIED_KNOTS:
            return False
        if self.gap_spread < MIN_GAP_SPREAD:
            return False
        # A curve that learned "no gap ever cools" is not usable.
        return max(self.values) > 0.0

    def rate_for_gap(self, gap: float) -> float:
        """Evaluate the monotone spline at *gap* (K)."""
        xs, ys = self.knots, self.values
        if gap <= xs[0]:
            # Linear from the origin: zero gap must mean zero incremental rate.
            return ys[0] * (gap / xs[0]) if xs[0] > 0 else 0.0
        if gap >= xs[-1]:
            return ys[-1]
        slopes = _monotone_slopes(xs, ys)
        for i in range(len(xs) - 1):
            if xs[i] <= gap <= xs[i + 1]:
                h = xs[i + 1] - xs[i]
                t = (gap - xs[i]) / h
                t2, t3 = t * t, t * t * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                return h00 * ys[i] + h10 * h * slopes[i] + h01 * ys[i + 1] + h11 * h * slopes[i + 1]
        return ys[-1]

    def gap_for_rate(self, rate: float, max_gap: float) -> float:
        """Smallest gap (K) whose learned response reaches *rate* °C/h.

        Saturation is answered honestly: if the curve never reaches the
        requested rate, the gap where it stops climbing is returned rather than
        an extrapolation, because past that point more gap is only noise.
        """
        if rate <= 0:
            return 0.0
        ceiling = min(max_gap, self.knots[-1])
        if rate >= self.rate_for_gap(ceiling):
            return ceiling
        lo, hi = 0.0, ceiling
        for _ in range(40):  # bisection to well under head resolution
            mid = (lo + hi) / 2.0
            if self.rate_for_gap(mid) < rate:
                lo = mid
            else:
                hi = mid
        return round(hi, 2)

    # ----------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "knots": list(self.knots),
            "values": [round(v, 4) for v in self.values],
            "counts": list(self.counts),
            "n_observations": self.n_observations,
            "gap_min": self.gap_min,
            "gap_max": self.gap_max,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GapResponse:
        knots = tuple(data.get("knots") or DEFAULT_KNOTS)
        obj = cls(knots)
        values = data.get("values") or []
        counts = data.get("counts") or []
        if len(values) == len(obj.knots):
            obj.values = [float(v) for v in values]
        if len(counts) == len(obj.knots):
            obj.counts = [int(c) for c in counts]
        obj.n_observations = int(data.get("n_observations", 0))
        obj.gap_min = data.get("gap_min")
        obj.gap_max = data.get("gap_max")
        obj._project_monotone()
        return obj


class HeadOffset:
    """Smoothed T_head - T_room, tracked separately for running and idle."""

    def __init__(self) -> None:
        self.running: float | None = None
        self.idle: float | None = None
        self.n_running: int = 0
        self.n_idle: int = 0

    def observe(self, head_temp: float, room_temp: float, *, is_running: bool) -> bool:
        offset = head_temp - room_temp
        if abs(offset) > MAX_PLAUSIBLE_OFFSET:
            return False
        if is_running:
            self.running = offset if self.running is None else self.running + OFFSET_ALPHA * (offset - self.running)
            self.n_running += 1
        else:
            self.idle = offset if self.idle is None else self.idle + OFFSET_ALPHA * (offset - self.idle)
            self.n_idle += 1
        return True

    def commanding_offset(self) -> float:
        """Offset to apply when commanding a setpoint.

        The running value is the operative one — it is the bias present while
        the unit is deciding whether it has reached its setpoint. Falls back to
        the idle estimate, then to zero, so an unknown offset degrades to
        "trust the device's sensor" rather than to a guess.
        """
        if self.running is not None and self.n_running >= MIN_KNOT_SAMPLES:
            return round(self.running, 2)
        if self.idle is not None and self.n_idle >= MIN_KNOT_SAMPLES:
            return round(self.idle, 2)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "idle": self.idle,
            "n_running": self.n_running,
            "n_idle": self.n_idle,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HeadOffset:
        obj = cls()
        obj.running = data.get("running")
        obj.idle = data.get("idle")
        obj.n_running = int(data.get("n_running", 0))
        obj.n_idle = int(data.get("n_idle", 0))
        return obj


class GapResponseManager:
    """Per-device, per-mode gap response curves and head offsets."""

    def __init__(self) -> None:
        self._curves: dict[str, GapResponse] = {}
        self._offsets: dict[str, HeadOffset] = {}

    @staticmethod
    def _key(entity_id: str, mode: str) -> str:
        return f"{entity_id}|{mode}"

    def curve(self, entity_id: str, mode: str) -> GapResponse:
        key = self._key(entity_id, mode)
        if key not in self._curves:
            self._curves[key] = GapResponse()
        return self._curves[key]

    def offset(self, entity_id: str) -> HeadOffset:
        if entity_id not in self._offsets:
            self._offsets[entity_id] = HeadOffset()
        return self._offsets[entity_id]

    def observe_offset(self, entity_id: str, head_temp: float, room_temp: float, *, is_running: bool) -> None:
        if self.offset(entity_id).observe(head_temp, room_temp, is_running=is_running):
            _LOGGER.debug(
                "gap-response: %s offset observation head=%.2f room=%.2f delta=%.2f running=%s",
                entity_id,
                head_temp,
                room_temp,
                head_temp - room_temp,
                is_running,
            )

    def observe_response(
        self,
        entity_id: str,
        mode: str,
        *,
        gap: float,
        observed_temp_change: float,
        predicted_passive_change: float,
        dt_minutes: float,
    ) -> None:
        """Record one (gap → incremental rate) sample.

        *observed_temp_change* and *predicted_passive_change* are both in °C over
        *dt_minutes*; their difference is the work the unit did, which is what
        the curve is a function of.
        """
        if dt_minutes < MIN_OBSERVATION_DT:
            return
        incremental = predicted_passive_change - observed_temp_change
        if mode != "cooling":
            incremental = -incremental
        rate = incremental * 60.0 / dt_minutes
        curve = self.curve(entity_id, mode)
        accepted = curve.observe(gap, rate)
        _LOGGER.debug(
            "gap-response: %s/%s gap=%.2fK observed=%+.3f°C passive=%+.3f°C dt=%.1fmin "
            "→ rate=%.2f°C/h accepted=%s n=%d identified=%d/%d",
            entity_id,
            mode,
            gap,
            observed_temp_change,
            predicted_passive_change,
            dt_minutes,
            rate,
            accepted,
            curve.n_observations,
            curve.identified_knots,
            len(curve.knots),
        )

    def to_dict(self) -> dict:
        return {
            "curves": {k: v.to_dict() for k, v in self._curves.items()},
            "offsets": {k: v.to_dict() for k, v in self._offsets.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> GapResponseManager:
        obj = cls()
        for k, v in (data.get("curves") or {}).items():
            obj._curves[k] = GapResponse.from_dict(v)
        for k, v in (data.get("offsets") or {}).items():
            obj._offsets[k] = HeadOffset.from_dict(v)
        return obj
