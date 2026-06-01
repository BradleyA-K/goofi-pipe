"""
SyncAnalyzer — Groovy pSync Node 3 (simplified)
================================================
Real-time sensorimotor synchronisation analysis node.

Design
------
SYNC phase (metronome audible):
  Per participant:
    • mean ITI vs metronome ITI            → tolerance check
    • Rayleigh test on tap phase           → relative to metronome beat
  Pairwise (all unique pairs):
    • |ITI_i − ITI_j|                      → tolerance check

CONTINUATION phase (metronome silent, tempo held in background):
  Per participant:
    • mean ITI vs held metronome ITI       → tolerance check
    • NO phase calculation                 → Rayleigh columns = NaN
  Pairwise (all unique pairs):
    • |ITI_i − ITI_j|                      → tolerance check

Scientific basis
----------------
Tap-to-stimulus asynchrony
  Repp (2005) Processes underlying human sensorimotor synchronization.
  Neuroscience & Biobehavioural Reviews 29(6):872-885.
  → 100 ms window; negative async = anticipatory (common in trained tappers).

Phase coherence via Rayleigh test
  Fisher (1993) Statistical Analysis of Circular Data. Cambridge UP.
  Zar (1999) Biostatistical Analysis 4th ed. Prentice Hall.
  → Mean resultant vector length R ∈ [0,1]; Z = N·R²; p-value via
    Zar (1999) approximation. Window matches offline R pipeline:
    binWidthSec = 2.5 s, minNEventsCirc = 3 (pilot-4-diag-plots.R).

Tempo coherence
  Pecenka & Keller (2011) The role of temporal prediction abilities in
  interpersonal sensorimotor synchronization. Exp Brain Res 211:505-515.
  → ITI matching across participants and against held metronome tempo.

Inputs
------
tap  : TABLE   from MidiIn               — triggers process()
beat : TABLE   from MetronomeGenerator   — non-triggering

Outputs
-------
sync_result : TABLE

  Identity (per tap):
    tap_time          wall-clock seconds since epoch
    participant       auto-label "P1", "P2", ... (string)
    port_name         raw MIDI port name (string)
    note              MIDI note number
    velocity          MIDI velocity
    phase             "synchronization"|"continuation"|"stopped" (string)
    bpm               trial BPM (held during continuation)
    elapsed           seconds since trial start

  Per-tap async (SYNC phase only — NaN in continuation):
    async_ms          signed asynchrony ms: tap_time − nearest_beat_time
    within_window     1.0 if |async_ms| ≤ threshold_ms else 0.0
    beat_index        nearest beat index
    nearest_beat_t    time of nearest beat

  Per-participant windowed tempo (BOTH phases):
    mean_iti_ms              mean ITI over window
    tempo_diff_vs_metro_ms   |mean_iti − metro_iti|  (metro_iti held in continuation)
    tempo_sync_vs_metro      1.0 if tempo_diff ≤ tempo_threshold_ms
    n_taps_window            number of taps in window

  Per-participant windowed phase (SYNC phase only — NaN in continuation):
    rayleigh_R        mean resultant vector length ∈ [0,1]
    rayleigh_Z        Rayleigh Z statistic = N·R²
    rayleigh_p        p-value (Zar 1999 approx)
    rayleigh_sig      1.0 if p < alpha else 0.0
    mean_phase_deg    mean phase direction in degrees
    mean_async_ms     mean signed async over window
    sd_async_ms       SD of async over window

  Dyadic (per unique pair, BOTH phases):
    <Pi>_<Pj>_iti_diff_ms    |mean_ITI_i − mean_ITI_j|
    <Pi>_<Pj>_tempo_sync     1.0 if iti_diff ≤ tempo_threshold_ms

Parameters
----------
sync / threshold_ms          Repp (2005) acceptance window, default 100 ms.
sync / window_sec            Sliding window, default 2.5 s (matches R pipeline).
sync / min_n_events          Min taps in window for Rayleigh, default 3.
sync / alpha                 Significance threshold for Rayleigh, default 0.05.
sync / tempo_threshold_ms    Max |ITI diff| for tempo sync, default 50 ms.
sync / beat_buffer_size      Beat ring buffer size, default 32.
sync / timeout_sec           Fallback stop if no beat for this long, default 5 s.
"""

import itertools
import math
import time
from collections import defaultdict, deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node, InputSlot
from goofi.params import FloatParam, IntParam


# ═══════════════════════════════════════════════════════════════════════
# Circular statistics — self-contained
# ═══════════════════════════════════════════════════════════════════════

def _phases_to_angles(async_ms_list, beat_interval_ms: float) -> np.ndarray:
    """
    Convert asynchrony values (ms) → phase angles (radians).
    Phase = (async mod beat_interval) / beat_interval * 2π.
    """
    a = np.array(async_ms_list, dtype=float)
    wrapped = np.mod(a, beat_interval_ms)
    return wrapped / beat_interval_ms * 2.0 * np.pi


def _rayleigh(angles: np.ndarray):
    """
    Rayleigh test of uniformity for circular data.
    Returns (R, Z, p, mu_deg).
    """
    n = len(angles)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan

    C = float(np.mean(np.cos(angles)))
    S = float(np.mean(np.sin(angles)))
    R = math.sqrt(C ** 2 + S ** 2)
    Z = n * R ** 2

    # Zar (1999) p-value approximation
    inner = 1.0 + 4.0 * n + 4.0 * (n ** 2 - Z ** 2)
    if inner >= 0:
        p = math.exp(math.sqrt(inner) - (1.0 + 2.0 * n))
    else:
        p = 1.0
    p = max(0.0, min(1.0, p))

    mu_deg = math.degrees(math.atan2(S, C)) % 360.0
    return R, Z, p, mu_deg


# ═══════════════════════════════════════════════════════════════════════
# Node
# ═══════════════════════════════════════════════════════════════════════

class SyncAnalyzer(Node):
    """
    Simplified sensorimotor synchrony analyser.

      • Per-participant Rayleigh phase test → only during synchronisation.
      • Tempo tolerance (per participant vs metronome, and pairwise) →
        both phases. Metronome tempo is held during continuation.

    Participants register automatically on first tap.
    All dyadic pairs auto-generated from active participants.
    """

    NO_MULTIPROCESSING = True

    # ── goofi-pipe interface ───────────────────────────────────────────

    @staticmethod
    def config_input_slots():
        return {
            "tap":  DataType.TABLE,
            "beat": InputSlot(DataType.TABLE, trigger_process=False),
        }

    @staticmethod
    def config_output_slots():
        return {"sync_result": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "sync": {
                "threshold_ms": FloatParam(
                    100.0, 1.0, 500.0,
                    doc="Repp (2005) synchrony window in ms. "
                        "|async_ms| ≤ threshold → within_window = 1.",
                ),
                "window_sec": FloatParam(
                    2.5, 0.5, 60.0,
                    doc="Sliding window duration (s) for Rayleigh and rolling "
                        "stats. Default 2.5 s matches binWidthSec in R pipeline.",
                ),
                "min_n_events": IntParam(
                    3, 2, 32,
                    doc="Minimum taps in window before computing Rayleigh / ITI.",
                ),
                "alpha": FloatParam(
                    0.05, 0.001, 0.10,
                    doc="Significance threshold for Rayleigh p-value.",
                ),
                "tempo_threshold_ms": FloatParam(
                    50.0, 1.0, 500.0,
                    doc="Max |ITI diff| in ms to count as tempo-synchronised "
                        "(both vs metronome and pairwise).",
                ),
                "beat_buffer_size": IntParam(
                    32, 4, 256,
                    doc="Max recent beats kept for nearest-beat lookup.",
                ),
                "timeout_sec": FloatParam(
                    5.0, 0.0, 60.0,
                    doc="Safety fallback: stop emitting if no beat arrives for "
                        "this many seconds. 0 = disabled.",
                ),
            }
        }

    # ── lifecycle ─────────────────────────────────────────────────────

    def setup(self):
        self._beats: deque = deque(maxlen=256)
        self._port_to_label: dict[str, str] = {}
        self._n_participants: int = 0
        self._tap_windows: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=512)
        )
        self._trial_start: float | None = None
        self._current_phase: str = "stopped"
        self._last_beat_wall: float | None = None
        # Held metronome interval — last seen beat_interval, used in continuation
        self._held_beat_interval_ms: float = float("nan")
        self._held_bpm: float = float("nan")

    # ── participant registry ───────────────────────────────────────────

    def _get_label(self, participant: str) -> str:
        if participant not in self._port_to_label:
            self._n_participants += 1
            if participant.startswith("P") and participant[1:].isdigit():
                label = participant
            else:
                label = f"P{self._n_participants}"
            self._port_to_label[participant] = label
            print(f"[SyncAnalyzer] New participant: {label}  ← '{participant}'")
        return self._port_to_label[participant]

    def _all_pairs(self) -> list[tuple[str, str]]:
        labels = sorted(self._port_to_label.values(),
                        key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
        return list(itertools.combinations(labels, 2))

    # ── beat buffer ────────────────────────────────────────────────────

    def _ingest_beat(self, beat_table) -> None:
        if beat_table is None:
            return
        try:
            bt       = float(beat_table.data["beat_time"].data[0])
            bi       = int(beat_table.data["beat_index"].data[0])
            bpm      = float(beat_table.data["bpm"].data[0])
            interval = float(beat_table.data["beat_interval"].data[0])
            phase    = str(beat_table.data["phase"].data)
            elapsed  = float(beat_table.data["elapsed"].data[0])

            self._beats.append({
                "beat_time":     bt,
                "beat_index":    bi,
                "bpm":           bpm,
                "beat_interval": interval,
                "phase":         phase,
                "elapsed":       elapsed,
            })
            self._last_beat_wall = time.time()

            # Hold tempo in background — updated every beat,
            # so it carries the last known good value into continuation.
            if interval > 0:
                self._held_beat_interval_ms = interval * 1000.0
                self._held_bpm = bpm

            # Detect trial restart: stopped → synchronization = new trial
            if (self._current_phase == "stopped"
                    and phase == "synchronization"):
                self._reset_trial_state()

            self._current_phase = phase

            if self._trial_start is None:
                self._trial_start = bt - elapsed
        except Exception:
            pass

    def _reset_trial_state(self) -> None:
        self._beats.clear()
        self._tap_windows.clear()
        self._trial_start = None
        self._held_beat_interval_ms = float("nan")
        self._held_bpm = float("nan")
        print("[SyncAnalyzer] ── New trial detected: state reset ──")

    def _nearest_beat(self, tap_time: float):
        """Return (beat_dict, async_ms). NaN async if no beats."""
        if not self._beats:
            return None, float("nan")
        beats = list(self._beats)
        diffs = [tap_time - b["beat_time"] for b in beats]
        idx = int(np.argmin(np.abs(diffs)))
        return beats[idx], diffs[idx] * 1000.0

    # ── windowed helpers ───────────────────────────────────────────────

    def _window_taps(self, label: str, now: float) -> list[dict]:
        w = self.params.sync.window_sec.value
        return [t for t in self._tap_windows[label]
                if now - t["tap_time"] <= w]

    def _mean_iti_ms(self, label: str, now: float) -> float:
        """Mean ITI in ms over the current window. NaN if < 2 taps."""
        taps = self._window_taps(label, now)
        if len(taps) < 2:
            return float("nan")
        times = [t["tap_time"] for t in taps]
        itis = [(times[i + 1] - times[i]) * 1000.0
                for i in range(len(times) - 1)]
        return float(np.mean(itis))

    def _rayleigh_for_label(self, label: str, now: float,
                             beat_interval_ms: float):
        """Per-participant Rayleigh over current window."""
        taps = self._window_taps(label, now)
        n_min = self.params.sync.min_n_events.value

        nan = float("nan")
        if len(taps) < n_min or beat_interval_ms <= 0:
            return nan, nan, nan, nan, nan, nan, len(taps)

        async_vals = [t["async_ms"] for t in taps
                      if not math.isnan(t["async_ms"])]
        if len(async_vals) < n_min:
            return nan, nan, nan, nan, nan, nan, len(taps)

        angles = _phases_to_angles(async_vals, beat_interval_ms)
        R, Z, p, mu_deg = _rayleigh(angles)

        mean_async = float(np.mean(async_vals))
        sd_async = (float(np.std(async_vals, ddof=1))
                    if len(async_vals) > 1 else 0.0)

        return R, Z, p, mu_deg, mean_async, sd_async, len(async_vals)

    # ── TABLE builder ──────────────────────────────────────────────────

    def _arr(self, v) -> Data:
        return Data(DataType.ARRAY, np.array([float(v)]), {})

    def _str(self, v) -> Data:
        return Data(DataType.STRING, str(v), {})

    def _build_table(self, tap_time, label, port_name, note, velocity,
                     beat_entry, async_ms, within_window) -> dict:

        alpha   = self.params.sync.alpha.value
        t_thr   = self.params.sync.tempo_threshold_ms.value
        now     = tap_time
        nan     = float("nan")

        # ── Phase + tempo reference ──────────────────────────────────
        if beat_entry is not None:
            phase            = beat_entry["phase"]
            bpm              = beat_entry["bpm"]
            beat_interval_ms = beat_entry["beat_interval"] * 1000.0
            beat_index       = beat_entry["beat_index"]
            nearest_bt       = beat_entry["beat_time"]
            elapsed          = beat_entry.get("elapsed", nan)
        else:
            phase            = self._current_phase
            bpm              = self._held_bpm
            beat_interval_ms = self._held_beat_interval_ms
            beat_index       = -1
            nearest_bt       = nan
            elapsed          = nan

        # Reference interval for tempo comparison — always the held
        # metronome tempo. Same value in both phases.
        ref_iti_ms = (self._held_beat_interval_ms
                      if not math.isnan(self._held_beat_interval_ms)
                      else beat_interval_ms)

        is_sync = (phase == "synchronization")

        # ── Per-participant tempo (both phases) ──────────────────────
        mean_iti = self._mean_iti_ms(label, now)
        if not math.isnan(mean_iti) and not math.isnan(ref_iti_ms):
            tempo_diff_metro = abs(mean_iti - ref_iti_ms)
            tempo_sync_metro = 1.0 if tempo_diff_metro <= t_thr else 0.0
        else:
            tempo_diff_metro = nan
            tempo_sync_metro = nan

        n_win = len(self._window_taps(label, now))

        # ── Per-participant Rayleigh (sync phase only) ───────────────
        if is_sync:
            R, Z, p, mu_deg, mean_async, sd_async, _ = \
                self._rayleigh_for_label(label, now, beat_interval_ms)
            sig = 1.0 if (not math.isnan(p) and p < alpha) else 0.0
        else:
            R = Z = p = mu_deg = mean_async = sd_async = nan
            sig = nan

        # ── Per-tap async — only meaningful in sync phase ────────────
        if is_sync:
            out_async_ms      = async_ms
            out_within_window = within_window
            out_beat_index    = float(beat_index)
            out_nearest_bt    = nearest_bt
        else:
            out_async_ms      = nan
            out_within_window = nan
            out_beat_index    = nan
            out_nearest_bt    = nan

        table = {
            # identity
            "tap_time":       self._arr(tap_time),
            "participant":    self._str(label),
            "port_name":      self._str(port_name),
            "note":           self._arr(note),
            "velocity":       self._arr(velocity),
            "phase":          self._str(phase),
            "bpm":            self._arr(bpm),
            "elapsed":        self._arr(elapsed),

            # per-tap async (sync only)
            "async_ms":       self._arr(out_async_ms),
            "within_window":  self._arr(out_within_window),
            "beat_index":     self._arr(out_beat_index),
            "nearest_beat_t": self._arr(out_nearest_bt),

            # per-participant tempo (both phases)
            "mean_iti_ms":            self._arr(mean_iti),
            "tempo_diff_vs_metro_ms": self._arr(tempo_diff_metro),
            "tempo_sync_vs_metro":    self._arr(tempo_sync_metro),
            "n_taps_window":          self._arr(float(n_win)),

            # per-participant Rayleigh (sync only)
            "rayleigh_R":     self._arr(R),
            "rayleigh_Z":     self._arr(Z),
            "rayleigh_p":     self._arr(p),
            "rayleigh_sig":   self._arr(sig),
            "mean_phase_deg": self._arr(mu_deg),
            "mean_async_ms":  self._arr(mean_async),
            "sd_async_ms":    self._arr(sd_async),
        }

        # ── Dyadic pairwise tempo (both phases) ──────────────────────
        # The big for loop: every unique pair, |ITI_i − ITI_j| < threshold.
        for (pi, pj) in self._all_pairs():
            key = f"{pi}_{pj}"
            iti_i = self._mean_iti_ms(pi, now)
            iti_j = self._mean_iti_ms(pj, now)

            if math.isnan(iti_i) or math.isnan(iti_j):
                iti_diff = nan
                pair_sync = nan
            else:
                iti_diff = abs(iti_i - iti_j)
                pair_sync = 1.0 if iti_diff <= t_thr else 0.0

            table[f"{key}_iti_diff_ms"] = self._arr(iti_diff)
            table[f"{key}_tempo_sync"]  = self._arr(pair_sync)

        return table

    # ── process() ─────────────────────────────────────────────────────

    def process(self, tap, beat):
        # Passively ingest beat stream — also updates _current_phase
        # and the held metronome interval.
        if beat is not None:
            self._ingest_beat(beat)

        if tap is None:
            return None

        # Phase gate — only emit during active trial
        if self._current_phase == "stopped":
            return None

        # Timeout fallback
        timeout = self.params.sync.timeout_sec.value
        if timeout > 0 and self._last_beat_wall is not None:
            if time.time() - self._last_beat_wall > timeout:
                return None

        # Extract tap fields
        try:
            tap_time    = float(tap.data["tap_time"].data[0])
            port_name   = str(tap.data["port_name"].data)
            note        = float(tap.data["note"].data[0])
            velocity    = float(tap.data["velocity"].data[0])
            participant = str(tap.data["participant"].data)
        except Exception:
            return None

        label = self._get_label(participant)

        # Nearest beat + asynchrony (used for sync-phase Rayleigh)
        threshold_ms = self.params.sync.threshold_ms.value
        beat_entry, async_ms = self._nearest_beat(tap_time)
        within_window = (1.0 if not math.isnan(async_ms)
                         and abs(async_ms) <= threshold_ms else 0.0)
        beat_index = beat_entry["beat_index"] if beat_entry else -1

        # Push into sliding window
        self._tap_windows[label].append({
            "tap_time":   tap_time,
            "async_ms":   async_ms,
            "beat_index": beat_index,
        })

        # Console feedback
        ph = beat_entry["phase"] if beat_entry else self._current_phase
        sync_ch = "✓" if within_window == 1.0 else "✗"
        if ph == "synchronization":
            print(f"[SyncAnalyzer] {label:3s} | sync | "
                  f"async={async_ms:+7.1f}ms {sync_ch}")
        else:
            print(f"[SyncAnalyzer] {label:3s} | {ph[:4]:4s} | tap")

        table = self._build_table(
            tap_time, label, port_name, note, velocity,
            beat_entry, async_ms, within_window,
        )

        meta = {
            "participant":   label,
            "phase":         ph,
            "async_ms":      async_ms,
            "within_window": within_window,
        }

        return {"sync_result": (table, meta)}
"""
SyncAnalyzer — Groovy pSync Node 3
=====================================
Real-time sensorimotor synchronisation analysis node.

Scientific basis
----------------
Tap-to-stimulus asynchrony
  Repp (2005) Processes underlying human sensorimotor synchronization.
  Neuroscience & Biobehavioural Reviews 29(6):872-885.
  → 100 ms window; negative async = anticipatory (common in trained tappers).

Phase coherence via Rayleigh test
  Fisher (1993) Statistical Analysis of Circular Data. Cambridge UP.
  Zar (1999) Biostatistical Analysis 4th ed. Prentice Hall.
  → Mean resultant vector length R ∈ [0,1]; Z = N·R²; p-value via
    Zar (1999) approximation.  Window matches offline R pipeline:
    binWidthSec = 2.5 s, minNEventsCirc = 3  (pilot-4-diag-plots.R).

Tempo coherence (dyadic)
  Pecenka & Keller (2011) The role of temporal prediction abilities in
  interpersonal sensorimotor synchronization. Exp Brain Res 211:505-515.
  → ITI matching across participants; rolling window in seconds.

Dyadic phase coherence
  Nozaradan et al. (2012) Tagging the neuronal entrainment to beat and
  meter. J Neurosci 32(26):10024-10036.
  → Phase of each participant's tap relative to beat cycle; Rayleigh on
    the distribution of *phase differences* between pairs.

Architecture
------------
- N participants, auto-registered on first tap from each MIDI port.
- All dyadic pairs auto-generated via itertools.combinations.
- Window-based statistics: taps collected in a deque per participant,
  kept for `window_sec` seconds; statistics recomputed on every tap.
- Fully stateless between process() calls — all state in setup().

Inputs
------
tap  : TABLE   from MidiIn          — triggers process()
beat : TABLE   from MetronomeGenerator — non-triggering

Outputs
-------
sync_result : TABLE
  Per-tap identity:
    tap_time          wall-clock seconds since epoch
    participant       auto-label "P1", "P2", ... (string)
    port_name         raw MIDI port name (string)
    note              MIDI note number (float[1])
    velocity          MIDI velocity (float[1])
    phase             "synchronization"|"continuation"|"stopped" (string)
    bpm               trial BPM (float[1])
    elapsed           seconds since trial start (float[1])

  Tap-to-stimulus (per tap):
    async_ms          signed asynchrony ms: tap_time − nearest_beat_time
    within_window     1.0 if |async_ms| ≤ threshold_ms else 0.0
    beat_index        nearest beat index (float[1])
    nearest_beat_t    time of nearest beat (float[1])

  Tap-to-stimulus (windowed, recomputed each tap):
    rayleigh_R        mean resultant vector length ∈ [0,1] (float[1])
    rayleigh_Z        Rayleigh Z statistic = N·R² (float[1])
    rayleigh_p        p-value (Zar 1999 approx) (float[1])
    rayleigh_sig      1.0 if p < alpha else 0.0 (float[1])
    mean_phase_deg    mean phase direction in degrees (float[1])
    mean_async_ms     mean signed async over window (float[1])
    sd_async_ms       SD of async over window (float[1])
    n_taps_window     number of taps used for window stats (float[1])

  Dyadic (one set of columns per pair, e.g. "P1_P2_..."):
    <Pi>_<Pj>_async_ms        tap-time difference on same beat (ms)
    <Pi>_<Pj>_phase_R         Rayleigh R on phase *differences* in window
    <Pi>_<Pj>_phase_p         Rayleigh p-value for phase differences
    <Pi>_<Pj>_phase_sig       1.0 if phase_p < alpha
    <Pi>_<Pj>_tempo_iti_diff  |ITI_i − ITI_j| in ms (tempo coherence)
    <Pi>_<Pj>_sync            1.0 if both phase_sig AND tempo within threshold

Parameters
----------
sync / threshold_ms
    Repp (2005) acceptance window (default 100 ms).
sync / window_sec
    Sliding window duration in seconds for Rayleigh and rolling stats.
    Default 2.5 s — matches binWidthSec in pilot-4-diag-plots.R.
sync / min_n_events
    Minimum taps in window before computing Rayleigh (default 3).
    Matches minNEventsCirc in pilot-4-diag-plots.R.
sync / alpha
    Significance threshold for Rayleigh test (default 0.05).
sync / beat_buffer_size
    Max recent beats to keep for nearest-beat lookup (default 32).
sync / dyadic_mode
    "phase_coherence"  — Rayleigh on phase differences between pairs.
    "tempo_coherence"  — |ITI difference| in ms.
    "both"             — compute and output both.
sync / tempo_threshold_ms
    |ITI diff| below which a dyadic pair is flagged as tempo-synchronised
    (default 50 ms). Used when dyadic_mode is "tempo_coherence" or "both".
sync / timeout_sec
    Safety fallback: if no beat arrives for this many seconds, SyncAnalyzer
    stops emitting. 0 = disabled. Default 5.0 s.
    Primary gate is always the MetronomeGenerator phase field.
"""

import itertools
import math
import time
from collections import defaultdict, deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node, InputSlot
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


# ═══════════════════════════════════════════════════════════════════════
# Circular statistics — self-contained, no scipy dependency
# ═══════════════════════════════════════════════════════════════════════

def _phases_to_angles(async_ms_list: list, beat_interval_ms: float) -> np.ndarray:
    """
    Convert a list of asynchrony values (ms) to phase angles (radians).
    Phase = (async mod beat_interval) / beat_interval * 2π
    Matches asyncVsMetPct/100*360 in pilot-4-diag-plots.R.
    """
    a = np.array(async_ms_list, dtype=float)
    # Wrap into [0, beat_interval)
    wrapped = np.mod(a, beat_interval_ms)
    return wrapped / beat_interval_ms * 2.0 * np.pi


def _rayleigh(angles: np.ndarray):
    """
    Rayleigh test of uniformity for circular data.

    Returns
    -------
    R     : mean resultant vector length ∈ [0, 1]
    Z     : test statistic = N * R²
    p     : p-value (Zar 1999 approximation)
    mu_deg: mean direction in degrees
    """
    n = len(angles)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = math.sqrt(C ** 2 + S ** 2)
    Z = n * R ** 2

    # Zar (1999) p-value approximation — valid for n >= 3
    # p ≈ exp(sqrt(1 + 4n + 4(n²-Z²)) - (1+2n))
    inner = 1.0 + 4.0 * n + 4.0 * (n ** 2 - Z ** 2)
    if inner >= 0:
        p = math.exp(math.sqrt(inner) - (1.0 + 2.0 * n))
    else:
        p = 1.0  # fallback if numerical issues

    # Clamp p to [0, 1]
    p = max(0.0, min(1.0, p))

    mu_rad = math.atan2(S, C)
    mu_deg = math.degrees(mu_rad) % 360.0

    return R, Z, p, mu_deg


def _rayleigh_phase_diff(angles_i: np.ndarray, angles_j: np.ndarray):
    """
    Rayleigh test on the distribution of phase *differences* between
    two participants.  Aligns arrays by length (use the shorter).
    Nozaradan et al. (2012) approach for dyadic phase coherence.
    """
    n = min(len(angles_i), len(angles_j))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs = angles_i[-n:] - angles_j[-n:]
    return _rayleigh(diffs)


# ═══════════════════════════════════════════════════════════════════════
# Node
# ═══════════════════════════════════════════════════════════════════════

class SyncAnalyzer(Node):
    """
    Window-based sensorimotor synchrony analyser for Groovy pSync.

    Participants register automatically on first tap — no config needed.
    All dyadic pairs auto-generated from active participants.
    Window size defaults to 2.5 s, matching the offline R analysis pipeline.
    """

    NO_MULTIPROCESSING = True

    # ── goofi-pipe interface ───────────────────────────────────────────

    @staticmethod
    def config_input_slots():
        return {
            "tap":  DataType.TABLE,
            # beat does NOT trigger process() — consumed passively
            "beat": InputSlot(DataType.TABLE, trigger_process=False),
        }

    @staticmethod
    def config_output_slots():
        return {"sync_result": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "sync": {
                "threshold_ms": FloatParam(
                    100.0, 1.0, 500.0,
                    doc="Repp (2005) synchrony window in ms. "
                        "|async_ms| ≤ threshold → within_window = 1.",
                ),
                "window_sec": FloatParam(
                    2.5, 0.5, 60.0,
                    doc="Sliding window duration (s) for Rayleigh and rolling stats. "
                        "Default 2.5 s matches binWidthSec in pilot-4-diag-plots.R.",
                ),
                "min_n_events": IntParam(
                    3, 2, 32,
                    doc="Minimum taps in window before computing Rayleigh. "
                        "Default 3 matches minNEventsCirc in pilot-4-diag-plots.R.",
                ),
                "alpha": FloatParam(
                    0.05, 0.001, 0.10,
                    doc="Significance threshold for Rayleigh p-value (default 0.05).",
                ),
                "beat_buffer_size": IntParam(
                    32, 4, 256,
                    doc="Max recent beats to keep for nearest-beat lookup.",
                ),
                "dyadic_mode": StringParam(
                    "both",
                    options=["phase_coherence", "tempo_coherence", "both"],
                    doc=(
                        "phase_coherence : Rayleigh on phase differences (Nozaradan 2012). "
                        "Easier — rewards tapping at same beat phase.\n"
                        "tempo_coherence : |ITI difference| in ms (Pecenka & Keller 2011). "
                        "Harder — requires matching internal tempo.\n"
                        "both            : compute and output both metrics."
                    ),
                ),
                "tempo_threshold_ms": FloatParam(
                    50.0, 1.0, 500.0,
                    doc="Max |ITI diff| in ms for tempo-coherence sync flag "
                        "(Pecenka & Keller 2011). Default 50 ms.",
                ),
                "timeout_sec": FloatParam(
                    5.0, 0.0, 60.0,
                    doc="Safety fallback: stop emitting if no beat arrives for "
                        "this many seconds. 0 = disabled. Primary gate is always "
                        "the MetronomeGenerator phase field.",
                ),
            }
        }

    # ── lifecycle ─────────────────────────────────────────────────────

    def setup(self):
        # Beat ring buffer: list of dicts with beat metadata
        self._beats: deque = deque(maxlen=256)

        # Participant registry: port_name → "P1", "P2", ...
        self._port_to_label: dict[str, str] = {}
        self._n_participants: int = 0

        # Per-participant sliding window of tap records
        # Each record: {"tap_time": float, "async_ms": float, "beat_index": int}
        self._tap_windows: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=512)
        )

        # Per-participant last tap time for ITI computation
        self._last_tap_time: dict[str, float] = {}

        # Trial start time (set on first beat received)
        self._trial_start: float | None = None

        # Phase gate — track MetronomeGenerator phase
        # "stopped" or None → do not emit; anything else → emit
        self._current_phase: str = "stopped"
        self._last_beat_wall: float | None = None  # for timeout fallback

    # ── participant registry ───────────────────────────────────────────

    def _get_label(self, participant: str) -> str:
        """
        Register and return a stable label for this participant identifier.
        If MidiIn resolved a label via participant_map, that label is used
        directly.  If not (blank map), participant is the raw port_name and
        we auto-number: P1, P2, P3...
        """
        if participant not in self._port_to_label:
            self._n_participants += 1
            # If participant already looks like a label (e.g. "P1" from map),
            # use it as-is.  Otherwise auto-assign a number.
            if participant.startswith("P") and participant[1:].isdigit():
                label = participant
            else:
                label = f"P{self._n_participants}"
            self._port_to_label[participant] = label
            print(f"[SyncAnalyzer] New participant: {label}  ← '{participant}'")
        return self._port_to_label[participant]

    def _all_pairs(self) -> list[tuple[str, str]]:
        labels = sorted(self._port_to_label.values(),
                        key=lambda s: int(s[1:]))
        return list(itertools.combinations(labels, 2))

    # ── beat buffer ────────────────────────────────────────────────────

    def _ingest_beat(self, beat_table) -> None:
        if beat_table is None:
            return
        try:
            bt       = float(beat_table.data["beat_time"].data[0])
            bi       = int(beat_table.data["beat_index"].data[0])
            bpm      = float(beat_table.data["bpm"].data[0])
            interval = float(beat_table.data["beat_interval"].data[0])
            phase    = str(beat_table.data["phase"].data)
            elapsed  = float(beat_table.data["elapsed"].data[0])
            self._beats.append({
                "beat_time":     bt,
                "beat_index":    bi,
                "bpm":           bpm,
                "beat_interval": interval,
                "phase":         phase,
                "elapsed":       elapsed,
            })
            self._last_beat_wall = time.time()

            # Detect trial restart: stopped → synchronization means new trial
            if (self._current_phase == "stopped"
                    and phase == "synchronization"):
                self._reset_trial_state()

            self._current_phase = phase

            if self._trial_start is None:
                self._trial_start = bt - elapsed
        except Exception:
            pass

    def _reset_trial_state(self) -> None:
        """Clear all per-trial state ready for a fresh trial."""
        self._beats.clear()
        self._tap_windows.clear()
        self._last_tap_time.clear()
        self._trial_start = None
        # Keep participant registry — ports don't change between trials
        print("[SyncAnalyzer] ── New trial detected: state reset ──")

    def _nearest_beat(self, tap_time: float):
        """Return (beat_dict, async_ms) for the nearest beat, or (None, nan)."""
        if not self._beats:
            return None, float("nan")
        beats = list(self._beats)
        diffs = [tap_time - b["beat_time"] for b in beats]
        idx   = int(np.argmin(np.abs(diffs)))
        return beats[idx], diffs[idx] * 1000.0

    # ── sliding window helpers ─────────────────────────────────────────

    def _window_taps(self, label: str, now: float) -> list[dict]:
        """Return taps within window_sec of now."""
        w = self.params.sync.window_sec.value
        return [t for t in self._tap_windows[label]
                if now - t["tap_time"] <= w]

    def _rayleigh_for_label(self, label: str, now: float,
                             beat_interval_ms: float):
        """Compute Rayleigh stats for one participant's window."""
        taps = self._window_taps(label, now)
        n_min = self.params.sync.min_n_events.value

        if len(taps) < n_min or beat_interval_ms <= 0:
            nan = float("nan")
            return nan, nan, nan, nan, nan, nan, len(taps)

        async_vals = [t["async_ms"] for t in taps
                      if not math.isnan(t["async_ms"])]
        if len(async_vals) < n_min:
            nan = float("nan")
            return nan, nan, nan, nan, nan, nan, len(taps)

        angles = _phases_to_angles(async_vals, beat_interval_ms)
        R, Z, p, mu_deg = _rayleigh(angles)

        mean_async = float(np.mean(async_vals))
        sd_async   = float(np.std(async_vals, ddof=1)) if len(async_vals) > 1 else 0.0

        return R, Z, p, mu_deg, mean_async, sd_async, len(async_vals)

    # ── dyadic helpers ─────────────────────────────────────────────────

    def _dyadic_direct_async(self, pi: str, pj: str,
                              tap_time: float, now: float) -> float:
        """
        Find the closest tap from Pj within the window around tap_time.
        Used for BOTH phases:
          sync phase       — compare taps on the same projected beat
          continuation     — compare taps directly, no beat reference needed

        Returns signed ms: tap_Pi_time - nearest_tap_Pj_time.
        Returns NaN if Pj has no taps in the window.
        """
        w = self.params.sync.window_sec.value
        taps_j = [t for t in self._tap_windows[pj]
                  if abs(t["tap_time"] - tap_time) <= w]
        if not taps_j:
            return float("nan")
        # Nearest tap from Pj to this tap from Pi
        nearest = min(taps_j, key=lambda t: abs(t["tap_time"] - tap_time))
        return (tap_time - nearest["tap_time"]) * 1000.0

    def _mean_iti(self, label: str, now: float) -> float:
        """Mean ITI in ms over the current window. NaN if < 2 taps."""
        taps = self._window_taps(label, now)
        if len(taps) < 2:
            return float("nan")
        times = [t["tap_time"] for t in taps]
        itis  = [(times[i+1] - times[i]) * 1000.0 for i in range(len(times)-1)]
        return float(np.mean(itis))

    def _dyadic_phase_coherence(self, pi: str, pj: str,
                                 now: float, beat_interval_ms: float,
                                 phase: str):
        """
        Rayleigh test on phase differences between Pi and Pj.

        sync phase   : phases computed relative to metronome beat interval
                       (Nozaradan et al. 2012) — beat_interval_ms used.
        continuation : phases computed relative to each participant's own
                       mean ITI — no external reference needed.
                       (Pecenka & Keller 2011 mutual entrainment approach)
        """
        taps_i = self._window_taps(pi, now)
        taps_j = self._window_taps(pj, now)
        n_min  = self.params.sync.min_n_events.value

        if len(taps_i) < n_min or len(taps_j) < n_min:
            nan = float("nan")
            return nan, nan, nan, nan

        if phase == "continuation":
            # Use mean ITI of the pair as shared reference cycle
            iti_i = self._mean_iti(pi, now)
            iti_j = self._mean_iti(pj, now)
            if math.isnan(iti_i) or math.isnan(iti_j):
                nan = float("nan")
                return nan, nan, nan, nan
            # Use average ITI of the pair as the reference cycle
            ref_interval_ms = (iti_i + iti_j) / 2.0
            # Phases relative to tap times directly (not async_ms)
            t0 = taps_i[0]["tap_time"] * 1000.0  # anchor
            ai = [(t["tap_time"] * 1000.0 - t0) % ref_interval_ms
                  for t in taps_i]
            aj = [(t["tap_time"] * 1000.0 - t0) % ref_interval_ms
                  for t in taps_j]
        else:
            # sync phase — use beat-referenced async values
            if beat_interval_ms <= 0:
                nan = float("nan")
                return nan, nan, nan, nan
            ai = [t["async_ms"] for t in taps_i if not math.isnan(t["async_ms"])]
            aj = [t["async_ms"] for t in taps_j if not math.isnan(t["async_ms"])]
            if len(ai) < n_min or len(aj) < n_min:
                nan = float("nan")
                return nan, nan, nan, nan

        ang_i = _phases_to_angles(ai, ref_interval_ms
                                  if phase == "continuation"
                                  else beat_interval_ms)
        ang_j = _phases_to_angles(aj, ref_interval_ms
                                  if phase == "continuation"
                                  else beat_interval_ms)
        return _rayleigh_phase_diff(ang_i, ang_j)

    def _dyadic_tempo_coherence(self, pi: str, pj: str, now: float) -> float:
        """
        |mean_ITI_i − mean_ITI_j| in ms over the current window.
        Works in both phases — ITI is always meaningful.
        Pecenka & Keller (2011).
        """
        iti_i = self._mean_iti(pi, now)
        iti_j = self._mean_iti(pj, now)
        if math.isnan(iti_i) or math.isnan(iti_j):
            return float("nan")
        return abs(iti_i - iti_j)

    # ── TABLE builder ──────────────────────────────────────────────────

    def _arr(self, v) -> Data:
        return Data(DataType.ARRAY, np.array([float(v)]), {})

    def _str(self, v) -> Data:
        return Data(DataType.STRING, str(v), {})

    def _build_table(self, tap_time, label, port_name, note, velocity,
                     beat_entry, async_ms, within_window) -> dict:

        alpha   = self.params.alpha if hasattr(self.params, "alpha") else 0.05
        alpha   = self.params.sync.alpha.value
        mode    = self.params.sync.dyadic_mode.value
        t_thr   = self.params.sync.tempo_threshold_ms.value
        now     = tap_time

        if beat_entry:
            bpm           = beat_entry["bpm"]
            phase         = beat_entry["phase"]
            beat_interval_ms = beat_entry["beat_interval"] * 1000.0
            beat_index    = beat_entry["beat_index"]
            nearest_bt    = beat_entry["beat_time"]
            elapsed       = beat_entry.get("elapsed", float("nan"))
        else:
            bpm = beat_interval_ms = float("nan")
            phase = "unknown"
            beat_index = -1
            nearest_bt = elapsed = float("nan")

        # Per-participant Rayleigh
        R, Z, p, mu_deg, mean_async, sd_async, n_win = self._rayleigh_for_label(
            label, now, beat_interval_ms
        )
        sig = 1.0 if (not math.isnan(p) and p < alpha) else 0.0

        table = {
            # ── identity ─────────────────────────────────────────────
            "tap_time":       self._arr(tap_time),
            "participant":    self._str(label),
            "port_name":      self._str(port_name),
            "note":           self._arr(note),
            "velocity":       self._arr(velocity),
            "phase":          self._str(phase),
            "bpm":            self._arr(bpm),
            "elapsed":        self._arr(elapsed),
            # ── tap-to-stimulus (per tap) ────────────────────────────
            "async_ms":       self._arr(async_ms),
            "within_window":  self._arr(within_window),
            "beat_index":     self._arr(float(beat_index)),
            "nearest_beat_t": self._arr(nearest_bt),
            # ── tap-to-stimulus (windowed Rayleigh) ──────────────────
            "rayleigh_R":     self._arr(R),
            "rayleigh_Z":     self._arr(Z),
            "rayleigh_p":     self._arr(p),
            "rayleigh_sig":   self._arr(sig),
            "mean_phase_deg": self._arr(mu_deg),
            "mean_async_ms":  self._arr(mean_async),
            "sd_async_ms":    self._arr(sd_async),
            "n_taps_window":  self._arr(float(n_win)),
        }

        # ── dyadic pairs ─────────────────────────────────────────────
        # Computed in BOTH phases:
        #   sync phase       : direct async + beat-referenced phase coherence
        #   continuation     : direct async + ITI-referenced phase coherence
        # This ensures dyadic columns are always populated in the CSV
        # regardless of whether the metronome is audible.
        for (pi, pj) in self._all_pairs():
            key = f"{pi}_{pj}"

            # Direct tap-time difference — works in both phases
            # (nearest tap from Pj to the current Pi tap, within window)
            raw_ms = self._dyadic_direct_async(pi, pj, tap_time, now)
            table[f"{key}_async_ms"] = self._arr(raw_ms)

            phase_sig  = float("nan")
            tempo_diff = float("nan")
            both_sync  = float("nan")

            # Phase coherence — beat-referenced in sync, ITI-referenced in continuation
            if mode in ("phase_coherence", "both"):
                dR, dZ, dp, dmu = self._dyadic_phase_coherence(
                    pi, pj, now, beat_interval_ms, phase
                )
                table[f"{key}_phase_R"]   = self._arr(dR)
                table[f"{key}_phase_Z"]   = self._arr(dZ)
                table[f"{key}_phase_p"]   = self._arr(dp)
                ps = 1.0 if (not math.isnan(dp) and dp < alpha) else 0.0
                table[f"{key}_phase_sig"] = self._arr(ps)
                phase_sig = ps

            # Tempo coherence — ITI difference, always meaningful
            if mode in ("tempo_coherence", "both"):
                tempo_diff = self._dyadic_tempo_coherence(pi, pj, now)
                ts = 1.0 if (not math.isnan(tempo_diff)
                             and tempo_diff <= t_thr) else 0.0
                table[f"{key}_tempo_iti_diff"] = self._arr(tempo_diff)
                table[f"{key}_tempo_sig"]      = self._arr(ts)

            # Combined sync flag
            if mode == "phase_coherence":
                both_sync = phase_sig
            elif mode == "tempo_coherence":
                ts = (1.0 if not math.isnan(tempo_diff)
                      and tempo_diff <= t_thr else 0.0)
                both_sync = ts
            else:  # both
                ts = (1.0 if not math.isnan(tempo_diff)
                      and tempo_diff <= t_thr else 0.0)
                if not math.isnan(phase_sig) and not math.isnan(ts):
                    both_sync = 1.0 if (phase_sig == 1.0 and ts == 1.0) else 0.0

            table[f"{key}_sync"] = self._arr(both_sync)

        return table

    # ── process() ─────────────────────────────────────────────────────

    def process(self, tap, beat):
        # Passively ingest beat stream — also updates _current_phase
        if beat is not None:
            self._ingest_beat(beat)

        if tap is None:
            return None

        # ── Phase gate (primary) ──────────────────────────────────────
        # Only emit during active trial phases from MetronomeGenerator
        if self._current_phase == "stopped":
            return None

        # ── Timeout fallback (secondary) ──────────────────────────────
        timeout = self.params.sync.timeout_sec.value
        if timeout > 0 and self._last_beat_wall is not None:
            if time.time() - self._last_beat_wall > timeout:
                return None

        if tap is None:
            return None

        # Extract tap fields
        try:
            tap_time    = float(tap.data["tap_time"].data[0])
            port_name   = str(tap.data["port_name"].data)
            note        = float(tap.data["note"].data[0])
            velocity    = float(tap.data["velocity"].data[0])
            # participant set by MidiIn — from participant_map or port_name fallback
            participant = str(tap.data["participant"].data)
        except Exception:
            return None

        label = self._get_label(participant)

        # Nearest beat + asynchrony
        threshold_ms = self.params.sync.threshold_ms.value
        beat_entry, async_ms = self._nearest_beat(tap_time)
        within_window = (1.0 if not math.isnan(async_ms)
                         and abs(async_ms) <= threshold_ms else 0.0)

        beat_index = beat_entry["beat_index"] if beat_entry else -1

        # Update sliding window
        self._tap_windows[label].append({
            "tap_time":   tap_time,
            "async_ms":   async_ms,
            "beat_index": beat_index,
        })

        # Console feedback
        phase = beat_entry["phase"] if beat_entry else "?"
        sync_ch = "✓" if within_window == 1.0 else "✗"
        print(
            f"[SyncAnalyzer] {label:3s} | {phase[:4]:4s} | "
            f"async={async_ms:+7.1f}ms {sync_ch}"
        )

        table = self._build_table(
            tap_time, label, port_name, note, velocity,
            beat_entry, async_ms, within_window,
        )

        meta = {
            "participant":   label,
            "phase":         phase,
            "async_ms":      async_ms,
            "within_window": within_window,
        }

        return {"sync_result": (table, meta)}
