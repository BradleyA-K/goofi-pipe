"""
SyncAnalyzer — Groovy pSync Node 3  (simplified)
=================================================
Per-participant, windowed sensorimotor-synchrony summary.

This node does NOT decide who is "in sync" — it only reports the
windowed *averages* for each participant.  The grouping decision
(how many participants are synchronised) is made downstream by
GroupSyncScore, which reads these averages.

What it emits (one row per tap, for the tapping participant)
------------------------------------------------------------
  SYNC phase  (metronome audible)
      • async_ms        : signed tap-to-beat asynchrony (ms) for this tap
      • within_window   : 1.0 if |async_ms| <= threshold_ms for this tap
      • metro_sync      : windowed fraction of taps within the beat window
                          ∈ [0,1]  ← per-participant "matches the metronome"
      • mean_iti_ms     : windowed mean inter-tap interval (ms)

  CONTINUATION phase  (metronome silent, tempo held in background)
      • async_ms / within_window / metro_sync = NaN  (no beat to compare to)
      • mean_iti_ms     : windowed mean inter-tap interval (ms)
                          ← GroupSyncScore compares these across participants

Design notes
------------
  • Participants register automatically on first tap (P1, P2, ...).
  • A sliding window of `window_sec` seconds is kept per participant;
    every statistic is recomputed on each tap from that window.
  • No pairwise / dyadic columns, no circular statistics — the group
    metric is computed in GroupSyncScore from the per-participant rows.

Scientific basis
----------------
  Tap-to-stimulus asynchrony window
      Repp (2005) Neuroscience & Biobehavioural Reviews 29(6):872-885.
      → default 100 ms acceptance window.
  Tempo (ITI) matching across participants
      Pecenka & Keller (2011) Exp Brain Res 211:505-515.
      → windowed mean ITI; matched downstream in GroupSyncScore.

Inputs
------
tap  : TABLE   from MidiIn               — triggers process()
beat : TABLE   from MetronomeGenerator   — non-triggering (passive)

Outputs
-------
sync_result : TABLE
    tap_time        wall-clock seconds since epoch (float[1])
    participant     auto-label "P1", "P2", ... (string)
    port_name       raw MIDI port name (string)
    note            MIDI note number (float[1])
    velocity        MIDI velocity (float[1])
    phase           "synchronization"|"continuation"|"stopped" (string)
    bpm             trial BPM, held during continuation (float[1])
    elapsed         seconds since trial start (float[1])
    async_ms        signed tap-to-beat async (ms), NaN in continuation
    within_window   1.0/0.0 for this tap, NaN in continuation
    metro_sync      windowed fraction within beat window, NaN in continuation
    mean_iti_ms     windowed mean inter-tap interval (ms), both phases
    mean_async_ms   windowed mean signed async (ms), NaN in continuation
    n_taps_window   number of taps in the current window (float[1])

Parameters
----------
sync / threshold_ms     Repp (2005) beat window in ms (default 100).
sync / window_sec       Sliding window length in s (default 2.5).
sync / min_n_events     Min taps in window before mean_iti is reported (2).
sync / timeout_sec      Stop emitting if no beat for this long (0 = off).
"""

import time
from collections import defaultdict, deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node, InputSlot
from goofi.params import FloatParam, IntParam


class SyncAnalyzer(Node):
    """
    Simplified per-participant synchrony analyser for Groovy pSync.

    Reports windowed averages per participant; the group-level
    "how many are in sync" decision lives in GroupSyncScore.
    """

    NO_MULTIPROCESSING = True

    # ── goofi-pipe interface ───────────────────────────────────────────

    @staticmethod
    def config_input_slots():
        return {
            "tap":  DataType.TABLE,
            # beat triggers process() too, so the trial's "stopped" signal is
            # noticed immediately even if no tap arrives. Retained-tap re-runs
            # are filtered out in process() via tap_time de-duplication.
            "beat": InputSlot(DataType.TABLE, trigger_process=True),
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
                    doc="Repp (2005) beat window in ms. "
                        "|async_ms| <= threshold counts as on-beat.",
                ),
                "window_sec": FloatParam(
                    2.5, 0.5, 60.0,
                    doc="Sliding window length (s) for all windowed averages.",
                ),
                "min_n_events": IntParam(
                    2, 2, 32,
                    doc="Minimum taps in the window before mean_iti is reported.",
                ),
                "timeout_sec": FloatParam(
                    5.0, 0.0, 60.0,
                    doc="Stop emitting if no beat arrives for this many seconds. "
                        "0 = disabled. Primary gate is the metronome phase field.",
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
        # tap_time of the last tap actually emitted — used to ignore retained
        # taps when process() is triggered by a beat (or by autotrigger).
        self._last_tap_seen: float | None = None
        # Held metronome tempo — carried into the silent continuation phase.
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
            print(f"[SyncAnalyzer] New participant: {label}  <- '{participant}'")
        return self._port_to_label[participant]

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
        except Exception:
            return

        self._beats.append({
            "beat_time":     bt,
            "beat_index":    bi,
            "bpm":           bpm,
            "beat_interval": interval,
            "phase":         phase,
            "elapsed":       elapsed,
        })
        self._last_beat_wall = time.time()

        if interval > 0:
            self._held_beat_interval_ms = interval * 1000.0
            self._held_bpm = bpm

        # Detect a fresh trial: stopped -> synchronization.
        if self._current_phase == "stopped" and phase == "synchronization":
            self._reset_trial_state()

        self._current_phase = phase
        if self._trial_start is None:
            self._trial_start = bt - elapsed

    def _reset_trial_state(self) -> None:
        self._beats.clear()
        self._tap_windows.clear()
        self._trial_start = None
        self._held_beat_interval_ms = float("nan")
        self._held_bpm = float("nan")
        self._last_tap_seen = None
        print("[SyncAnalyzer] -- New trial detected: state reset --")

    def _nearest_beat(self, tap_time: float):
        """Return (beat_dict, async_ms). NaN async if no beats buffered."""
        if not self._beats:
            return None, float("nan")
        beats = list(self._beats)
        diffs = [tap_time - b["beat_time"] for b in beats]
        idx = int(np.argmin(np.abs(diffs)))
        return beats[idx], diffs[idx] * 1000.0

    # ── windowed helpers ───────────────────────────────────────────────

    def _window_taps(self, label: str, now: float) -> list[dict]:
        w = self.params.sync.window_sec.value
        return [t for t in self._tap_windows[label] if now - t["tap_time"] <= w]

    def _mean_iti_ms(self, taps: list[dict]) -> float:
        if len(taps) < self.params.sync.min_n_events.value:
            return float("nan")
        times = [t["tap_time"] for t in taps]
        itis = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
        return float(np.mean(itis)) if itis else float("nan")

    def _metro_sync(self, taps: list[dict]) -> tuple[float, float]:
        """Windowed (fraction within beat window, mean signed async)."""
        threshold = self.params.sync.threshold_ms.value
        asyncs = [t["async_ms"] for t in taps if not np.isnan(t["async_ms"])]
        if not asyncs:
            return float("nan"), float("nan")
        within = [1.0 if abs(a) <= threshold else 0.0 for a in asyncs]
        return float(np.mean(within)), float(np.mean(asyncs))

    # ── TABLE builders ──────────────────────────────────────────────────

    def _arr(self, v) -> Data:
        return Data(DataType.ARRAY, np.array([float(v)]), {})

    def _str(self, v) -> Data:
        return Data(DataType.STRING, str(v), {})

    # ── process() ─────────────────────────────────────────────────────

    def process(self, tap, beat):
        # Passively ingest the beat stream (updates phase + held tempo).
        if beat is not None:
            self._ingest_beat(beat)

        if tap is None:
            return None

        # Only emit during an active trial.
        if self._current_phase == "stopped":
            return None

        # Safety timeout: stop if the metronome stream has gone silent.
        timeout = self.params.sync.timeout_sec.value
        if timeout > 0 and self._last_beat_wall is not None:
            if time.time() - self._last_beat_wall > timeout:
                return None

        try:
            tap_time    = float(tap.data["tap_time"].data[0])
            port_name   = str(tap.data["port_name"].data)
            note        = float(tap.data["note"].data[0])
            velocity    = float(tap.data["velocity"].data[0])
            participant = str(tap.data["participant"].data)
        except Exception:
            return None

        # Ignore retained taps: if this is the exact same tap we already
        # handled (e.g. process() was triggered by a beat, not a new tap),
        # do nothing. Keyed on (participant, tap_time) so two participants
        # tapping at the same instant are both kept.
        tap_key = (participant, tap_time)
        if tap_key == self._last_tap_seen:
            return None
        self._last_tap_seen = tap_key

        label = self._get_label(participant)
        now = tap_time

        # Nearest beat + asynchrony for this tap.
        threshold_ms = self.params.sync.threshold_ms.value
        beat_entry, async_ms = self._nearest_beat(tap_time)
        within_window = (1.0 if not np.isnan(async_ms)
                         and abs(async_ms) <= threshold_ms else 0.0)

        # Record this tap in the sliding window.
        self._tap_windows[label].append({"tap_time": tap_time, "async_ms": async_ms})

        # Phase + tempo reference for this tap.
        phase = beat_entry["phase"] if beat_entry else self._current_phase
        bpm   = beat_entry["bpm"] if beat_entry else self._held_bpm
        elapsed = beat_entry.get("elapsed", float("nan")) if beat_entry else float("nan")
        is_sync = (phase == "synchronization")

        taps = self._window_taps(label, now)
        mean_iti = self._mean_iti_ms(taps)

        if is_sync:
            metro_sync, mean_async = self._metro_sync(taps)
            out_async, out_within = async_ms, within_window
        else:
            metro_sync = mean_async = float("nan")
            out_async = out_within = float("nan")

        # Console feedback.
        if is_sync:
            ch = "ok" if within_window == 1.0 else " x"
            print(f"[SyncAnalyzer] {label:3s} | sync | async={async_ms:+7.1f}ms {ch} "
                  f"| metro_sync={metro_sync:.2f}")
        else:
            print(f"[SyncAnalyzer] {label:3s} | cont | mean_iti={mean_iti:7.1f}ms")

        table = {
            "tap_time":      self._arr(tap_time),
            "participant":   self._str(label),
            "port_name":     self._str(port_name),
            "note":          self._arr(note),
            "velocity":      self._arr(velocity),
            "phase":         self._str(phase),
            "bpm":           self._arr(bpm),
            "elapsed":       self._arr(elapsed),
            "async_ms":      self._arr(out_async),
            "within_window": self._arr(out_within),
            "metro_sync":    self._arr(metro_sync),
            "mean_iti_ms":   self._arr(mean_iti),
            "mean_async_ms": self._arr(mean_async),
            "n_taps_window": self._arr(float(len(taps))),
        }

        meta = {"participant": label, "phase": phase}
        return {"sync_result": (table, meta)}
