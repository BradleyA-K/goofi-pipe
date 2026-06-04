"""
GroupSyncScore — Groovy pSync Node 4
======================================
Turns the per-participant windowed averages from SyncAnalyzer into a
single, scalable group-synchronisation metric:

    "How many of the N participants are currently in sync?"

It keeps the latest windowed values for every participant it has seen,
decides per participant whether they are in sync (phase-dependent rule
below), and reports the count.  Works for any number of participants.

In-sync rule (phase dependent)
------------------------------
  SYNC phase  (metronome audible)
      A participant is in sync if their windowed metro_sync
      (fraction of taps landing on the beat) >= metro_sync_min.
      → they are matching the metronome.

  CONTINUATION phase  (metronome silent)
      Compute the group reference tempo = median of all participants'
      windowed mean_iti_ms.  A participant is in sync if their mean ITI
      is within tempo_tol_ms of that group median.
      → they are matching each other's tempo.

Stale participants (no tap for expiry_sec) are dropped from the roster.

Outputs
-------
group_score : ARRAY[1]   smoothed fraction in sync ∈ [0,1]
scores      : TABLE      drives OSCOut -> standalone pygame dance display
    phase                "synchronization"|"continuation"|"stopped" (string)
    elapsed              seconds since trial start (float[1])
    n_in_sync            number of participants in sync right now (float[1])
    n_total              number of active participants (float[1])
    frac_in_sync         smoothed n_in_sync / n_total ∈ [0,1] (float[1])
    group_median_iti_ms  reference tempo in continuation (float[1])
    P1_in_sync ...       per-participant smoothed flag ∈ [0,1] (float[1])

Wire scores -> OSCOut.  Each table key becomes an OSC message at
"<prefix>/<key>", e.g. /goofi/P1_in_sync, /goofi/n_total — that is what
dance_display.py listens for.

Parameters
----------
scoring / metro_sync_min   Min windowed metro_sync to count as on-beat
                           during the sync phase (default 0.5).
scoring / tempo_tol_ms     Max |ITI - group median| in ms to count as
                           tempo-matched during continuation (default 50).
scoring / smooth           EMA factor 0-0.99 on flags + group score, for a
                           smooth dance fade-in/out (default 0.7).
scoring / expiry_sec       Drop a participant after this many seconds with
                           no new tap (default 3.0).
"""

import time
from collections import defaultdict

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


def _plabel_key(label: str) -> int:
    """Sort key for "P1", "P2", ... (falls back to 0 for odd labels)."""
    return int(label[1:]) if label[1:].isdigit() else 0


class GroupSyncScore(Node):
    """
    Aggregates SyncAnalyzer's per-participant averages into a group
    "n in sync / n total" metric plus per-participant flags.
    """

    NO_MULTIPROCESSING = True

    @staticmethod
    def config_input_slots():
        return {"sync_result": DataType.TABLE}

    @staticmethod
    def config_output_slots():
        return {
            "group_score": DataType.ARRAY,
            "scores":      DataType.TABLE,
        }

    @staticmethod
    def config_params():
        return {
            "scoring": {
                "metro_sync_min": FloatParam(
                    0.5, 0.0, 1.0,
                    doc="Sync phase: min windowed metro_sync (fraction of taps "
                        "on the beat) for a participant to count as in sync.",
                ),
                "tempo_tol_ms": FloatParam(
                    50.0, 1.0, 500.0,
                    doc="Continuation: max |mean_iti - group median| in ms for a "
                        "participant to count as tempo-matched to the group.",
                ),
                "smooth": FloatParam(
                    0.7, 0.0, 0.99,
                    doc="Exponential smoothing on flags and group score "
                        "(0 = raw, 0.99 = very slow fade). Default 0.7.",
                ),
                "expiry_sec": FloatParam(
                    3.0, 0.5, 30.0,
                    doc="Drop a participant from the roster after this many "
                        "seconds with no new tap.",
                ),
            }
        }

    # ------------------------------------------------------------------
    def setup(self):
        # participant -> latest record {iti, metro_sync, t, elapsed}
        self._latest: dict[str, dict] = {}
        self._smoothed_flags: dict[str, float] = defaultdict(float)
        self._smoothed_group: float = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _scalar(table, key, default=float("nan")) -> float:
        try:
            return float(table[key].data[0])
        except Exception:
            return default

    # ------------------------------------------------------------------
    def process(self, sync_result):
        if sync_result is None:
            return None

        data = sync_result.data

        metro_min  = self.params.scoring.metro_sync_min.value
        tempo_tol  = self.params.scoring.tempo_tol_ms.value
        smooth     = self.params.scoring.smooth.value
        expiry     = self.params.scoring.expiry_sec.value

        # ── update roster with this tap's participant ────────────────
        try:
            participant = str(data["participant"].data)
        except Exception:
            return None
        phase = str(data["phase"].data) if "phase" in data else "unknown"

        this_t = self._scalar(data, "tap_time", time.time())
        self._latest[participant] = {
            "iti":        self._scalar(data, "mean_iti_ms"),
            "metro_sync": self._scalar(data, "metro_sync"),
            "elapsed":    self._scalar(data, "elapsed"),
            "t":          this_t,
        }

        # ── expire stale participants ────────────────────────────────
        # Clock comes from the tap stream itself (tap_time), so this works
        # for live MIDI and for replayed/offline data alike.
        now = max(r["t"] for r in self._latest.values())
        for p in [p for p, r in self._latest.items() if now - r["t"] > expiry]:
            del self._latest[p]
            self._smoothed_flags.pop(p, None)
        if not self._latest:
            return None

        labels = sorted(self._latest.keys(), key=_plabel_key)

        # ── group reference tempo (median of valid ITIs) ─────────────
        itis = [self._latest[p]["iti"] for p in labels
                if not np.isnan(self._latest[p]["iti"])]
        group_median = float(np.median(itis)) if itis else float("nan")

        # ── per-participant in-sync decision ─────────────────────────
        raw_flags: dict[str, float] = {}
        for p in labels:
            rec = self._latest[p]
            if phase == "synchronization":
                ms = rec["metro_sync"]
                raw = 1.0 if (not np.isnan(ms) and ms >= metro_min) else 0.0
            elif phase == "continuation":
                iti = rec["iti"]
                if np.isnan(iti) or np.isnan(group_median):
                    raw = 0.0
                else:
                    raw = 1.0 if abs(iti - group_median) <= tempo_tol else 0.0
            else:
                raw = 0.0
            raw_flags[p] = raw

            prev = self._smoothed_flags.get(p, raw)
            self._smoothed_flags[p] = smooth * prev + (1.0 - smooth) * raw

        n_total = len(labels)
        n_in_sync = int(sum(raw_flags.values()))
        raw_frac = n_in_sync / n_total if n_total else 0.0
        self._smoothed_group = (smooth * self._smoothed_group
                                + (1.0 - smooth) * raw_frac)

        elapsed = self._latest[participant]["elapsed"]
        if np.isnan(elapsed):
            elapsed = 0.0

        print(f"[GroupSyncScore] {phase[:4]} | in sync: {n_in_sync}/{n_total} "
              f"| frac={self._smoothed_group:.2f}")

        # ── build scores table ───────────────────────────────────────
        scores = {
            "phase":               Data(DataType.STRING, phase, {}),
            "elapsed":             Data(DataType.ARRAY, np.array([elapsed]), {}),
            "n_in_sync":           Data(DataType.ARRAY, np.array([float(n_in_sync)]), {}),
            "n_total":             Data(DataType.ARRAY, np.array([float(n_total)]), {}),
            "frac_in_sync":        Data(DataType.ARRAY, np.array([self._smoothed_group]), {}),
            "group_median_iti_ms": Data(DataType.ARRAY, np.array([group_median]), {}),
        }
        for p in labels:
            scores[f"{p}_in_sync"] = Data(
                DataType.ARRAY, np.array([self._smoothed_flags[p]]), {}
            )

        return {
            "group_score": (np.array([self._smoothed_group]), {}),
            "scores":      (scores, {"phase": phase, "n_in_sync": n_in_sync}),
        }
