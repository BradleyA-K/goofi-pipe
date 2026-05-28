"""
GroupSyncScore — Groovy pSync Node 4
======================================
Reads the sync_result TABLE from SyncAnalyzer and computes:

  1. Per-participant score  : how well each person is synchronised
                              to the stimulus (sync phase) or to
                              the group mean (continuation phase).
                              Range 0.0 → 1.0.

  2. Group score            : mean synchrony across all active dyadic
                              pairs.  Range 0.0 → 1.0.
                              Partial credit — if 2 of 3 pairs are
                              synchronised the score is ~0.67.

  3. Sync mode              : "phase" or "tempo" — selectable toggle.
                              Reads the matching _phase_sig or
                              _tempo_sig columns from SyncAnalyzer.

Outputs
-------
group_score : ARRAY   shape [1]  — 0.0 → 1.0 group synchrony
scores      : TABLE   — one float[1] column per participant +
                        one float[1] column per pair +
                        phase string +
                        elapsed float[1]

Wire scores → OSCOut to drive the pygame orb display.

Parameters
----------
scoring / mode
    "phase"  — use <Pi>_<Pj>_phase_sig for pair sync flags
    "tempo"  — use <Pi>_<Pj>_tempo_sig for pair sync flags
scoring / smooth
    Exponential smoothing factor 0–1 applied to group_score.
    0 = no smoothing (raw), 0.9 = heavy smoothing.
    Default 0.7 — gives a natural orb glow fade.
"""

import math
import re
from collections import defaultdict

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class GroupSyncScore(Node):
    """
    Aggregates SyncAnalyzer output into a single group sync score
    and per-participant scores for the pygame orb display.
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
                "mode": StringParam(
                    "phase",
                    options=["phase", "tempo"],
                    doc=(
                        "phase : use <Pi>_<Pj>_phase_sig columns from SyncAnalyzer.\n"
                        "tempo : use <Pi>_<Pj>_tempo_sig columns."
                    ),
                ),
                "smooth": FloatParam(
                    0.7, 0.0, 0.99,
                    doc="Exponential smoothing on group_score. "
                        "0 = raw, 0.99 = very slow fade. Default 0.7.",
                ),
            }
        }

    # ------------------------------------------------------------------
    def setup(self):
        self._smoothed_group = 0.0
        self._smoothed_participants: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    def process(self, sync_result):
        if sync_result is None:
            return None

        data   = sync_result.data
        mode   = self.params.scoring.mode.value
        smooth = self.params.scoring.smooth.value
        suffix = f"_{mode}_sig"   # e.g. "_phase_sig" or "_tempo_sig"

        # ── collect pair sync flags ───────────────────────────────────
        pair_flags = {}
        pair_pattern = re.compile(r"^(P\d+)_(P\d+)" + re.escape(suffix) + r"$")
        for key, val in data.items():
            m = pair_pattern.match(key)
            if m:
                try:
                    flag = float(val.data[0])
                    if not math.isnan(flag):
                        pair_flags[key] = flag
                except Exception:
                    pass

        # Group score = mean of all pair sync flags
        raw_group = float(np.mean(list(pair_flags.values()))) \
            if pair_flags else 0.0

        # ── per-participant score ─────────────────────────────────────
        # During sync phase: use within_window (tap-to-stimulus)
        # During continuation: use mean of all pair flags involving this participant
        phase = "unknown"
        try:
            phase = str(data["phase"].data)
        except Exception:
            pass

        participants = set()
        for key in pair_flags:
            m = pair_pattern.match(key)
            if m:
                participants.add(m.group(1))
                participants.add(m.group(2))

        participant_scores = {}
        for p in participants:
            if phase == "synchronization":
                # Individual score = within_window from tap-to-stimulus
                try:
                    participant = str(data["participant"].data)
                    if participant == p:
                        ww = float(data["within_window"].data[0])
                        raw_p = ww if not math.isnan(ww) else 0.0
                    else:
                        raw_p = self._smoothed_participants.get(p, 0.0)
                except Exception:
                    raw_p = self._smoothed_participants.get(p, 0.0)
            else:
                # Continuation: mean of pair flags involving this participant
                my_flags = [v for k, v in pair_flags.items()
                            if k.startswith(f"{p}_") or
                            f"_{p}{suffix}" in k]
                raw_p = float(np.mean(my_flags)) if my_flags else 0.0

            # Smooth
            prev = self._smoothed_participants.get(p, raw_p)
            smoothed_p = smooth * prev + (1.0 - smooth) * raw_p
            self._smoothed_participants[p] = smoothed_p
            participant_scores[p] = smoothed_p

        # ── smooth group score ────────────────────────────────────────
        self._smoothed_group = (smooth * self._smoothed_group
                                + (1.0 - smooth) * raw_group)

        # ── build outputs ─────────────────────────────────────────────
        scores_table = {
            "group_score": Data(DataType.ARRAY,
                                np.array([self._smoothed_group]), {}),
            "phase":       Data(DataType.STRING, phase, {}),
        }
        try:
            scores_table["elapsed"] = Data(
                DataType.ARRAY,
                np.array([float(data["elapsed"].data[0])]), {}
            )
        except Exception:
            scores_table["elapsed"] = Data(DataType.ARRAY, np.array([0.0]), {})

        for p, score in participant_scores.items():
            scores_table[f"{p}_score"] = Data(
                DataType.ARRAY, np.array([score]), {}
            )

        # Also pass pair flags through for the orb to use
        for key, flag in pair_flags.items():
            scores_table[key] = Data(DataType.ARRAY, np.array([flag]), {})

        return {
            "group_score": (np.array([self._smoothed_group]), {}),
            "scores":      (scores_table, {"phase": phase}),
        }