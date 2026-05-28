"""
MetronomeGenerator — Groovy pSync Node 2
=========================================
Produces a metronomic beat stream for sensorimotor synchronisation trials.

Every beat this node emits one TABLE containing precise timing metadata.
During the synchronisation phase it also emits an audio click buffer on
the audio_click output — wire that directly to AudioOut for sound.
During continuation the audio_click output returns None so AudioOut
goes silent automatically.

Trial phase timeline
---------------------
  |<-------- sync_duration -------->|<--- continuation_duration --->|
  0s                                45s                             80s
  [CLICK + beat events]             [silent + beat events]          [stopped]

Outputs
-------
beat : TABLE
    beat_time      : wall-clock time of this beat (float, seconds since epoch)
    beat_index     : cumulative beat counter from trial start (float[1])
    bpm            : BPM used for this trial (float[1])
    beat_interval  : seconds per beat (float[1])
    phase          : "synchronization" | "continuation" | "stopped"
    elapsed        : seconds since trial start (float[1])

audio_click : ARRAY
    Short sine-wave click buffer (float32 mono, shape [n_samples]).
    Wire this to AudioOut.  Emitted during synchronisation only;
    None during continuation so AudioOut stays silent automatically.

Parameters
----------
trial / bpm
    Fixed tempo in BPM (100–140).  Ignored when randomise_bpm is True.
trial / sync_duration_sec
    Length of synchronisation phase in seconds (default 45).
trial / continuation_duration_sec
    Length of continuation phase in seconds (default 35).
    Total trial = sync + continuation = 80 s by default.
trial / randomise_bpm
    Pick a random BPM in [bpm_min, bpm_max] each time the trial starts.
trial / bpm_min / bpm_max
    Range for random BPM (default 100–140, per Groovy pSync protocol).
trial / running
    Toggle False → True to start / restart a trial.

click / frequency_hz
    Pitch of the click sine tone (Hz, default 1000).
click / duration_ms
    Length of each click sound (ms, default 12).
click / volume
    Click amplitude 0–1 (default 0.8).
click / sample_rate
    Sample rate for click buffer generation (default 44100).

Install requirements:
    pip install numpy
    (sounddevice only needed by AudioOut, not this node)
"""

import random
import threading
import time
from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sine_click(sample_rate: int, freq: float, dur_ms: float, volume: float) -> np.ndarray:
    """Short sine-wave click with cosine fade-out, float32 mono."""
    n = max(1, int(sample_rate * dur_ms / 1000.0))
    t = np.linspace(0.0, dur_ms / 1000.0, n, endpoint=False)
    wave = np.sin(2.0 * np.pi * freq * t)
    fade = np.cos(np.linspace(0.0, np.pi / 2.0, n))
    return (wave * fade * volume).astype(np.float32)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MetronomeGenerator(Node):
    """
    Drift-corrected metronome for Groovy pSync.

    Set 'running' to True in the parameter panel to start a trial.
    Toggle it False → True to restart with fresh timing (and a new random
    BPM if randomise_bpm is enabled).

    Wire audio_click → AudioOut for audible clicks during synchronisation.
    Wire beat → SyncAnalyzer for asynchrony computation.
    """

    NO_MULTIPROCESSING = True  # background timing thread, avoid pickling

    # ------------------------------------------------------------------
    @staticmethod
    def config_input_slots():
        return {}

    @staticmethod
    def config_output_slots():
        return {
            "beat":        DataType.TABLE,
            "audio_click": DataType.ARRAY,
        }

    @staticmethod
    def config_params():
        return {
            "trial": {
                "bpm": FloatParam(
                    120.0, 100.0, 140.0,
                    doc="Metronome tempo in BPM. Used when randomise_bpm is False.",
                ),
                "sync_duration_sec": FloatParam(
                    45.0, 5.0, 300.0,
                    doc="Duration of synchronisation phase (s). Metronome clicks are audible.",
                ),
                "continuation_duration_sec": FloatParam(
                    35.0, 5.0, 300.0,
                    doc="Duration of continuation phase (s). Metronome is silent; "
                        "participant continues tapping from memory.",
                ),
                "randomise_bpm": BoolParam(
                    False,
                    doc="If True, pick a random BPM in [bpm_min, bpm_max] at trial start.",
                ),
                "bpm_min": FloatParam(
                    100.0, 60.0, 140.0,
                    doc="Lower bound for random BPM (used when randomise_bpm is True).",
                ),
                "bpm_max": FloatParam(
                    140.0, 100.0, 200.0,
                    doc="Upper bound for random BPM (used when randomise_bpm is True).",
                ),
                "running": BoolParam(
                    False,
                    doc="Toggle True to start the trial. Toggle False then True to restart.",
                ),
            },
            "click": {
                "frequency_hz": FloatParam(
                    1000.0, 100.0, 8000.0,
                    doc="Frequency of the click tone (Hz).",
                ),
                "duration_ms": FloatParam(
                    12.0, 1.0, 100.0,
                    doc="Duration of each click sound (ms).",
                ),
                "volume": FloatParam(
                    0.8, 0.0, 1.0,
                    doc="Click amplitude (0 = silent, 1 = full scale).",
                ),
                "sample_rate": IntParam(
                    44100, 8000, 192000,
                    doc="Sample rate for click buffer generation (Hz). "
                        "Must match AudioOut sample rate.",
                ),
            },
        }

    # ------------------------------------------------------------------
    def setup(self):
        self._queue: deque = deque()
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = None

    def terminate(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Parameter callback — fires when 'running' changes in the GUI
    # ------------------------------------------------------------------

    def trial_running_changed(self, value):
        if value:
            self._start_trial()
        else:
            self._stop.set()
            print("[MetronomeGenerator] Paused.")

    # ------------------------------------------------------------------
    # Trial start
    # ------------------------------------------------------------------

    def _start_trial(self):
        # Stop any running thread first
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._stop.clear()

        with self._lock:
            self._queue.clear()

        # Resolve BPM
        if self.params.trial.randomise_bpm.value:
            lo  = self.params.trial.bpm_min.value
            hi  = self.params.trial.bpm_max.value
            bpm = round(random.uniform(lo, hi), 3)
        else:
            bpm = self.params.trial.bpm.value

        sync_dur = self.params.trial.sync_duration_sec.value
        cont_dur = self.params.trial.continuation_duration_sec.value
        total    = sync_dur + cont_dur

        print(f"\n[MetronomeGenerator] ══ Trial starting ══")
        print(f"  BPM              : {bpm:.3f}")
        print(f"  Beat interval    : {60.0/bpm:.4f} s  ({60000.0/bpm:.1f} ms)")
        print(f"  Sync phase       : 0 – {sync_dur:.0f} s   (clicks audible)")
        print(f"  Continuation     : {sync_dur:.0f} – {total:.0f} s  (silent)")
        print(f"  Total duration   : {total:.0f} s")
        print(f"  Beats in sync    : {int(sync_dur / (60.0/bpm))}")
        print(f"  Beats total      : {int(total / (60.0/bpm))}\n")

        self._thread = threading.Thread(
            target=self._beat_loop,
            args=(bpm, sync_dur, cont_dur),
            daemon=True,
            name="MetronomeGenerator-beat",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # High-resolution beat loop
    # ------------------------------------------------------------------

    def _beat_loop(self, bpm: float, sync_dur: float, cont_dur: float):
        beat_interval = 60.0 / bpm
        total_dur     = sync_dur + cont_dur

        # Build click buffer once per trial (params won't change mid-trial)
        sr       = self.params.click.sample_rate.value
        freq     = self.params.click.frequency_hz.value
        dur_ms   = self.params.click.duration_ms.value
        volume   = self.params.click.volume.value
        click_buf = _sine_click(sr, freq, dur_ms, volume)

        t0_perf  = time.perf_counter()
        t0_wall  = time.time()
        beat_idx = 0
        prev_phase = None

        while not self._stop.is_set():
            # Target time for this beat
            target_perf = t0_perf + beat_idx * beat_interval

            # Coarse sleep, then busy-wait the final 1 ms for precision
            slack = target_perf - time.perf_counter()
            if slack > 0.001:
                time.sleep(slack - 0.001)
            while time.perf_counter() < target_perf:
                if self._stop.is_set():
                    return

            elapsed   = time.perf_counter() - t0_perf
            beat_wall = t0_wall + elapsed

            # Determine phase
            if elapsed < sync_dur:
                phase = "synchronization"
            elif elapsed < total_dur:
                phase = "continuation"
            else:
                phase = "stopped"

            # Announce phase transitions
            if phase != prev_phase:
                if phase == "continuation":
                    print(f"[MetronomeGenerator] ── Continuation phase (beat {beat_idx}) ──")
                elif phase == "stopped":
                    print(f"[MetronomeGenerator] ── Trial complete after {beat_idx} beats ──\n")
            prev_phase = phase

            # Emit event
            click_out = click_buf.copy() if phase == "synchronization" else None
            with self._lock:
                self._queue.append({
                    "beat_wall":     beat_wall,
                    "beat_index":    beat_idx,
                    "bpm":           bpm,
                    "beat_interval": beat_interval,
                    "phase":         phase,
                    "elapsed":       elapsed,
                    "click_buf":     click_out,
                })

            if phase == "stopped":
                break

            beat_idx += 1

    # ------------------------------------------------------------------
    # process() — drains one event per call
    # ------------------------------------------------------------------

    def process(self):
        with self._lock:
            if not self._queue:
                return None
            event = self._queue.popleft()

        sr = float(self.params.click.sample_rate.value)

        beat_table = {
            "beat_time":     Data(DataType.ARRAY,  np.array([event["beat_wall"]]),           {}),
            "beat_index":    Data(DataType.ARRAY,  np.array([float(event["beat_index"])]),   {}),
            "bpm":           Data(DataType.ARRAY,  np.array([event["bpm"]]),                 {}),
            "beat_interval": Data(DataType.ARRAY,  np.array([event["beat_interval"]]),       {}),
            "phase":         Data(DataType.STRING, event["phase"],                           {}),
            "elapsed":       Data(DataType.ARRAY,  np.array([event["elapsed"]]),             {}),
        }

        beat_meta = {
            "bpm":        event["bpm"],
            "phase":      event["phase"],
            "beat_index": event["beat_index"],
        }

        if event["click_buf"] is not None:
            click_out = (event["click_buf"], {"sfreq": sr})
        else:
            click_out = None

        return {
            "beat":        (beat_table, beat_meta),
            "audio_click": click_out,
        }