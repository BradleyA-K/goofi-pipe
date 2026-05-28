"""
MidiIn — Groovy pSync Node 1  (MIDI edition)
===============================================
Listens on any number of MIDI input ports and emits one TABLE per
note-on tap event.

Outputs
-------
tap : TABLE
    tap_time    : wall-clock time of the tap (float, seconds since epoch)
    source      : "midi:<port_name>"
    port_name   : full MIDI port name string
    participant : resolved participant label — from participant_map,
                  or port_name if no map is set / no rule matches
    note        : MIDI note number (float array, shape [1])
    velocity    : MIDI velocity 0-127 (float array, shape [1])
    channel     : MIDI channel 0-15 (float array, shape [1])

Parameters
----------
midi / ports
    Comma-separated substrings matched against available MIDI input port
    names.  Use "*" to open every available port.  Leave blank → no input.

midi / participant_map
    Maps incoming MIDI messages to participant labels.
    Format: port:note:label, comma-separated. Use * as wildcard.
    Rules are checked top to bottom — first match wins.

    Examples:
      *:38:P1, *:42:P2, *:46:P3      one device, participants by note
      Alesis:*:P1, Roland:*:P2       one participant per device
      Alesis:38:P1, Alesis:42:P2     specific device + specific note

    Leave blank → participant copies port_name (existing behaviour).
    Reparsed automatically whenever you edit it — safe to change
    between trials, do not change mid-trial.

midi / note_on_only
    When True (default) only react to note_on messages with velocity > 0.
    When False, note_off / velocity-0 note_on also count as taps.

midi / min_velocity
    Ignore notes below this velocity (0 = accept all).

midi / print_available_ports
    Toggle to True once to print all detected MIDI ports to the console.

Install requirements:
    pip install mido python-rtmidi
"""

import threading
import time
from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


# ── TABLE factory ─────────────────────────────────────────────────────

def _make_tap_table(tap_time, port_name, participant, note, velocity, channel):
    return {
        "tap_time":    Data(DataType.ARRAY,  np.array([tap_time]),        {}),
        "source":      Data(DataType.STRING, f"midi:{port_name}",         {}),
        "port_name":   Data(DataType.STRING, port_name,                   {}),
        "participant": Data(DataType.STRING, participant,                  {}),
        "note":        Data(DataType.ARRAY,  np.array([float(note)]),     {}),
        "velocity":    Data(DataType.ARRAY,  np.array([float(velocity)]), {}),
        "channel":     Data(DataType.ARRAY,  np.array([float(channel)]),  {}),
    }


# ── Participant map parser ─────────────────────────────────────────────

def _parse_participant_map(raw: str) -> list[tuple[str, str, str]]:
    """
    Parse "port:note:label, ..." into a list of (port_pat, note_pat, label).
    Logs warnings for malformed rules.  Returns [] if raw is blank.
    """
    rules = []
    if not raw.strip():
        return rules

    for i, rule in enumerate(raw.split(",")):
        rule = rule.strip()
        if not rule:
            continue
        parts = [p.strip() for p in rule.split(":")]
        if len(parts) != 3:
            print(f"[MidiIn] participant_map rule {i+1} malformed "
                  f"(expected port:note:label, got '{rule}') — skipped.")
            continue
        port_pat, note_pat, label = parts
        if not label:
            print(f"[MidiIn] participant_map rule {i+1} has empty label "
                  f"('{rule}') — skipped.")
            continue
        rules.append((port_pat, note_pat, label))

    return rules


class MidiIn(Node):
    """
    MIDI tap-input node for Groovy pSync.

    Opens one or more MIDI input ports and enqueues every qualifying
    note-on message as a tap event.  Each tap is labelled with a
    participant identifier resolved from participant_map, or the raw
    port name if no map is configured.

    Tip: toggle print_available_ports to see connected devices.
    """

    NO_MULTIPROCESSING = True  # background threads — avoid pickling

    # ── goofi-pipe interface ───────────────────────────────────────────

    @staticmethod
    def config_input_slots():
        return {}

    @staticmethod
    def config_output_slots():
        return {"tap": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "midi": {
                "ports": StringParam(
                    "*",
                    doc="Comma-separated substrings matched against MIDI input port names. "
                        "'*' opens all available ports. Leave blank to disable.",
                ),
                "participant_map": StringParam(
                    "",
                    doc=(
                        "Maps MIDI messages to participant labels.\n"
                        "Format: port:note:label, comma-separated. * = wildcard.\n"
                        "Examples:\n"
                        "  *:38:P1, *:42:P2       by note, any device\n"
                        "  Alesis:*:P1            any note from Alesis\n"
                        "  Alesis:38:P1           specific device + note\n"
                        "Leave blank → participant = port_name."
                    ),
                ),
                "note_on_only": BoolParam(
                    True,
                    doc="Only react to note_on messages (velocity > 0). "
                        "False also captures note_off / velocity-0 note_on.",
                ),
                "min_velocity": IntParam(
                    1, 0, 127,
                    doc="Ignore notes with velocity below this value (0 = accept all).",
                ),
                "print_available_ports": BoolParam(
                    False,
                    doc="Toggle True to print available MIDI ports to the console.",
                ),
            }
        }

    # ── lifecycle ─────────────────────────────────────────────────────

    def setup(self):
        self._tap_queue: deque = deque()
        self._lock = threading.Lock()
        self._open_ports = []

        # Parse map once at startup
        self._rules: list[tuple[str, str, str]] = _parse_participant_map(
            self.params.midi.participant_map.value
        )
        self._log_rules()
        self._open_midi_ports()

    def terminate(self):
        for port in self._open_ports:
            try:
                port.close()
            except Exception:
                pass
        self._open_ports.clear()

    # ── parameter change callbacks ─────────────────────────────────────

    def midi_print_available_ports_changed(self, value):
        if value:
            self._print_ports()

    def midi_participant_map_changed(self, value):
        """Reparse the map whenever it is edited in the GUI."""
        self._rules = _parse_participant_map(value)
        self._log_rules()

    # ── helpers ───────────────────────────────────────────────────────

    def _log_rules(self):
        if not self._rules:
            print("[MidiIn] participant_map: blank — participant = port_name")
            return
        print(f"[MidiIn] participant_map: {len(self._rules)} rule(s)")
        for port_pat, note_pat, label in self._rules:
            print(f"  port='{port_pat}'  note='{note_pat}'  → '{label}'")

    def _resolve_participant(self, port_name: str, note: int) -> str:
        """
        Walk the cached rules top-to-bottom.
        First matching rule wins.  Falls back to port_name if nothing matches.
        """
        for port_pat, note_pat, label in self._rules:
            port_match = (port_pat == "*" or port_pat in port_name)
            note_match = (note_pat == "*" or note_pat == str(note))
            if port_match and note_match:
                return label
        # No match — fall back to port_name (blank map or unrecognised note)
        return port_name

    def _print_ports(self):
        try:
            import mido
            names = mido.get_input_names()
        except ImportError:
            print("[MidiIn] mido not installed.")
            return
        print("\n[MidiIn] Available MIDI input ports:")
        if names:
            for i, n in enumerate(names):
                print(f"  [{i}] {n}")
        else:
            print("  (none found)")
        print()

    def _open_midi_ports(self):
        try:
            import mido
        except ImportError:
            print("[MidiIn] mido / python-rtmidi not installed — MIDI disabled.\n"
                  "         Install with: pip install mido python-rtmidi")
            return

        available = mido.get_input_names()
        ports_param = self.params.midi.ports.value.strip()

        if not ports_param:
            print("[MidiIn] 'ports' param is empty — no MIDI ports opened.")
            return

        targets = available if ports_param == "*" else [
            p for p in available
            if any(s.strip() in p for s in ports_param.split(",") if s.strip())
        ]

        if not targets:
            print(f"[MidiIn] No MIDI ports matched '{ports_param}'.")
            print(f"[MidiIn] Available: {available}")
            return

        for port_name in targets:
            try:
                port = mido.open_input(port_name)
            except Exception as e:
                print(f"[MidiIn] Could not open '{port_name}': {e}")
                continue
            self._open_ports.append(port)
            print(f"[MidiIn] Opened MIDI port: {port_name}")
            t = threading.Thread(
                target=self._listen_loop,
                args=(port, port_name),
                daemon=True,
                name=f"MidiIn-{port_name}",
            )
            t.start()

    def _listen_loop(self, port, port_name):
        """Blocking message loop — one daemon thread per open port."""
        for msg in port:
            note_on_only = self.params.midi.note_on_only.value
            min_vel      = self.params.midi.min_velocity.value

            is_note_on  = (msg.type == "note_on" and msg.velocity > 0)
            is_note_off = (msg.type == "note_off" or
                           (msg.type == "note_on" and msg.velocity == 0))

            if note_on_only and not is_note_on:
                continue
            if not note_on_only and not (is_note_on or is_note_off):
                continue
            if is_note_on and msg.velocity < min_vel:
                continue

            # Resolve participant label using cached rules
            participant = self._resolve_participant(port_name, msg.note)

            with self._lock:
                self._tap_queue.append((
                    time.time(),
                    port_name,
                    participant,
                    msg.note,
                    msg.velocity,
                    msg.channel,
                ))

    # ── process() ─────────────────────────────────────────────────────

    def process(self):
        with self._lock:
            if not self._tap_queue:
                return None
            tap_time, port_name, participant, note, velocity, channel = \
                self._tap_queue.popleft()

        table = _make_tap_table(tap_time, port_name, participant,
                                note, velocity, channel)
        meta  = {
            "source":      f"midi:{port_name}",
            "participant": participant,
            "note":        note,
            "velocity":    velocity,
        }
        return {"tap": (table, meta)}