#!/usr/bin/env python3
"""
psync_visualizer.py — standalone pygame moth driven by Groovy pSync, live.

    goofi  (GroupSyncScore.scores --> OSCOut)  ==UDP/OSC==>  this window

No browser, no HTTP, no custom goofi node. Just run it. It listens for OSC
from goofi's built-in OSCOut node and flies a moth: still when nobody is in
sync, roaming + flapping when the group locks in.

GOOFI WIRING (one built-in node, no custom code):
    GroupSyncScore.scores  -->  OSCOut.data
    OSCOut params:  address 127.0.0.1   port 9001   prefix /goofi

RUN:
    python psync_visualizer.py
    python psync_visualizer.py --port 9001 --prefix /goofi   # if you changed them

Keys:  Esc / Q = quit       T = toggle a self-test wave (no goofi needed)
"""
import argparse
import math
import threading
import time

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server

import pygame


# ── shared state (written by the OSC thread, read by the draw loop) ──────
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.sync_target = 0.0      # 0..1 fraction of the group in sync
        self.n_in_sync = 0
        self.n_total = 0
        self.phase = ""
        self.last_update = 0.0      # wall-clock of last OSC message


def make_dispatcher(state, prefix):
    def handler(address, *args):
        key = address.rstrip("/").split("/")[-1]
        v = args[0] if args else None
        if v is None:
            return
        with state.lock:
            if key in ("frac_in_sync", "group_score"):
                state.sync_target = max(0.0, min(1.0, float(v)))
                state.last_update = time.time()
            elif key == "n_in_sync":
                state.n_in_sync = int(float(v))
            elif key == "n_total":
                state.n_total = int(float(v))
            elif key == "phase":
                state.phase = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
    d = osc_dispatcher.Dispatcher()
    d.set_default_handler(handler)   # catch every <prefix>/* address
    return d


# ── colours ──────────────────────────────────────────────────────────────
BG       = (29, 27, 34)
WING     = (146, 112, 74)
WING_EDGE = (58, 42, 31)
BODY     = (213, 185, 139)
BODY_HI  = (226, 205, 162)
HEAD     = (23, 24, 32)
CREAM    = (244, 231, 208)
EYE      = (17, 16, 24)
ANTENNA  = (201, 154, 95)


def _wing(surface, cx, cy, side, spread):
    """One wing as a teardrop polygon; `spread` 0..1 opens/closes it (flap)."""
    w = (60 + 150 * spread) * side
    pts = [
        (cx, cy),
        (cx + 0.7 * w, cy - 70),
        (cx + 1.05 * w, cy + 20),
        (cx + 0.85 * w, cy + 130),
        (cx + 0.3 * w, cy + 95),
    ]
    pygame.draw.polygon(surface, WING, pts)
    pygame.draw.polygon(surface, WING_EDGE, pts, 3)
    # eye-spot
    pygame.draw.circle(surface, (43, 33, 26), (int(cx + 0.78 * w), int(cy + 60)), 11)
    pygame.draw.circle(surface, ANTENNA, (int(cx + 0.78 * w), int(cy + 60)), 5)


def draw_moth(surface, cx, cy, flap, scale=1.0):
    # wings (behind body)
    spread = 0.15 + 0.85 * flap
    _wing(surface, cx, cy, -1, spread)
    _wing(surface, cx, cy, +1, spread)
    # body
    pygame.draw.ellipse(surface, BODY, (cx - 26, cy - 10, 52, 120))
    pygame.draw.ellipse(surface, BODY_HI, (cx - 30, cy - 60, 60, 70))
    # head
    pygame.draw.circle(surface, HEAD, (int(cx), int(cy - 58)), 26)
    # eyes
    pygame.draw.circle(surface, CREAM, (int(cx - 9), int(cy - 62)), 5)
    pygame.draw.circle(surface, CREAM, (int(cx + 9), int(cy - 62)), 5)
    pygame.draw.circle(surface, EYE, (int(cx - 9), int(cy - 62)), 2)
    pygame.draw.circle(surface, EYE, (int(cx + 9), int(cy - 62)), 2)
    # antennae (sway with the flap)
    sway = (flap - 0.5) * 26
    for s in (-1, 1):
        tip = (cx + s * 34 + sway * s, cy - 110)
        pygame.draw.line(surface, ANTENNA, (cx + s * 8, cy - 78), tip, 4)
        for i in range(6):
            pygame.draw.line(surface, ANTENNA, tip,
                             (tip[0] + s * 16, tip[1] + 6 + i * 4), 2)


def draw_hud(surface, font, big, W, H, sync, n_in, n_tot, phase):
    # sync bar
    pygame.draw.rect(surface, (60, 55, 65), (20, 20, 260, 16), border_radius=8)
    pygame.draw.rect(surface, (255, 200, 110),
                     (20, 20, int(260 * sync), 16), border_radius=8)
    label = big.render(f"sync {sync:0.2f}", True, CREAM)
    surface.blit(label, (20, 44))
    sub = f"in-sync {n_in}/{n_tot}    {phase or '-'}"
    if n_tot == 0 and phase not in ("test",):
        sub += "    (waiting for goofi)"
    surface.blit(font.render(sub, True, (210, 200, 180)), (20, 80))
    surface.blit(font.render("esc/q quit   t self-test", True, (120, 115, 110)),
                 (20, H - 30))


def run(state, args, max_frames=None):
    pygame.init()
    W, H = 820, 820
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Groovy pSync — Moth")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 20)
    big = pygame.font.SysFont("menlo,consolas,monospace", 28)

    sync = 0.0
    t = 0.0
    flap_phase = 0.0
    self_test = False
    frames = 0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_t:
                    self_test = not self_test

        with state.lock:
            target = state.sync_target
            n_in, n_tot, phase, last = (state.n_in_sync, state.n_total,
                                        state.phase, state.last_update)

        if self_test:
            target = 0.5 + 0.5 * math.sin(time.time() * 1.4)
            phase, n_in, n_tot = "test", 0, 0
        elif last and (time.time() - last) > args.freeze_after:
            # no data for a while (trial ended) → ease back to stillness
            target *= max(0.0, 1.0 - (time.time() - last - args.freeze_after))

        sync += (target - sync) * min(1.0, dt * 6.0)        # smooth value
        flap_phase += dt * (1.0 + sync * 9.0)               # faster flap when synced
        t += dt * (0.3 + sync * 1.6)

        flap = 0.5 + 0.5 * math.sin(flap_phase) * (0.1 + 0.9 * sync)
        cx = W / 2 + math.sin(t * 0.8) * 200 * sync          # roam more when synced
        cy = H / 2 + math.sin(t * 1.3) * 130 * sync + 40

        screen.fill(BG)
        draw_moth(screen, cx, cy, flap)
        draw_hud(screen, font, big, W, H, sync, n_in, n_tot, phase)
        pygame.display.flip()

        frames += 1
        if max_frames and frames >= max_frames:
            running = False
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="pygame moth driven by Groovy pSync over OSC")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001, help="must match goofi OSCOut port")
    ap.add_argument("--prefix", default="/goofi")
    ap.add_argument("--freeze-after", type=float, default=1.5,
                    help="seconds without OSC before the moth eases to stillness")
    args = ap.parse_args()

    state = State()
    srv = osc_server.ThreadingOSCUDPServer((args.ip, args.port),
                                           make_dispatcher(state, args.prefix))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[psync] listening for OSC on {args.ip}:{args.port}  (prefix {args.prefix})")
    print("[psync] wire goofi:  GroupSyncScore.scores -> OSCOut.data")
    try:
        run(state, args)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
