#!/usr/bin/env python3
"""
corner_chroma_flicker.py
Flicker corner patches between two very-similar RGB colors (e.g., [250,0,0] <-> [255,0,0]).
Supports fixed frequency or linear chirp sweeping from f_start -> f_end over chirp_duration seconds.
"""

import glfw
from OpenGL.GL import *
import time
import math

# ------------------------
# Visual parameters
# ------------------------
patch_size = 120   # pixels (corner patch width/height)
alpha = 1.0        # patch alpha (1.0 opaque)
bg_alpha = 0.0     # background alpha to keep transparent overlay window

# Two very similar colors (0..255). Example: two reds with small delta.
colorA = (250, 0, 0)
colorB = (255, 0, 0)

# Convert to normalized floats 0..1 for GL
def norm(c): return (c[0]/255.0, c[1]/255.0, c[2]/255.0)

colorA_f = norm(colorA)
colorB_f = norm(colorB)

# ------------------------
# Temporal / chirp parameters
# ------------------------
use_chirp = True
f_start = 60.0         # starting frequency (Hz)
f_end = 240.0          # end frequency (Hz)
chirp_duration = 8.0   # seconds to sweep from f_start to f_end (then repeat)
duty_ms = 2.0          # pulse width in milliseconds (very short -> more invisible to RGB)
repeat = True

# Per-corner independent phase offsets (optional)
corner_phase_offsets = {
    "topleft": 0.0,
    "topright": 0.25,    # fractional offset (0..1) of the chirp timeline
    "bottomleft": 0.5,
    "bottomright": 0.75,
}

# ------------------------
# GLFW / OpenGL init
# ------------------------
def init_window():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    width, height = mode.size.width, mode.size.height

    # Transparent overlay window hints (may be platform-dependent)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)

    window = glfw.create_window(width, height, "Chroma Flicker Patches", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)

    # Set up orthographic projection matching pixel coords
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # blending & clear color (transparent)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.0, 0.0, 0.0, 0.0)

    return window, width, height

def draw_quad(x, y, w, h, rgb_f, a=1.0):
    r,g,b = rgb_f
    glColor4f(r, g, b, a)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

# compute instantaneous frequency for a linear chirp (triangular repeat if desired)
def inst_freq_linear(t_local, start, end, duration):
    # t_local in [0,duration)
    return start + (end - start) * (t_local / duration)

# main loop
def main():
    window, W, H = init_window()
    start_time = time.time()
    last_log = 0.0

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        now = time.time()
        t = now - start_time

        # For each corner, compute a local chirp time (with phase offsets) so corners can be slightly desynced
        for corner, offset_frac in corner_phase_offsets.items():
            # local time for this corner's chirp
            if repeat:
                # incorporate phase offset by shifting t before taking mod
                t_corner = (t + offset_frac * chirp_duration) % chirp_duration
            else:
                t_corner = min(t + offset_frac * chirp_duration, chirp_duration)

            # instantaneous frequency (Hz)
            if use_chirp:
                f_inst = inst_freq_linear(t_corner, f_start, f_end, chirp_duration)
            else:
                f_inst = f_start

            period_ms = 1000.0 / max(1e-6, f_inst)
            phase_ms = (now * 1000.0) % period_ms
            display_on = (phase_ms < duty_ms)

            # choose color for this corner (alternate between A and B each pulse)
            # compute a simple square wave phase to alternate color per half-cycle:
            half_cycle_idx = int((now * f_inst) * 2)  # increments twice per cycle
            use_colorB = (half_cycle_idx % 2) == 0

            rgb = colorB_f if use_colorB else colorA_f

            if display_on:
                # draw the respective corner patch
                if corner == "topleft":
                    x = 0
                    y = H - patch_size
                elif corner == "topright":
                    x = W - patch_size
                    y = H - patch_size
                elif corner == "bottomleft":
                    x = 0
                    y = 0
                elif corner == "bottomright":
                    x = W - patch_size
                    y = 0
                else:
                    continue

                draw_quad(x, y, patch_size, patch_size, rgb, alpha)

        # Swap & poll
        glfw.swap_buffers(window)
        glfw.poll_events()

        # small sleep to avoid pegging CPU, but keep responsiveness
        time.sleep(0.0005)

    glfw.terminate()

if __name__ == "__main__":
    main()
