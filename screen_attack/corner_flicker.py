import glfw
from OpenGL.GL import *
import time

# ------------------------
# Flicker settings
# ------------------------
# Each corner has its own toggle frequency (Hz)
frequencies = {
    "topleft": 60.0,
    "topright": 60.0,
    "bottomleft": 60.0,
    "bottomright": 60.0,
}
patch_size = 100  # in pixels

def init_window():
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)

    # Transparent overlay window
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)   # no border
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)     # always on top

    window = glfw.create_window(
        mode.size.width, mode.size.height, "Flicker Patches", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)

    glClearColor(0.0, 0.0, 0.0, 0.0)  # alpha=0 background
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, mode.size.width, 0, mode.size.height, -1, 1)
    glMatrixMode(GL_MODELVIEW)

    return window, mode.size.width, mode.size.height

def draw_patch(x, y, w, h, on):
    if on:
        glColor4f(1.0, 1.0, 1.0, 1.0)  # opaque white
    else:
        glColor4f(0.0, 0.0, 0.0, 0.0)  # fully transparent
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

def main():
    window, W, H = init_window()
    start_time = time.time()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        t = time.time() - start_time

        # Compute flicker states
        states = {}
        for key, f in frequencies.items():
            phase = int(t * f) % 2  # simple square wave
            states[key] = (phase == 0)

        # Draw 4 patches
        draw_patch(0, H - patch_size, patch_size, patch_size, states["topleft"])
        draw_patch(W - patch_size, H - patch_size, patch_size, patch_size, states["topright"])
        draw_patch(0, 0, patch_size, patch_size, states["bottomleft"])
        draw_patch(W - patch_size, 0, patch_size, patch_size, states["bottomright"])

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
