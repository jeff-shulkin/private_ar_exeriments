import glfw
from OpenGL.GL import *
from PIL import Image
import time
import argparse
import pathlib

# Flicker frequency (Hz)
frequency = 60.0

def init_window():
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)

    # Transparent overlay window
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)

    window = glfw.create_window(
        mode.size.width, mode.size.height, "Flicker Image", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, mode.size.width, 0, mode.size.height, -1, 1)
    glMatrixMode(GL_MODELVIEW)

    return window, mode.size.width, mode.size.height

def load_texture(path):
    img = Image.open(path).convert("RGBA")
    img_data = img.tobytes("raw", "RGBA", 0, -1)
    width, height = img.size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return tex_id, width, height

def draw_fullscreen_image(tex_id, W, H, img_w, img_h):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(W, 0)
    glTexCoord2f(1, 1); glVertex2f(W, H)
    glTexCoord2f(0, 1); glVertex2f(0, H)
    glEnd()

    glDisable(GL_TEXTURE_2D)

def main(image_path: pathlib.Path, flicker_type="fixed",):

    # LFM chirp settings
    f_start = 1.0
    f_end = 5.0
    chirp_duration = 10.0
    repeat = True

    window, W, H = init_window()
    tex_id, img_w, img_h = load_texture(image_path)
    start_time = time.time()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        t = time.time() - start_time
        if flicker_type == "fixed":
            phase = int(t * frequency) % 2  # square wave
            if phase == 0:
                draw_fullscreen_image(tex_id, W, H, img_w, img_h)
        elif flicker_type == "lfm":
            t_chirp = t & chirp_duration if repeat else min(t, chirp_duration)
            curr_freq = f_start + (f_end - f_start) * (t_chirp / chirp_duration)
            phase = math.sin(2 * math.pi * ((f_start * t_chirp) + 0.5 * ((f_end - f_start) / chirp_duration) * t_chirp**2))
            if phase > 0:
                draw_fullscreen_image(tex_id, H, W)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image path parser")
    parser.add_argument('-i', '--image_filename')
    parser.add_argument('-f', '--flicker_type')
    args = parser.parse_args()
    image_path = pathlib.Path(args.image_filename)
    main(image_path)
