class WindowDance:
    def __init__(self):
        import warnings
        # Ignore deprecation warnings from pygame
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API"
        )

        import tkinter as tk
        import librosa
        import pygame
        import math
        import random
        import numpy as np
        from PIL import Image, ImageTk
        from io import BytesIO
        from mutagen.id3 import ID3

        self.tk = tk
        self.librosa = librosa
        self.pygame = pygame
        self.math = math
        self.random = random
        self.np = np
        self.Image = Image
        self.ImageTk = ImageTk
        self.BytesIO = BytesIO
        self.ID3 = ID3

        self.setup_config()
        self.analyze_audio()
        self.setup_windows()
        self.setup_state()
        self.setup_audio()

    # -------------------------------------------------
    # Configuration

    def setup_config(self):
        self.AUDIO_FILE = input("\n\nEnter your file name (e.g., song.mp3): ")
        self.MAIN_WINDOW_JUMP = -int(input("Enter how much the windows jump (0 for none): "))
        self.UPDATE_HZ = max(1, int(input("Enter your frame rate: ")))
        self.TEMPO_INPUT = int(input("Enter the tempo (0-9 for auto): "))

        self.W_WIDTH, self.W_HEIGHT = 500, 300  # Main window size WxH -  -  -  -  -  -  -  (Default: 500 x 300)

        self.SQUARE_SIZE = 200                  # Dancer window size (square)   -  -  -  -  (Default: 200)
        self.ORBIT_RADIUS = 500                 # Orbit radius in pixels  -  -  -  -  -  -  (Default: 500)

        self.GATE_MULTIPLIER = 0.48             # Adaptive gate sensitivity  -  -  -  -  -  (Default: 0.48)
        self.BASS_RADIUS_PULL = 0.53            # Minimum orbit size   -  -  -  -  -  -  -  (Default: 0.53)
        self.MIN_GATE_DROP = 0.025              # Min RMS drop to trigger gate  -  -  -  -  (Default: 0.025)
        self.BASS_SPEED_BOOST = 3.6             # How much faster the orbit spins on bass   (Default: 3.6)
        self.BASS_SPEED_THRESHOLD = 0.45        # Bass strength to trigger speed boost   -  (Default: 0.45)
        self.VOLUME = 0.38                      # Audio volume (0.0 to 1.0)  -  -  -  -  -  (Default: 0.38)

        self.JUMP_DAMPING = 0.85                # Jump bounce damping  -  -  -  -  -  -  -  (Default: 0.85)
        self.JUMP_GRAVITY = 1.4                 # Jump gravity effect  -  -  -  -  -  -  -  (Default: 1.4)
        self.JUMP_BASS_THRESHOLD = 0.32         # Bass strength to trigger jump -  -  -  -  (Default: 0.32)
        self.JUMP_REARM_VELOCITY = 1.0          # Min downward velocity to rearm jump -  -  (Default: 1.0)

        self.TELEPORT_COOLDOWN = 2.7            # Seconds between teleports  -  -  -  -  -  (Default: 2.7)
        self.BG_FADE_SPEED = 0.12               # Background color fade speed   -  -  -  -  (Default: 0.12)
        self.BG_DANCERS_FADE_SPEED = 0.1        # Dancer bg color fade speed -  -  -  -  -  (Default: 0.1)

    # -------------------------------------------------
    # Audio analysis

    def analyze_audio(self):
        print("\nAnalyzing audio...\n")

        y, sr = self.librosa.load(self.AUDIO_FILE)

        if self.TEMPO_INPUT <= 0 or self.TEMPO_INPUT >= 9:
            tempo, _ = self.librosa.beat.beat_track(y=y, sr=sr)
            self.tempo = self.math.ceil(tempo.item())
        else:
            self.tempo = self.TEMPO_INPUT

        self.TELEPORT_COOLDOWN_FRAMES = self.math.floor(
            (60 / self.tempo) * self.UPDATE_HZ / self.TELEPORT_COOLDOWN
        )

        S = self.np.abs(self.librosa.stft(y))
        freqs = self.librosa.fft_frequencies(sr=sr)
        bass = self.np.mean(S[freqs < 150, :], axis=0)

        denom = bass.max() - bass.min()
        self.bass_energy = bass / denom if denom > 0 else self.np.zeros_like(bass)

        self.rms = self.librosa.feature.rms(y=y)[0]
        self.rms_times = self.librosa.frames_to_time(
            self.np.arange(len(self.rms)), sr=sr
        )

        self.HALF_BEAT_FRAMES = self.math.floor((60 / self.tempo) * self.UPDATE_HZ / 2)

        print(f"Finished analyzing\n{'Detected' if self.tempo <= 0 else 'Chosen'} BPM: {self.tempo}\nSetting up windows...")

    # -------------------------------------------------
    # Utility helpers

    def random_hex_color(self):
        return "#{:02X}{:02X}{:02X}".format(
            self.random.randint(0,255),
            self.random.randint(0,255),
            self.random.randint(0,255)
        )

    def hex_to_rgb(self, c):
        c = c.lstrip("#")
        return self.np.array([int(c[i:i+2],16) for i in (0,2,4)], dtype=float)

    def rgb_to_hex(self, rgb):
        return "#{:02X}{:02X}{:02X}".format(*rgb.astype(int))

    def get_window_pos(self, win):
        geo = win.geometry().split("+")
        return [int(geo[1]), int(geo[2])]

    def extract_and_display_image(self, audio_file, canvas):
        try:
            tags = self.ID3(audio_file)
            for tag in tags.getall("APIC"):
                img = self.Image.open(self.BytesIO(tag.data))
                img = img.resize((self.SQUARE_SIZE, self.SQUARE_SIZE), self.Image.Resampling.LANCZOS)
                tk_img = self.ImageTk.PhotoImage(img)

                canvas.image = tk_img
                canvas.create_image(0, 0, anchor="nw", image=tk_img)
                return True
        except Exception:
            pass
        return False

    # -------------------------------------------------
    # Dancer windows + flash

    def make_dancer_window(self):
        win = self.tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.geometry(f"{self.SQUARE_SIZE}x{self.SQUARE_SIZE}+0+0")

        canvas = self.tk.Canvas(
            win,
            width=self.SQUARE_SIZE,
            height=self.SQUARE_SIZE,
            highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)
        return win, canvas

    def trigger_flash(self, canvas, color="#FFFFFF"):
        return canvas.create_rectangle(
            0, 0, self.SQUARE_SIZE, self.SQUARE_SIZE,
            fill=color, outline=""
        )

    def update_flash(self, canvas, rect, timer):
        if rect is None:
            return None, 0
        timer -= 1
        if timer <= 0:
            canvas.delete(rect)
            return None, 0
        return rect, timer

    # -------------------------------------------------
    # Windows + audio

    def setup_windows(self):
        self.root = self.tk.Tk()
        self.root.title(self.AUDIO_FILE.rsplit(".",1)[0])
        self.root.config(bg="#000000")
        self.root.resizable(False, False)

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.BASE_X = sw//2 - self.W_WIDTH//2
        self.BASE_Y = sh//2 - self.W_HEIGHT//2

        self.DEFAULT_D1_POS = [int(self.BASE_X/1.55), int(self.BASE_Y*1.15)]
        self.DEFAULT_D2_POS = [int(self.BASE_X*1.8), int(self.BASE_Y*1.15)]

        self.root.geometry(f"{self.W_WIDTH}x{self.W_HEIGHT}+{self.BASE_X}+{self.BASE_Y}")

        self.dancer1, self.canvas1 = self.make_dancer_window()
        self.dancer2, self.canvas2 = self.make_dancer_window()

        for d in (self.dancer1, self.dancer2):
            d.overrideredirect(True)
            d.config(bg="#a00000")

        for c in (self.canvas1, self.canvas2):
            c.config(bg="#a00000")

        self.dancer1.geometry(f"+{self.DEFAULT_D1_POS[0]}+{self.DEFAULT_D1_POS[1]}")
        self.dancer2.geometry(f"+{self.DEFAULT_D2_POS[0]}+{self.DEFAULT_D2_POS[1]}")

        print("\nFinished setting up windows\nWaiting for user input on main window...")

    def setup_audio(self):
        self.pygame.mixer.pre_init(44100, -16, 2, 512)
        self.pygame.mixer.init()
        self.pygame.mixer.music.load(self.AUDIO_FILE)
        self.pygame.mixer.music.set_volume(self.VOLUME)

    # -------------------------------------------------
    # State

    def setup_state(self):
        self.angle = 0.0
        self.jump_velocity = 0.0
        self.orbit_radius = self.ORBIT_RADIUS

        # Swing / orbit
        self.angle_accumulator = 0.0
        self.speed_boost_timer = 0
        self.gate_cooldown_timer = 0

        self.orbit_radius_current = self.ORBIT_RADIUS
        self.orbit_radius_target = self.ORBIT_RADIUS

        # Teleport / easing
        self.teleport_timer = 0
        self.speed_boost_timer = 0
        self.TELEPORT_EASE_FRAMES = max(3, self.UPDATE_HZ // 15)

        self.teleport_d1_start = [0, 0]
        self.teleport_d2_start = [0, 0]
        self.teleport_d1_target = [0, 0]
        self.teleport_d2_target = [0, 0]

        # Background / dancer colors
        self.bg_color = self.np.array([0, 0, 0], dtype=float)
        self.bg_target = self.np.array([0, 0, 0], dtype=float)
        self.bg_dancer1_color = self.np.array([0, 0, 0], dtype=float)
        self.bg_dancer1_target = self.np.array([160, 0, 0], dtype=float)
        self.bg_dancer2_color = self.np.array([0, 0, 0], dtype=float)
        self.bg_dancer2_target = self.np.array([160, 0, 0], dtype=float)
        self.bg_dancer1_teleport_target = self.np.array([0, 0, 0], dtype=float)
        self.bg_dancer2_teleport_target = self.np.array([0, 0, 0], dtype=float)

        # Images
        self.has_image1 = False
        self.has_image2 = False

        self.FLASH_FRAMES = max(3, self.UPDATE_HZ // 17)
        self.flash_timer_1 = 0
        self.flash_timer_2 = 0
        self.flash_rect_1 = None
        self.flash_rect_2 = None

        self.is_running = False

    # -------------------------------------------------
    # Main update loop

    def update_loop(self):
        # If music stopped -> reset
        if not self.pygame.mixer.music.get_busy():
            self.angle_accumulator = 0.0
            self.jump_velocity = 0.0
            self.orbit_radius_current = self.ORBIT_RADIUS
            self.orbit_radius_target = self.ORBIT_RADIUS
            self.speed_boost_timer = 0
            self.gate_cooldown_timer = 0
            self.teleport_timer = 0
            self.TELEPORT_EASE_FRAMES = max(3, self.UPDATE_HZ // 15)
            self.teleport_d1_start = [0, 0]
            self.teleport_d2_start = [0, 0]
            self.teleport_d1_target = [0, 0]
            self.teleport_d2_target = [0, 0]

            self.dancer1.geometry(f"+{self.DEFAULT_D1_POS[0]}+{self.DEFAULT_D1_POS[1]}")
            self.dancer2.geometry(f"+{self.DEFAULT_D2_POS[0]}+{self.DEFAULT_D2_POS[1]}")

            self.root.geometry(f"+{self.BASE_X}+{self.BASE_Y}")
            self.root.config(bg="#000000")

            self.is_running = False
            self.start_button.pack(expand=True)
            return

        t = self.pygame.mixer.music.get_pos() / 1000
        rms_idx = min(self.np.searchsorted(self.rms_times, t), len(self.rms)-1)
        bass_idx = min(rms_idx, len(self.bass_energy) - 1)

        # -------- ADAPTIVE GATE --------
        if rms_idx < len(self.rms):
            window = self.rms[max(0, rms_idx - 20):rms_idx + 1]
            noise_floor = self.np.median(window) if len(window) else self.rms[rms_idx]
        else:
            noise_floor = self.rms[-1]

        adaptive_gate = noise_floor * self.GATE_MULTIPLIER

        is_gated = (
            rms_idx < len(self.rms)
            and self.rms[rms_idx] < adaptive_gate
            and (noise_floor - self.rms[rms_idx]) > self.MIN_GATE_DROP
        )

        # -------- SWING --------
        swing_freq = (self.tempo / 2) / 60
        base_speed = swing_freq / self.UPDATE_HZ
        speed_multiplier = 1.0

        if self.speed_boost_timer > 0:
            speed_lerp = self.speed_boost_timer / self.HALF_BEAT_FRAMES
            speed_multiplier = 1 + (self.BASS_SPEED_BOOST - 1) * speed_lerp
            self.speed_boost_timer -= 1

        self.angle_accumulator += base_speed * speed_multiplier

        raw_sine = self.math.sin(t * swing_freq * 2 * self.math.pi)
        raw_fast = self.math.sin(t * (self.tempo / 1.4 / 60) * 1.5 * self.math.pi)

        swing_x = 150 * (abs(raw_sine) ** 0.5 * (1 if raw_sine > 0 else -1))
        swing_y = 100 * (abs(raw_fast) ** 0.6 * (1 if raw_fast <= 0 else -1))

        # -------- BASS then ORBIT RADIUS --------
        bass_strength = self.bass_energy[bass_idx]

        if bass_strength > 0.65:
            self.orbit_radius_current = self.ORBIT_RADIUS * (
                1 - bass_strength * (1 - self.BASS_RADIUS_PULL)
            )
            self.orbit_radius_target = self.ORBIT_RADIUS
            self.boost_timer = self.HALF_BEAT_FRAMES

        if getattr(self, 'boost_timer', 0) > 0:
            lerp = self.boost_timer / self.HALF_BEAT_FRAMES
            self.orbit_radius_current = (
                self.orbit_radius_target * (1 - lerp) +
                self.orbit_radius_current * lerp
            )
            self.boost_timer -= 1
        else:
            self.orbit_radius_current = (
                self.orbit_radius_current * 0.9 +
                self.ORBIT_RADIUS * 0.1
            )

        # -------- BASS then SPEED BOOST --------
        if bass_strength > self.BASS_SPEED_THRESHOLD:
            self.speed_boost_timer = self.HALF_BEAT_FRAMES

        # -------- BASS then JUMP --------
        if bass_strength > self.JUMP_BASS_THRESHOLD and self.jump_velocity > -self.JUMP_REARM_VELOCITY:
            self.jump_velocity = self.MAIN_WINDOW_JUMP
            self.bg_color[:] = [255, 255, 255]
            self.bg_target[:] = [0, 0, 0]

            if not self.has_image1:
                self.bg_dancer1_target[:] = self.bg_dancer1_teleport_target
                self.bg_dancer1_color[:] = [255, 255, 255]
            else:
                if self.flash_rect_1:
                    self.canvas1.delete(self.flash_rect_1)
                self.flash_rect_1 = self.trigger_flash(self.canvas1)
                self.flash_timer_1 = self.FLASH_FRAMES

            if not self.has_image2:
                self.bg_dancer2_target[:] = self.bg_dancer2_teleport_target
                self.bg_dancer2_color[:] = [255, 255, 255]
            else:
                if self.flash_rect_2:
                    self.canvas2.delete(self.flash_rect_2)
                self.flash_rect_2 = self.trigger_flash(self.canvas2)
                self.flash_timer_2 = self.FLASH_FRAMES

        cx = self.math.cos(self.angle_accumulator) * self.orbit_radius_current
        cy = self.math.sin(self.angle_accumulator) * self.orbit_radius_current

        center_x = self.BASE_X + self.W_WIDTH // 3.6 + swing_x
        center_y = self.BASE_Y + self.W_HEIGHT // 3.6 + swing_y

        # -------- JUMP PHYSICS --------
        self.jump_velocity += self.JUMP_GRAVITY
        self.jump_velocity *= self.JUMP_DAMPING

        jump_offset = self.jump_velocity
        if abs(jump_offset) < 3.0:
            jump_offset = 0.0

        self.root.geometry(
            f"+{int(self.BASE_X + swing_x)}+{int(self.BASE_Y + swing_y + jump_offset)}"
        )

        # -------- TELEPORT --------
        if is_gated and self.gate_cooldown_timer == 0:
            max_x = self.root.winfo_screenwidth() - self.SQUARE_SIZE
            max_y = self.root.winfo_screenheight() - self.SQUARE_SIZE

            self.teleport_d1_start[:] = self.get_window_pos(self.dancer1)
            self.teleport_d2_start[:] = self.get_window_pos(self.dancer2)

            self.teleport_d1_target[:] = [self.random.randint(0, max_x), self.random.randint(0, max_y)]
            self.teleport_d2_target[:] = [self.random.randint(0, max_x), self.random.randint(0, max_y)]

            self.teleport_timer = self.TELEPORT_EASE_FRAMES
            self.gate_cooldown_timer = self.TELEPORT_COOLDOWN_FRAMES

            self.bg_color[:] = self.hex_to_rgb(self.random_hex_color())
            self.bg_target[:] = [0, 0, 0]

            if not self.has_image1:
                self.bg_dancer1_target[:] = self.hex_to_rgb(self.random_hex_color())
                self.bg_dancer1_teleport_target[:] = self.bg_dancer1_target
            else:
                if self.flash_rect_1:
                    self.canvas1.delete(self.flash_rect_1)
                self.flash_rect_1 = self.trigger_flash(self.canvas1, self.random_hex_color())
                self.flash_timer_1 = self.FLASH_FRAMES * 2

            if not self.has_image2:
                self.bg_dancer2_target[:] = self.hex_to_rgb(self.random_hex_color())
                self.bg_dancer2_teleport_target[:] = self.bg_dancer2_target
            else:
                if self.flash_rect_2:
                    self.canvas2.delete(self.flash_rect_2)
                self.flash_rect_2 = self.trigger_flash(self.canvas2, self.random_hex_color())
                self.flash_timer_2 = self.FLASH_FRAMES * 2

        if self.teleport_timer > 0:
            t_norm = 1 - (self.teleport_timer / self.TELEPORT_EASE_FRAMES)
            t_ease = 1 - (1 - t_norm) ** 3

            d1x = self.teleport_d1_start[0] + (self.teleport_d1_target[0] - self.teleport_d1_start[0]) * t_ease
            d1y = self.teleport_d1_start[1] + (self.teleport_d1_target[1] - self.teleport_d1_start[1]) * t_ease
            d2x = self.teleport_d2_start[0] + (self.teleport_d2_target[0] - self.teleport_d2_start[0]) * t_ease
            d2y = self.teleport_d2_start[1] + (self.teleport_d2_target[1] - self.teleport_d2_start[1]) * t_ease

            self.dancer1.geometry(f"+{int(d1x)}+{int(d1y)}")
            self.dancer2.geometry(f"+{int(d2x)}+{int(d2y)}")

            self.teleport_timer -= 1

        elif self.gate_cooldown_timer > 0:
            self.gate_cooldown_timer -= 1

        else:
            self.dancer1.geometry(f"+{int(center_x + cx)}+{int(center_y + cy)}")
            self.dancer2.geometry(f"+{int(center_x - cx)}+{int(center_y - cy)}")

        # -------- BACKGROUND FADE --------
        self.bg_color[:] += (self.bg_target - self.bg_color) * self.BG_FADE_SPEED
        self.bg_color[:] = self.np.clip(self.bg_color, 0, 255)

        self.root.config(bg=self.rgb_to_hex(self.bg_color))

        if not self.has_image1:
            self.bg_dancer1_color[:] += (self.bg_dancer1_target - self.bg_dancer1_color) * self.BG_DANCERS_FADE_SPEED
            self.bg_dancer1_color[:] = self.np.clip(self.bg_dancer1_color, 0, 255)
            self.dancer1.config(bg=self.rgb_to_hex(self.bg_dancer1_color))
        else:
            self.flash_rect_1, self.flash_timer_1 = self.update_flash(
                self.canvas1, self.flash_rect_1, self.flash_timer_1
            )

        if not self.has_image2:
            self.bg_dancer2_color[:] += (self.bg_dancer2_target - self.bg_dancer2_color) * self.BG_DANCERS_FADE_SPEED
            self.bg_dancer2_color[:] = self.np.clip(self.bg_dancer2_color, 0, 255)
            self.dancer2.config(bg=self.rgb_to_hex(self.bg_dancer2_color))
        else:
            self.flash_rect_2, self.flash_timer_2 = self.update_flash(
                self.canvas2, self.flash_rect_2, self.flash_timer_2
            )

        self.root.after(1000 // self.UPDATE_HZ, self.update_loop)

    # -------------------------------------------------
    # Start

    def start(self):
        if self.is_running:
            return

        self.is_running = True

        self.pygame.mixer.music.play()
        self.has_image1 = self.extract_and_display_image(self.AUDIO_FILE, self.canvas1)
        self.has_image2 = self.extract_and_display_image(self.AUDIO_FILE, self.canvas2)

        if not self.has_image1:
            self.canvas1.pack_forget()

        if not self.has_image2:
            self.canvas2.pack_forget()

        self.start_button.pack_forget()
        self.update_loop()

    def run(self):
        self.start_button = self.tk.Button(
            self.root,
            text="Start the window dance",
            command=self.start
        )
        self.start_button.pack(expand=True)
        self.root.mainloop()

if __name__ == "__main__":
    WindowDance().run()