"""Non-blocking computer-speaker warning tone."""
from __future__ import annotations

import math
import queue
import struct
import threading


class AudioAlert:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._muted = False
        self._volume = 0.60
        self._pygame = None
        self._sound_advisory = None
        self._sound_critical = None
        self._speech_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._speech_stop = threading.Event()
        self._speech_available = False
        self._speech_thread: threading.Thread | None = None
        try:
            import pyttsx3  # noqa: F401 - initialized inside its owning worker thread
            self._speech_available = True
            self._speech_thread = threading.Thread(target=self._speech_loop, name="warning-speech", daemon=True)
            self._speech_thread.start()
        except Exception:
            pass
        try:
            import pygame
            pygame.mixer.init()
            self._pygame = pygame
            sample_rate = 22050

            # 1. Synthesize Advisory Tone (Soft 550Hz chime)
            dur_adv = 0.25
            n_adv = int(sample_rate * dur_adv)
            buf_adv = bytearray()
            for i in range(n_adv):
                env = min(1.0, i / 200, (n_adv - i) / 2000)
                val = int(8000 * env * math.sin(2 * math.pi * 550 * i / sample_rate))
                buf_adv.extend(struct.pack("<h", val))
            self._sound_advisory = pygame.mixer.Sound(buffer=bytes(buf_adv))
            self._sound_advisory.set_volume(self._volume * 0.7)

            # 2. Synthesize Critical Tone (High-urgency 1100Hz pulsed double-alarm)
            dur_crit = 0.45
            n_crit = int(sample_rate * dur_crit)
            buf_crit = bytearray()
            for i in range(n_crit):
                t = i / sample_rate
                # Pulsing 12Hz on-off gate
                gate = 1.0 if (int(t * 12) % 2 == 0) else 0.0
                env = min(1.0, (i % 1800) / 150) * gate
                val = int(12000 * env * math.sin(2 * math.pi * 1100 * i / sample_rate))
                buf_crit.extend(struct.pack("<h", val))
            self._sound_critical = pygame.mixer.Sound(buffer=bytes(buf_crit))
            self._sound_critical.set_volume(self._volume)
            self._sound = self._sound_advisory
        except Exception:
            self._pygame = None

    @property
    def available(self) -> bool:
        return self._pygame is not None

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def speech_available(self) -> bool:
        return self._speech_available

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self.stop()
            self._clear_speech_queue()

    def _clear_speech_queue(self) -> None:
        try:
            while True:
                self._speech_queue.get_nowait()
        except queue.Empty:
            pass

    def _speech_loop(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            while not self._speech_stop.is_set():
                try:
                    phrase = self._speech_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not self._muted:
                    engine.setProperty("volume", self._volume)
                    engine.say(phrase)
                    engine.runAndWait()
        except Exception:
            self._speech_available = False

    def speak_warning(self, object_name: str, side: str) -> None:
        """Speak a concise, one-time message for a new zone-entry event."""
        if not self._speech_available or self._muted:
            return
        phrase = f"There is a {object_name.lower()} on the {side} side. Caution."
        try:
            self._speech_queue.put_nowait(phrase)
        except queue.Full:
            try:
                pending = self._speech_queue.get_nowait()
                self._speech_queue.put_nowait(f"{pending} Also, {phrase}")
            except (queue.Empty, queue.Full):
                pass

    def set_volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, float(value)))
        if self._sound_advisory:
            self._sound_advisory.set_volume(self._volume * 0.7)
        if self._sound_critical:
            self._sound_critical.set_volume(self._volume)

    def play_advisory(self) -> None:
        if self._pygame and not self._muted and self._sound_advisory:
            try:
                self._sound_advisory.play()
            except Exception:
                pass

    def play_critical(self) -> None:
        if self._pygame and not self._muted and self._sound_critical:
            try:
                self._sound_critical.play()
            except Exception:
                pass

    def play(self, critical: bool = False) -> None:
        if critical:
            self.play_critical()
        else:
            self.play_advisory()

    def stop(self) -> None:
        if self._pygame:
            try:
                self._pygame.mixer.stop()
            except Exception:
                pass
