"""DRISHTI · Intelligent Blind-Spot Warning System for Heavy Vehicles.

An advanced automotive ADAS demonstration featuring:
- Real-time YOLOv8n + ByteTrack Road-User Tracking (Vehicles, Pedestrians, Two-Wheelers)
- Dual-Zone (Left & Right Blind Zones) geometric monitoring with adjustable width
- Automotive Two-Stage Warning Logic:
    * Stage 1 (Amber Advisory): Road user in blind zone with turn signal OFF
    * Stage 2 (Flashing Red Critical): Collision alarm when turn signal is ACTIVE toward blind zone
- Vehicle Speed Simulation & Stationary Noise Suppression (automatic mute at 0 km/h / red lights)
- Directional Motion Velocity Vectors (Approaching vs. Receding vehicles)
- Night & Fog Vision Enhancement (OpenCV CLAHE + Adaptive Gamma correction)
- Automated Black-Box Incident Snapshots (saving photo evidence of alerts)
- Interactive Snapshot Viewer in History Log
- Active Camera Auto-Scanner
- Driver Full-Screen Kiosk Mode (F11)
- Integrated Safety Audit & Next-Steps PDF Report Generator
"""
from __future__ import annotations

import math
import os
import queue
import sys
import time
import uuid
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from audio_alert import AudioAlert
from database import DetectionStore
from detector import (
    Detection,
    DetectionWorker,
    VEHICLES,
    TWO_WHEELERS,
    scan_available_cameras,
)
import generate_pdf_report

BASE_DIR = Path(__file__).resolve().parent
DEMO_VIDEO = BASE_DIR / "assets" / "demo_road.webm"
ASSETS_DIR = BASE_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Futuristic dark dashboard palette.
BG_MAIN = "#080f1a"
BG_HEADER = "#0c1726"
PANEL_BG = "#0f1c2d"
PANEL_ALT = "#15263c"
PANEL_BORDER = "#1b304c"
PANEL_BORDER_HOVER = "#2d527c"
TEXT_WHITE = "#f8fafc"
TEXT_LIGHT = "#e2e8f0"
TEXT_MUTED = "#8fa3be"
TEXT_ACCENT = "#38bdf8"
ACCENT_ORANGE = "#f59e0b"
ACCENT_ORANGE_HOVER = "#fbbf24"
ALERT_RED = "#ef4444"
ALERT_RED_BG = "#2a0d15"
ALERT_RED_BORDER = "#dc2626"
ALERT_RED_PULSE = "#ff4d6d"
ADVISORY_AMBER = "#f59e0b"
ADVISORY_AMBER_BG = "#2a1e0b"
ADVISORY_AMBER_BORDER = "#d97706"
SAFE_GREEN = "#10b981"
SAFE_GREEN_BG = "#0c241c"
SAFE_GREEN_BORDER = "#059669"

# Default Warning Zones (Normalized Coordinates)
DEFAULT_ZONES: dict[str, list[tuple[float, float]]] = {
    "left": [
        (0.01, 0.32),
        (0.28, 0.30),
        (0.38, 0.95),
        (0.01, 0.95),
    ],
    "right": [
        (0.53, 0.34),
        (0.98, 0.36),
        (0.98, 0.95),
        (0.48, 0.95),
    ],
}


def make_card_interactive(
    widget: ctk.CTkFrame,
    normal_bg: str,
    hover_bg: str,
    normal_border: str,
    hover_border: str,
    border_width_hover: int | None = None,
    on_click=None,
) -> None:
    """Attaches robust hover transitions, glowing borders, and hand cursor to a card."""
    def check_contains() -> bool:
        try:
            x, y = widget.winfo_pointerx(), widget.winfo_pointery()
            rx, ry = widget.winfo_rootx(), widget.winfo_rooty()
            rw, rh = widget.winfo_width(), widget.winfo_height()
            return rx <= x <= rx + rw and ry <= y <= ry + rh
        except Exception:
            return False

    def on_enter(_event) -> None:
        widget.configure(fg_color=hover_bg, border_color=hover_border)
        if border_width_hover is not None:
            widget.configure(border_width=border_width_hover)

    def on_leave(_event) -> None:
        if not check_contains():
            widget.configure(fg_color=normal_bg, border_color=normal_border)
            if border_width_hover is not None:
                widget.configure(border_width=1)

    def bind_tree(w) -> None:
        w.bind("<Enter>", on_enter, add="+")
        w.bind("<Leave>", on_leave, add="+")
        if on_click:
            w.bind("<Button-1>", lambda _e: on_click(), add="+")
        try:
            w.configure(cursor="hand2")
        except Exception:
            pass
        for ch in w.winfo_children():
            bind_tree(ch)

    bind_tree(widget)


class BlindSpotApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DRISHTI · Intelligent Blind-Spot Warning System (Automotive ADAS)")
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(1560, max(1180, screen_w - 32))
        height = min(960, max(720, screen_h - 60))
        self.geometry(f"{width}x{height}")
        self.minsize(1120, 680)
        self.configure(fg_color=BG_MAIN)

        self.store = DetectionStore(BASE_DIR / "data" / "detections.db")
        self.audio = AudioAlert()
        self.messages: queue.Queue = queue.Queue(maxsize=4)
        self.worker: DetectionWorker | None = None
        self.video_path: Path | None = None
        self.source_name = "Camera 0"
        self._feed_text = "READY"
        self.running = False
        self.paused = False
        self.loop_video = True
        self.alert_active = False
        self.alert_is_critical = False
        self.active_alert_zone = ""
        self.warning_events = self.store.alert_total()
        self.session_id = uuid.uuid4().hex
        self._hud_tick = 0
        self._last_ui_frame_time = 0.0
        self._ui_fps = 0.0
        self._last_audio_signature = None
        self.kiosk_active = False

        # ADAS Vehicle Telemetry Simulation
        self.indicator_left = False
        self.indicator_right = False
        self.indicator_blink = False
        self.vehicle_speed = 45  # km/h
        self.night_mode = False

        # Dual Zone Configuration
        self.zone_mode = "both"  # "left", "right", or "both"
        self.zone_width_factor = 0.70
        self.zone_sensitivity_pct = 65
        self.confidence_val = 0.35
        self.zones: dict[str, list[tuple[float, float]]] = {
            "left": list(DEFAULT_ZONES["left"]),
            "right": list(DEFAULT_ZONES["right"]),
        }
        self.zone_editing = False
        self.drag_vertex: tuple[str, int] | None = None
        self.display_geometry: tuple[float, float, float] | None = None
        self.frame_size = (0, 0)
        self._video_image = None

        # Tracking state
        self.known_tracks: set[tuple[int, int]] = set()
        self.zone_tracks: dict[tuple[int, int], Detection] = {}
        self.last_metrics = {"vehicles": 0, "pedestrians": 0, "two_wheelers": 0}
        self._last_snapshot_time: dict[int, float] = {}

        # Load graphical assets
        self._load_assets()

        # Build UI layout
        self._build_ui()
        self._refresh_history_tab()

        # Keyboard hotkeys for driver cockpit simulation
        self._bind_hotkeys()

        # Start background polling, live clock, and animated HUD heartbeat
        self.after(35, self._poll)
        self.after(100, self._maximize)
        self._tick_clock()
        self._pulse_hud()

        self.protocol("WM_DELETE_WINDOW", self._close)

    def _maximize(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

    def _bind_hotkeys(self) -> None:
        """Driver cockpit hotkeys for seamless demo controls."""
        self.bind("<Key-a>", lambda e: self.toggle_left_indicator())
        self.bind("<Key-A>", lambda e: self.toggle_left_indicator())
        self.bind("<Left>", lambda e: self.toggle_left_indicator())
        self.bind("<Key-d>", lambda e: self.toggle_right_indicator())
        self.bind("<Key-D>", lambda e: self.toggle_right_indicator())
        self.bind("<Right>", lambda e: self.toggle_right_indicator())
        self.bind("<Key-s>", lambda e: self.clear_indicators())
        self.bind("<Key-S>", lambda e: self.clear_indicators())
        self.bind("<Down>", lambda e: self.clear_indicators())
        self.bind("<F11>", lambda e: self.toggle_kiosk_mode())
        self.bind("<Escape>", lambda e: self._exit_kiosk_if_active())

    def _load_assets(self) -> None:
        self.img_drishti_logo = None
        self.img_india_banner = None
        self.icon_car = None
        self.icon_pedestrian = None
        self.icon_motorcycle = None
        self.icon_alert = None
        self.icon_speaker = None
        self.icon_zone_left = None
        self.icon_zone_right = None
        self.icon_zone_both = None

        try:
            p_logo = ASSETS_DIR / "drishti_logo.png"
            if p_logo.exists():
                pil_logo = Image.open(p_logo)
                self.img_drishti_logo = ctk.CTkImage(light_image=pil_logo, dark_image=pil_logo, size=(460, 52))

            p_banner = ASSETS_DIR / "india_banner.png"
            if p_banner.exists():
                pil_banner = Image.open(p_banner)
                self.img_india_banner = ctk.CTkImage(light_image=pil_banner, dark_image=pil_banner, size=(340, 132))

            p_car = ICONS_DIR / "car.png"
            if p_car.exists():
                self.icon_car = ctk.CTkImage(light_image=Image.open(p_car), size=(24, 24))

            p_ped = ICONS_DIR / "pedestrian.png"
            if p_ped.exists():
                self.icon_pedestrian = ctk.CTkImage(light_image=Image.open(p_ped), size=(24, 24))

            p_moto = ICONS_DIR / "motorcycle.png"
            if p_moto.exists():
                self.icon_motorcycle = ctk.CTkImage(light_image=Image.open(p_moto), size=(24, 24))

            p_alert = ICONS_DIR / "alert.png"
            if p_alert.exists():
                self.icon_alert = ctk.CTkImage(light_image=Image.open(p_alert), size=(34, 34))

            p_speaker = ICONS_DIR / "speaker.png"
            if p_speaker.exists():
                self.icon_speaker = ctk.CTkImage(light_image=Image.open(p_speaker), size=(30, 30))

            p_zl = ICONS_DIR / "zone_left.png"
            if p_zl.exists():
                self.icon_zone_left = ctk.CTkImage(light_image=Image.open(p_zl), size=(22, 22))

            p_zr = ICONS_DIR / "zone_right.png"
            if p_zr.exists():
                self.icon_zone_right = ctk.CTkImage(light_image=Image.open(p_zr), size=(22, 22))

            p_zb = ICONS_DIR / "zone_both.png"
            if p_zb.exists():
                self.icon_zone_both = ctk.CTkImage(light_image=Image.open(p_zb), size=(22, 22))
        except Exception as exc:
            print("Asset loading warning:", exc)

    def _tick_clock(self) -> None:
        now = datetime.now()
        stamp = now.strftime("%d %b %Y | %H:%M:%S")
        if hasattr(self, "clock_label"):
            self.clock_label.configure(text=stamp)
        self.after(1000, self._tick_clock)

    def _pulse_hud(self) -> None:
        """Lightweight HUD pulsing animation for live status, blinking turn signals, and critical alerts."""
        self._hud_tick += 1
        self.indicator_blink = (self._hud_tick % 2 == 0)

        # 1. Blink Left Turn Signal if active
        if hasattr(self, "btn_turn_left"):
            if self.indicator_left:
                self.btn_turn_left.configure(
                    fg_color=ACCENT_ORANGE if self.indicator_blink else "#15263c",
                    text_color="#080f1b" if self.indicator_blink else ACCENT_ORANGE,
                )
            else:
                self.btn_turn_left.configure(fg_color="#15263c", text_color=TEXT_MUTED)

        # 2. Blink Right Turn Signal if active
        if hasattr(self, "btn_turn_right"):
            if self.indicator_right:
                self.btn_turn_right.configure(
                    fg_color=ACCENT_ORANGE if self.indicator_blink else "#15263c",
                    text_color="#080f1b" if self.indicator_blink else ACCENT_ORANGE,
                )
            else:
                self.btn_turn_right.configure(fg_color="#15263c", text_color=TEXT_MUTED)

        # 3. Pulse live dot in camera header
        if hasattr(self, "live_dot") and self.running:
            glow_colors = ["#10b981", "#10b981", "#10b981", "#059669"]
            self.live_dot.configure(text_color=glow_colors[self._hud_tick % len(glow_colors)])

        # 4. Pulse alert card when alert is active
        if hasattr(self, "alert_card") and self.alert_active:
            if self.alert_is_critical:
                alert_borders = [ALERT_RED_BORDER, ALERT_RED_PULSE, ALERT_RED_BORDER, "#ef4444"]
                alert_bgs = [ALERT_RED_BG, "#15263c", ALERT_RED_BG, "#15263c"]
            else:
                alert_borders = [ADVISORY_AMBER_BORDER, ACCENT_ORANGE_HOVER, ADVISORY_AMBER_BORDER, "#8A5A12"]
                alert_bgs = [ADVISORY_AMBER_BG, "#15263c", ADVISORY_AMBER_BG, "#15263c"]

            idx = self._hud_tick % len(alert_borders)
            self.alert_card.configure(border_color=alert_borders[idx], fg_color=alert_bgs[idx])
            if hasattr(self, "alert_speaker_btn"):
                self.alert_speaker_btn.configure(fg_color="#15263c" if idx % 2 == 0 else "#15263c")

        self.after(380, self._pulse_hud)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Header (ISRO, DRISHTI, Skyline & Tricolor, System Status, Clock, Kiosk mode)
        self._build_header()

        # 2. Tab Navigation Bar
        self._build_nav_tabs()

        # 3. Main Views Container
        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.grid(row=2, column=0, sticky="nsew", padx=16, pady=(6, 14))
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # Build the tab views
        self.tabs: dict[str, ctk.CTkFrame] = {}
        self._build_live_monitor_view()
        self._build_analytics_view()
        self._build_history_view()
        self._build_settings_view()
        self._build_about_view()

        # Show Live Monitor initially
        self._switch_tab("Live Monitor")

    # ==========================================
    # HEADER & NAVIGATION
    # ==========================================
    def _build_header(self) -> None:
        self.header_frame = ctk.CTkFrame(self, fg_color=BG_HEADER, corner_radius=0, height=72)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        # Left Branding: Logo + ISRO + DRISHTI
        brand_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        brand_frame.grid(row=0, column=0, sticky="w", padx=(16, 10), pady=8)

        if self.img_drishti_logo:
            ctk.CTkLabel(brand_frame, text="", image=self.img_drishti_logo).pack(side="left")
        else:
            ctk.CTkLabel(brand_frame, text="DRISHTI", text_color=ACCENT_ORANGE,
                         font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")
            ctk.CTkLabel(brand_frame, text="  Intelligent Blind-Spot Warning System",
                         text_color=TEXT_WHITE, font=ctk.CTkFont(size=14)).pack(side="left")

        # Right Indicators (System Status pill, Kiosk mode button, Real-time Clock)
        right_info = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        right_info.grid(row=0, column=1, sticky="e", padx=(0, 18), pady=8)

        # Kiosk Mode Toggle Button
        btn_kiosk = ctk.CTkButton(
            right_info, text="⛶  Driver Kiosk (F11)", width=130, height=30, corner_radius=8,
            fg_color="#15263c", hover_color="#15263c", text_color=TEXT_ACCENT,
            border_width=1, border_color="#1b304c", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self.toggle_kiosk_mode,
        )
        btn_kiosk.pack(side="left", padx=(0, 10), pady=4)

        # System Online Pill with interactive hover
        self.status_pill = ctk.CTkFrame(right_info, fg_color="#0c241c", corner_radius=14,
                                        border_width=1, border_color="#10b981")
        self.status_pill.pack(side="left", padx=(0, 10), pady=4)
        ctk.CTkLabel(self.status_pill, text="●  System Online", text_color="#10b981",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(padx=14, pady=4)
        make_card_interactive(self.status_pill, "#0c241c", "#0f2e24", "#10b981", "#10b981", border_width_hover=2)

        # Live Clock with hover glow
        clock_box = ctk.CTkFrame(right_info, fg_color="#15263c", corner_radius=10,
                                 border_width=1, border_color="#1b304c")
        clock_box.pack(side="left", pady=4)
        self.clock_label = ctk.CTkLabel(clock_box, text="-- --- ---- | --:--:--", text_color=TEXT_LIGHT,
                                        font=ctk.CTkFont(size=11, weight="bold"))
        self.clock_label.pack(padx=12, pady=4)
        make_card_interactive(clock_box, "#15263c", "#15263c", "#1b304c", TEXT_ACCENT)

    def _build_nav_tabs(self) -> None:
        self.nav_bar = ctk.CTkFrame(self, fg_color="#15263c", height=46, corner_radius=0)
        self.nav_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 0))

        self.tab_buttons: dict[str, ctk.CTkButton] = {}
        tab_specs = [
            ("Live Monitor", "📷 Live Monitor"),
            ("Analytics", "📊 Analytics"),
            ("History", "🕒 History"),
            ("Settings", "⚙ Settings"),
            ("About", "ℹ About"),
        ]

        for tab_id, tab_label in tab_specs:
            btn = ctk.CTkButton(
                self.nav_bar,
                text=tab_label,
                command=lambda t=tab_id: self._switch_tab(t),
                fg_color=PANEL_BG,
                hover_color="#15263c",
                text_color=TEXT_MUTED,
                height=34,
                corner_radius=17,
                border_width=1,
                border_color="#15263c",
                cursor="hand2",
                font=ctk.CTkFont(size=11, weight="bold"),
            )
            btn.pack(side="left", padx=5, pady=5)

            def make_tab_hover(b=btn, tid=tab_id):
                def on_enter(_e):
                    if getattr(self, "_active_tab", "") != tid:
                        b.configure(fg_color="#15263c", text_color=TEXT_WHITE, border_color=TEXT_ACCENT)
                    else:
                        b.configure(fg_color=ACCENT_ORANGE_HOVER, border_color="#f8fafc")

                def on_leave(_e):
                    if getattr(self, "_active_tab", "") != tid:
                        b.configure(fg_color=PANEL_BG, text_color=TEXT_MUTED, border_color="#15263c")
                    else:
                        b.configure(fg_color=ACCENT_ORANGE, border_color=ACCENT_ORANGE)

                b.bind("<Enter>", on_enter)
                b.bind("<Leave>", on_leave)

            make_tab_hover()
            self.tab_buttons[tab_id] = btn

    def _switch_tab(self, tab_id: str) -> None:
        self._active_tab = tab_id
        for tid, frame in self.tabs.items():
            if tid == tab_id:
                frame.grid(row=0, column=0, sticky="nsew")
                self.tab_buttons[tid].configure(
                    fg_color=ACCENT_ORANGE,
                    text_color="#080f1b",
                    border_color=ACCENT_ORANGE,
                    hover_color=ACCENT_ORANGE_HOVER,
                )
            else:
                frame.grid_forget()
                self.tab_buttons[tid].configure(
                    fg_color=PANEL_BG,
                    text_color=TEXT_MUTED,
                    border_color="#15263c",
                    hover_color="#15263c",
                )

        if tab_id == "Analytics":
            self._update_analytics_charts()
        elif tab_id == "History":
            self._refresh_history_tab()

    # ==========================================
    # VIEW 1: LIVE MONITOR TAB (THE MAIN SCREEN)
    # ==========================================
    def _build_live_monitor_view(self) -> None:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.tabs["Live Monitor"] = view
        view.grid_columnconfigure(0, weight=68)
        view.grid_columnconfigure(1, weight=32)
        view.grid_rowconfigure(0, weight=75)
        view.grid_rowconfigure(1, weight=25)

        # ----------------------------------------------------
        # TOP ROW - LEFT: LIVE CAMERA VIEW CARD
        # ----------------------------------------------------
        cam_card = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=16,
                                border_width=1, border_color=PANEL_BORDER)
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 8))
        cam_card.grid_columnconfigure(0, weight=1)
        cam_card.grid_rowconfigure(1, weight=1)

        make_card_interactive(cam_card, PANEL_BG, PANEL_BG, PANEL_BORDER, PANEL_BORDER_HOVER, border_width_hover=2)

        # Card Header: Camera Icon + Title + Live indicator + Badges + FPS
        cam_header = ctk.CTkFrame(cam_card, fg_color="transparent")
        cam_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 6))

        h_left = ctk.CTkFrame(cam_header, fg_color="transparent")
        h_left.pack(side="left")
        ctk.CTkLabel(h_left, text="📷", font=ctk.CTkFont(size=14)).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(h_left, text="Live Camera View", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.live_dot = ctk.CTkLabel(h_left, text="●", text_color=SAFE_GREEN, font=ctk.CTkFont(size=12))
        self.live_dot.pack(side="left", padx=(8, 0))

        self.feed_source_badge = ctk.CTkLabel(h_left, text="Camera 0", text_color=TEXT_MUTED,
                                              font=ctk.CTkFont(size=10))
        self.feed_source_badge.pack(side="left", padx=(10, 0))

        # Night vision badge indicator
        self.night_badge = ctk.CTkLabel(h_left, text="🌙 Night Vision", text_color="#38bdf8",
                                        fg_color="#15263c", corner_radius=6, font=ctk.CTkFont(size=9, weight="bold"))
        self.night_badge.pack(side="left", padx=(8, 0))
        self.night_badge.pack_forget()  # Hidden by default

        # FPS & Latency Badge (Neon Green Pill with hover)
        self.fps_pill = ctk.CTkFrame(cam_header, fg_color="#0c241c", corner_radius=10,
                                     border_width=1, border_color="#10b981")
        self.fps_pill.pack(side="right")
        self.fps_label = ctk.CTkLabel(self.fps_pill, text="AI 0.0 · UI 0.0 FPS", text_color="#10b981",
                                      font=ctk.CTkFont(size=10, weight="bold"))
        self.fps_label.pack(padx=10, pady=2)
        make_card_interactive(self.fps_pill, "#0c241c", "#0f2e24", "#10b981", "#10b981", border_width_hover=2)

        # Video Canvas / Display Label
        self.video_view = ctk.CTkLabel(
            cam_card,
            text="\n\n  DRISHTI INTELLIGENT MONITOR  \n\n  Press Start to begin road-user detection\n  or load the bundled road demo.\n\n",
            fg_color="#060d16",
            corner_radius=5,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.video_view.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.video_view.bind("<ButtonPress-1>", self._zone_press)
        self.video_view.bind("<B1-Motion>", self._zone_drag)
        self.video_view.bind("<ButtonRelease-1>", self._zone_release)
        self.video_view.bind("<Button-3>", self._zone_remove_vertex)

        # Cockpit Controls Bar under Video (Turn Indicators & Speedometer)
        cockpit_bar = ctk.CTkFrame(cam_card, fg_color="#15263c", corner_radius=10,
                                   border_width=1, border_color="#1b304c")
        cockpit_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        cockpit_bar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Turn Signal Controls
        sig_frame = ctk.CTkFrame(cockpit_bar, fg_color="transparent")
        sig_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=6)

        ctk.CTkLabel(sig_frame, text="TURN SIGNALS (A/D):", text_color=TEXT_MUTED,
                     font=ctk.CTkFont(size=9, weight="bold")).pack(side="left", padx=(0, 6))

        self.btn_turn_left = ctk.CTkButton(
            sig_frame, text="◀  LEFT TURN", width=95, height=26, corner_radius=6,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#15263c",
            text_color=TEXT_MUTED, cursor="hand2", font=ctk.CTkFont(size=9, weight="bold"),
            command=self.toggle_left_indicator,
        )
        self.btn_turn_left.pack(side="left", padx=3)

        self.btn_turn_right = ctk.CTkButton(
            sig_frame, text="RIGHT TURN  ▶", width=95, height=26, corner_radius=6,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#15263c",
            text_color=TEXT_MUTED, cursor="hand2", font=ctk.CTkFont(size=9, weight="bold"),
            command=self.toggle_right_indicator,
        )
        self.btn_turn_right.pack(side="left", padx=3)

        btn_turn_off = ctk.CTkButton(
            sig_frame, text="⏏  OFF", width=55, height=26, corner_radius=6,
            fg_color="#15263c", hover_color="#15263c", text_color=TEXT_MUTED,
            cursor="hand2", font=ctk.CTkFont(size=9), command=self.clear_indicators,
        )
        btn_turn_off.pack(side="left", padx=3)

        # Speedometer & Noise Filter Controls
        speed_frame = ctk.CTkFrame(cockpit_bar, fg_color="transparent")
        speed_frame.grid(row=0, column=2, columnspan=2, sticky="e", padx=10, pady=6)

        ctk.CTkLabel(speed_frame, text="TRUCK SPEED:", text_color=TEXT_MUTED,
                     font=ctk.CTkFont(size=9, weight="bold")).pack(side="left", padx=(0, 4))

        self.lbl_speed_val = ctk.CTkLabel(speed_frame, text=f"{self.vehicle_speed} km/h", text_color=TEXT_WHITE,
                                          font=ctk.CTkFont(size=11, weight="bold"), width=55)
        self.lbl_speed_val.pack(side="left", padx=(0, 6))

        self.speed_slider = ctk.CTkSlider(
            speed_frame, from_=0, to=90, number_of_steps=90, width=110, height=12,
            progress_color=ACCENT_ORANGE, button_color=ACCENT_ORANGE, cursor="hand2",
            command=self._on_speed_change,
        )
        self.speed_slider.set(self.vehicle_speed)
        self.speed_slider.pack(side="left", padx=4)

        # Speed presets: 0 km/h (Red Light) and 45 km/h
        btn_idle = ctk.CTkButton(
            speed_frame, text="🛑 0 km/h", width=55, height=24, corner_radius=5,
            fg_color="#15263c", hover_color="#1b304c", text_color="#ef4444",
            cursor="hand2", font=ctk.CTkFont(size=9), command=lambda: self._set_speed_preset(0),
        )
        btn_idle.pack(side="left", padx=2)

        btn_cruise = ctk.CTkButton(
            speed_frame, text="🚚 45 km/h", width=60, height=24, corner_radius=5,
            fg_color="#15263c", hover_color="#1b304c", text_color="#10b981",
            cursor="hand2", font=ctk.CTkFont(size=9), command=lambda: self._set_speed_preset(45),
        )
        btn_cruise.pack(side="left", padx=2)

        # ----------------------------------------------------
        # TOP ROW - RIGHT: ALERT BANNER, METRICS & TIMELINE
        # ----------------------------------------------------
        dash_col = ctk.CTkFrame(view, fg_color="transparent")
        dash_col.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        dash_col.grid_columnconfigure(0, weight=1)
        dash_col.grid_rowconfigure(2, weight=1)

        # 1. Blind Spot Alert Banner (Two-Stage ADAS Display)
        self.alert_card = ctk.CTkFrame(dash_col, fg_color=SAFE_GREEN_BG, corner_radius=14,
                                       border_width=2, border_color=SAFE_GREEN_BORDER)
        self.alert_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.alert_card.grid_columnconfigure(1, weight=1)

        # Alert Left Icon Box
        self.alert_icon_label = ctk.CTkLabel(self.alert_card, text="🛡️", font=ctk.CTkFont(size=24))
        self.alert_icon_label.grid(row=0, column=0, rowspan=2, padx=(14, 10), pady=12)

        # Alert Title & Details
        self.alert_title = ctk.CTkLabel(self.alert_card, text="ALL CLEAR", text_color=SAFE_GREEN,
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.alert_title.grid(row=0, column=1, sticky="w", pady=(10, 1))

        self.alert_sub = ctk.CTkLabel(self.alert_card, text="No road users in warning zone",
                                      text_color="#a7f3d0", font=ctk.CTkFont(size=11))
        self.alert_sub.grid(row=1, column=1, sticky="w", pady=(0, 10))

        # Alert Speaker Icon with interactive hover
        self.alert_speaker_btn = ctk.CTkButton(
            self.alert_card,
            text="🔊",
            width=38,
            height=38,
            corner_radius=19,
            fg_color="#0f2e24",
            hover_color="#1f5844",
            border_width=1,
            border_color="#10b981",
            cursor="hand2",
            command=self.toggle_mute,
            font=ctk.CTkFont(size=14),
        )
        self.alert_speaker_btn.grid(row=0, column=2, rowspan=2, padx=(0, 14), pady=10)

        make_card_interactive(self.alert_card, SAFE_GREEN_BG, "#0c241c", SAFE_GREEN_BORDER, "#10b981")

        # 2. Three Metric Cards (Vehicles, Pedestrians, Two-Wheelers)
        metrics_bar = ctk.CTkFrame(dash_col, fg_color="transparent")
        metrics_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        metrics_bar.grid_columnconfigure((0, 1, 2), weight=1)

        self.card_vehicles = self._create_metric_card(metrics_bar, 0, "Vehicles", "0", "↑ +0", self.icon_car, ACCENT_ORANGE)
        self.card_pedestrians = self._create_metric_card(metrics_bar, 1, "Pedestrians", "0", "↑ +0", self.icon_pedestrian, SAFE_GREEN)
        self.card_two_wheelers = self._create_metric_card(metrics_bar, 2, "Two-Wheelers", "0", "↑ +0", self.icon_motorcycle, TEXT_ACCENT)

        # 3. Event Timeline Card
        timeline_card = ctk.CTkFrame(dash_col, fg_color=PANEL_BG, corner_radius=14,
                                     border_width=1, border_color=PANEL_BORDER)
        timeline_card.grid(row=2, column=0, sticky="nsew")
        timeline_card.grid_columnconfigure(0, weight=1)
        timeline_card.grid_rowconfigure(1, weight=1)

        t_header = ctk.CTkFrame(timeline_card, fg_color="transparent")
        t_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        ctk.CTkLabel(t_header, text="🕒  Event Timeline & Incident Vault", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")

        # View All shortcut button with glow on hover
        view_all_btn = ctk.CTkButton(
            t_header,
            text="View All →",
            width=76,
            height=24,
            corner_radius=8,
            fg_color="#15263c",
            hover_color="#15263c",
            border_width=1,
            border_color="#15263c",
            text_color=TEXT_ACCENT,
            cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._switch_tab("History"),
        )
        view_all_btn.pack(side="right")

        self.timeline_scroll = ctk.CTkScrollableFrame(timeline_card, fg_color="transparent")
        self.timeline_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self.timeline_scroll.grid_columnconfigure(1, weight=1)

        self.timeline_empty = ctk.CTkLabel(self.timeline_scroll, text="No events recorded yet.\nStart monitoring to track road users.",
                                           text_color=TEXT_MUTED, font=ctk.CTkFont(size=10))
        self.timeline_empty.grid(row=0, column=0, columnspan=2, padx=10, pady=16)

        make_card_interactive(timeline_card, PANEL_BG, PANEL_BG, PANEL_BORDER, PANEL_BORDER_HOVER)

        # ----------------------------------------------------
        # BOTTOM ROW: 3 CARDS
        # Card 1: Warning Zone Configuration
        # Card 2: Camera Controls + Night Mode
        # Card 3: Safer Roads Stronger India Banner
        # ----------------------------------------------------
        bot_row = ctk.CTkFrame(view, fg_color="transparent")
        bot_row.grid(row=1, column=0, columnspan=2, sticky="nsew")
        bot_row.grid_columnconfigure(0, weight=38)
        bot_row.grid_columnconfigure(1, weight=38)
        bot_row.grid_columnconfigure(2, weight=24)
        bot_row.grid_rowconfigure(0, weight=1)

        # Card 1: Warning Zone Configuration
        zone_card = ctk.CTkFrame(bot_row, fg_color=PANEL_BG, corner_radius=14,
                                 border_width=1, border_color=PANEL_BORDER)
        zone_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        zone_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(zone_card, text="⚙  Warning Zone Configuration", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=14, pady=(8, 4))

        # Zone mode buttons (Left Zone, Right Zone, Both Sides) with hover
        z_mode_bar = ctk.CTkFrame(zone_card, fg_color="transparent")
        z_mode_bar.pack(fill="x", padx=12, pady=(0, 6))
        z_mode_bar.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_zone_left = ctk.CTkButton(
            z_mode_bar, text="Left Zone", image=self.icon_zone_left, compound="top",
            height=42, corner_radius=8, border_width=1, border_color="#1b304c", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._set_zone_mode("left"),
        )
        self.btn_zone_left.grid(row=0, column=0, padx=3, sticky="ew")

        self.btn_zone_right = ctk.CTkButton(
            z_mode_bar, text="Right Zone", image=self.icon_zone_right, compound="top",
            height=42, corner_radius=8, border_width=1, border_color="#1b304c", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._set_zone_mode("right"),
        )
        self.btn_zone_right.grid(row=0, column=1, padx=3, sticky="ew")

        self.btn_zone_both = ctk.CTkButton(
            z_mode_bar, text="Both Sides", image=self.icon_zone_both, compound="top",
            height=42, corner_radius=8, border_width=1, border_color="#1b304c", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._set_zone_mode("both"),
        )
        self.btn_zone_both.grid(row=0, column=2, padx=3, sticky="ew")

        self._update_zone_mode_buttons()

        # Sliders: Zone Width & Zone Sensitivity
        sliders_frame = ctk.CTkFrame(zone_card, fg_color="transparent")
        sliders_frame.pack(fill="x", padx=14, pady=(0, 4))
        sliders_frame.grid_columnconfigure(1, weight=1)

        # Width
        ctk.CTkLabel(sliders_frame, text="Zone Width", text_color=TEXT_MUTED,
                     font=ctk.CTkFont(size=9)).grid(row=0, column=0, sticky="w", pady=1)
        self.slider_width = ctk.CTkSlider(
            sliders_frame, from_=0.40, to=1.00, number_of_steps=60, cursor="hand2",
            command=self._on_width_slider, progress_color=ACCENT_ORANGE, button_color=ACCENT_ORANGE,
            button_hover_color=ACCENT_ORANGE_HOVER, height=11,
        )
        self.slider_width.set(self.zone_width_factor)
        self.slider_width.grid(row=0, column=1, sticky="ew", padx=8, pady=1)
        self.lbl_width_val = ctk.CTkLabel(sliders_frame, text=f"{int(self.zone_width_factor * 100)}%",
                                          text_color=TEXT_WHITE, font=ctk.CTkFont(size=9, weight="bold"))
        self.lbl_width_val.grid(row=0, column=2, sticky="e", pady=1)

        # Sensitivity
        ctk.CTkLabel(sliders_frame, text="Sensitivity", text_color=TEXT_MUTED,
                     font=ctk.CTkFont(size=9)).grid(row=1, column=0, sticky="w", pady=1)
        self.slider_sens = ctk.CTkSlider(
            sliders_frame, from_=30, to=90, number_of_steps=60, cursor="hand2",
            command=self._on_sens_slider, progress_color=ACCENT_ORANGE, button_color=ACCENT_ORANGE,
            button_hover_color=ACCENT_ORANGE_HOVER, height=11,
        )
        self.slider_sens.set(self.zone_sensitivity_pct)
        self.slider_sens.grid(row=1, column=1, sticky="ew", padx=8, pady=1)
        self.lbl_sens_val = ctk.CTkLabel(sliders_frame, text=f"{self.zone_sensitivity_pct}%",
                                         text_color=TEXT_WHITE, font=ctk.CTkFont(size=9, weight="bold"))
        self.lbl_sens_val.grid(row=1, column=2, sticky="e", pady=1)

        # Reset button with hover
        btn_reset = ctk.CTkButton(
            zone_card, text="🔄  Reset Zones", width=90, height=22, corner_radius=6,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#1b304c",
            text_color=TEXT_LIGHT, cursor="hand2", font=ctk.CTkFont(size=9), command=self.reset_zones,
        )
        btn_reset.pack(anchor="center", pady=(0, 4))

        make_card_interactive(zone_card, PANEL_BG, PANEL_BG, PANEL_BORDER, PANEL_BORDER_HOVER)

        # Card 2: Camera Controls & ADAS Features
        ctrl_card = ctk.CTkFrame(bot_row, fg_color=PANEL_BG, corner_radius=14,
                                 border_width=1, border_color=PANEL_BORDER)
        ctrl_card.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        ctrl_card.grid_columnconfigure((0, 1), weight=1)

        c_top = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        c_top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        ctk.CTkLabel(c_top, text="📷  Camera Controls & Night Mode", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")

        # Night mode switch
        self.night_switch = ctk.CTkSwitch(
            c_top, text="🌙 Night Vision", text_color=TEXT_ACCENT,
            font=ctk.CTkFont(size=9, weight="bold"), cursor="hand2",
            command=self.toggle_night_mode,
        )
        self.night_switch.pack(side="right")

        # 2x2 Grid of buttons with high-contrast tactile hover
        self.btn_start = ctk.CTkButton(
            ctrl_card, text="▶  Start", height=34, corner_radius=8,
            fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE_HOVER, text_color="#080f1b",
            border_width=1, border_color="#fbbf24", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self.toggle_start,
        )
        self.btn_start.grid(row=1, column=0, padx=(10, 4), pady=3, sticky="ew")

        self.btn_pause = ctk.CTkButton(
            ctrl_card, text="⏸  Pause", height=34, corner_radius=8,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#15263c",
            text_color=TEXT_WHITE, cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self.toggle_pause,
        )
        self.btn_pause.grid(row=1, column=1, padx=(4, 10), pady=3, sticky="ew")

        btn_upload = ctk.CTkButton(
            ctrl_card, text="📁  Upload Video", height=32, corner_radius=8,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#1b304c",
            text_color=TEXT_LIGHT, cursor="hand2",
            font=ctk.CTkFont(size=9, weight="bold"), command=self.upload_video,
        )
        btn_upload.grid(row=2, column=0, padx=(10, 4), pady=3, sticky="ew")

        btn_demo = ctk.CTkButton(
            ctrl_card, text="✦  DRISHTI Road Demo", height=32, corner_radius=8,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#1b304c",
            text_color=TEXT_ACCENT, cursor="hand2",
            font=ctk.CTkFont(size=9, weight="bold"), command=self.run_demo,
        )
        btn_demo.grid(row=2, column=1, padx=(4, 10), pady=3, sticky="ew")

        # Snapshot folder shortcut
        btn_snap_folder = ctk.CTkButton(
            ctrl_card, text="📷  Open Incident Snapshots Folder", height=22, corner_radius=6,
            fg_color="transparent", hover_color="#15263c", text_color=TEXT_MUTED,
            cursor="hand2", font=ctk.CTkFont(size=9), command=self.open_snapshots_folder,
        )
        btn_snap_folder.grid(row=3, column=0, columnspan=2, pady=(1, 4))

        make_card_interactive(ctrl_card, PANEL_BG, PANEL_BG, PANEL_BORDER, PANEL_BORDER_HOVER)

        # Card 3: "Safer Roads, Stronger India" Banner Card
        banner_card = ctk.CTkFrame(bot_row, fg_color=PANEL_BG, corner_radius=14,
                                   border_width=1, border_color=PANEL_BORDER)
        banner_card.grid(row=0, column=2, sticky="nsew")

        if self.img_india_banner:
            banner_lbl = ctk.CTkLabel(banner_card, text="", image=self.img_india_banner)
            banner_lbl.pack(fill="both", expand=True, padx=4, pady=4)
        else:
            ctk.CTkLabel(banner_card, text="Safer Roads\nStronger India", text_color=ACCENT_ORANGE,
                         font=ctk.CTkFont(size=18, weight="bold")).pack(expand=True)

        make_card_interactive(banner_card, PANEL_BG, "#15263c", PANEL_BORDER, "#1b304c", border_width_hover=2)

    def _create_metric_card(self, parent, col, title, initial_value, initial_delta, icon, accent_col) -> dict:
        """Creates a metric card with glowing border, hover card lift, and click navigation."""
        card = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=12,
                            border_width=1, border_color=PANEL_BORDER)
        card.grid(row=0, column=col, padx=4, sticky="ew")

        top_f = ctk.CTkFrame(card, fg_color="transparent")
        top_f.pack(fill="x", padx=10, pady=(8, 2))

        if icon:
            ctk.CTkLabel(top_f, text="", image=icon).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(top_f, text=title, text_color=TEXT_MUTED,
                     font=ctk.CTkFont(size=10, weight="bold")).pack(side="left")

        val_f = ctk.CTkFrame(card, fg_color="transparent")
        val_f.pack(fill="x", padx=12, pady=(0, 8))

        lbl_val = ctk.CTkLabel(val_f, text=initial_value, text_color=TEXT_WHITE,
                               font=ctk.CTkFont(size=22, weight="bold"))
        lbl_val.pack(side="left")

        delta_pill = ctk.CTkFrame(val_f, fg_color="#0f2e24", corner_radius=8)
        delta_pill.pack(side="right", padx=(4, 0))
        lbl_delta = ctk.CTkLabel(delta_pill, text=initial_delta, text_color="#10b981",
                                 font=ctk.CTkFont(size=9, weight="bold"))
        lbl_delta.pack(padx=6, pady=2)

        make_card_interactive(
            card,
            normal_bg=PANEL_BG,
            hover_bg="#15263c",
            normal_border=PANEL_BORDER,
            hover_border=accent_col,
            border_width_hover=2,
            on_click=lambda: self._switch_tab("Analytics"),
        )

        return {"card": card, "val": lbl_val, "delta": lbl_delta}

    # ==========================================
    # ADAS COCKPIT ACTIONS (INDICATORS & SPEED)
    # ==========================================
    def toggle_left_indicator(self) -> None:
        self.indicator_left = not self.indicator_left
        if self.indicator_left:
            self.indicator_right = False
        self._recheck_alert_severity()

    def toggle_right_indicator(self) -> None:
        self.indicator_right = not self.indicator_right
        if self.indicator_right:
            self.indicator_left = False
        self._recheck_alert_severity()

    def clear_indicators(self) -> None:
        self.indicator_left = False
        self.indicator_right = False
        self._recheck_alert_severity()

    def _on_speed_change(self, val: float) -> None:
        self.vehicle_speed = int(val)
        self.lbl_speed_val.configure(text=f"{self.vehicle_speed} km/h")
        self._recheck_alert_severity()

    def _set_speed_preset(self, speed: int) -> None:
        self.vehicle_speed = speed
        self.speed_slider.set(speed)
        self.lbl_speed_val.configure(text=f"{self.vehicle_speed} km/h")
        self._recheck_alert_severity()

    def toggle_night_mode(self) -> None:
        self.night_mode = bool(self.night_switch.get())
        if self.worker:
            self.worker.night_mode = self.night_mode
        if self.night_mode:
            self.night_badge.pack(side="left", padx=(8, 0))
        else:
            self.night_badge.pack_forget()

    def _recheck_alert_severity(self) -> None:
        """Immediately re-evaluates Stage 1 (Amber) vs Stage 2 (Red Critical) alert."""
        if not self.alert_active:
            return
        self._evaluate_two_stage_alarm()

    def toggle_kiosk_mode(self) -> None:
        """Toggles distraction-free full-screen driver kiosk mode."""
        self.kiosk_active = not self.kiosk_active
        if self.kiosk_active:
            self.header_frame.grid_remove()
            self.nav_bar.grid_remove()
            self.attributes("-fullscreen", True)
            self._switch_tab("Live Monitor")
        else:
            self.attributes("-fullscreen", False)
            self.header_frame.grid()
            self.nav_bar.grid()

    def _exit_kiosk_if_active(self) -> None:
        if self.kiosk_active:
            self.toggle_kiosk_mode()

    def open_snapshots_folder(self) -> None:
        try:
            os.startfile(str(SNAPSHOTS_DIR))
        except Exception:
            messagebox.showinfo("Snapshots Vault", f"Snapshots are stored at:\n{SNAPSHOTS_DIR}")

    def generate_pdf_report_action(self) -> None:
        try:
            pdf_path = generate_pdf_report.build_pdf_report()
            ans = messagebox.askyesno(
                "PDF Generated Successfully",
                f"DRISHTI Safety & Architecture Report generated at:\n{pdf_path}\n\nDo you want to open the PDF now?",
            )
            if ans:
                os.startfile(pdf_path)
        except Exception as exc:
            messagebox.showerror("PDF Generation Failed", f"Could not create report:\n{exc}")

    def _set_zone_mode(self, mode: str) -> None:
        self.zone_mode = mode
        self._update_zone_mode_buttons()
        if self.worker:
            self.worker.zone_mode = mode

    def _update_zone_mode_buttons(self) -> None:
        modes = [
            ("left", self.btn_zone_left),
            ("right", self.btn_zone_right),
            ("both", self.btn_zone_both),
        ]
        for m, btn in modes:
            if self.zone_mode == m:
                btn.configure(
                    fg_color=ACCENT_ORANGE,
                    text_color="#080f1b",
                    border_color="#fbbf24",
                    hover_color=ACCENT_ORANGE_HOVER,
                )
            else:
                btn.configure(
                    fg_color="#15263c",
                    text_color=TEXT_MUTED,
                    border_color="#1b304c",
                    hover_color="#15263c",
                )

    def _on_width_slider(self, val: float) -> None:
        self.zone_width_factor = float(val)
        self.lbl_width_val.configure(text=f"{int(self.zone_width_factor * 100)}%")
        self._rescale_zones()

    def _on_sens_slider(self, val: float) -> None:
        self.zone_sensitivity_pct = int(val)
        self.lbl_sens_val.configure(text=f"{self.zone_sensitivity_pct}%")
        self.confidence_val = max(0.15, min(0.65, 0.70 - (self.zone_sensitivity_pct / 100.0) * 0.55))
        if self.worker:
            self.worker.confidence = self.confidence_val

    def _rescale_zones(self) -> None:
        w_factor = self.zone_width_factor
        # Scale right zone width
        rx_inner = 0.65 - (w_factor - 0.50) * 0.30
        self.zones["right"] = [
            (rx_inner, 0.34),
            (0.98, 0.36),
            (0.98, 0.95),
            (rx_inner - 0.05, 0.95),
        ]
        # Scale left zone width
        lx_inner = 0.20 + (w_factor - 0.50) * 0.35
        self.zones["left"] = [
            (0.01, 0.32),
            (lx_inner, 0.30),
            (lx_inner + 0.10, 0.95),
            (0.01, 0.95),
        ]
        if self.worker:
            self.worker.zones = {k: list(v) for k, v in self.zones.items()}

    def reset_zones(self) -> None:
        self.zone_mode = "both"
        self.zone_width_factor = 0.70
        self.zone_sensitivity_pct = 65
        self.slider_width.set(0.70)
        self.slider_sens.set(65)
        self.lbl_width_val.configure(text="70%")
        self.lbl_sens_val.configure(text="65%")
        self.confidence_val = 0.35
        self.zones = {
            "left": list(DEFAULT_ZONES["left"]),
            "right": list(DEFAULT_ZONES["right"]),
        }
        self._update_zone_mode_buttons()
        if self.worker:
            self.worker.zone_mode = "both"
            self.worker.confidence = self.confidence_val
            self.worker.zones = {k: list(v) for k, v in self.zones.items()}

    # ==========================================
    # VIEW 2: ANALYTICS TAB
    # ==========================================
    def _build_analytics_view(self) -> None:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.tabs["Analytics"] = view
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(1, weight=1)

        # Header bar with hover
        header = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=5, height=52)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(header, text="📊  DRISHTI Detection & Safety Analytics",
                     text_color=TEXT_WHITE, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=16, pady=10)

        # Action Buttons
        btn_box = ctk.CTkFrame(header, fg_color="transparent")
        btn_box.pack(side="right", padx=14)

        ctk.CTkButton(
            btn_box, text="📄  Generate Safety Report PDF", width=190, height=30,
            fg_color="#38bdf8", hover_color="#fbbf24", text_color=TEXT_WHITE,
            border_width=1, border_color="#38bdf8", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self.generate_pdf_report_action,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_box, text="🔄  Refresh Charts", width=120, height=30,
            fg_color="#15263c", hover_color="#1b304c", cursor="hand2",
            border_width=1, border_color="#15263c",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._update_analytics_charts,
        ).pack(side="left", padx=5)

        # Chart container
        self.analytics_container = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=14,
                                                border_width=1, border_color=PANEL_BORDER)
        self.analytics_container.grid(row=1, column=0, sticky="nsew")

    def _update_analytics_charts(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        for child in self.analytics_container.winfo_children():
            child.destroy()

        classes = self.store.class_totals(30)
        daily = self.store.daily_event_totals(30)

        fig = Figure(figsize=(10, 5.8), dpi=100, facecolor=PANEL_BG)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_facecolor("#15263c")
        ax2.set_facecolor("#15263c")

        # Chart 1: Road Users by Class
        if classes:
            names, counts = zip(*classes)
            bars = ax1.barh(names[::-1], counts[::-1], color=ACCENT_ORANGE, edgecolor="#fbbf24", height=0.55)
            ax1.bar_label(bars, padding=4, color=TEXT_WHITE, fontsize=9)
            ax1.set_title("Detected Road Users by Class (Last 30 Days)", color=TEXT_WHITE, pad=12, fontsize=11, fontweight="bold")
        else:
            ax1.text(0.5, 0.5, "No tracked detections yet", ha="center", color=TEXT_MUTED, transform=ax1.transAxes)
            ax1.set_title("Detected Road Users by Class", color=TEXT_WHITE, pad=12, fontsize=11, fontweight="bold")

        # Chart 2: Daily Activity Trends
        series = {"BLIND-SPOT ENTRY": ([], []), "ZONE CLEARED": ([], []), "OBJECT DETECTED": ([], [])}
        for date, event_type, count in daily:
            if event_type in series:
                series[event_type][0].append(date[-5:])  # MM-DD
                series[event_type][1].append(count)

        styles = {
            "BLIND-SPOT ENTRY": (ALERT_RED, "o", "Blind-Spot Alerts"),
            "ZONE CLEARED": (SAFE_GREEN, "s", "Zone Clears"),
            "OBJECT DETECTED": (TEXT_ACCENT, "^", "Objects Tracked"),
        }

        has_data = False
        for event_type, (dates, counts) in series.items():
            if dates:
                has_data = True
                color, marker, lbl = styles[event_type]
                ax2.plot(dates, counts, color=color, marker=marker, linewidth=2, label=lbl)

        if not has_data:
            ax2.text(0.5, 0.5, "No historical trends available yet", ha="center", color=TEXT_MUTED, transform=ax2.transAxes)

        ax2.set_title("Daily Safety Events Activity", color=TEXT_WHITE, pad=12, fontsize=11, fontweight="bold")
        ax2.legend(facecolor=PANEL_ALT, labelcolor=TEXT_WHITE, fontsize=9, loc="upper left")

        for ax in (ax1, ax2):
            ax.tick_params(colors=TEXT_MUTED, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(PANEL_BORDER)
            ax.grid(axis="x", color="#15263c", alpha=0.6, linestyle="--")

        ax2.tick_params(axis="x", rotation=35)
        fig.tight_layout(pad=2.5)

        canvas = FigureCanvasTkAgg(fig, master=self.analytics_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

    # ==========================================
    # VIEW 3: HISTORY TAB
    # ==========================================
    def _build_history_view(self) -> None:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.tabs["History"] = view
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(1, weight=1)

        # Controls & Filter Bar
        bar = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=5, height=50)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(bar, text="🕒  Detection Event History & Incident Snapshots", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=16, pady=10)

        # Filter dropdown
        self.history_filter_var = tk.StringVar(value="All Events")
        ctk.CTkOptionMenu(
            bar, variable=self.history_filter_var,
            values=["All Events", "BLIND-SPOT ENTRY", "OBJECT DETECTED", "ZONE CLEARED"],
            command=lambda _: self._refresh_history_tab(),
            width=150, height=30, fg_color=PANEL_ALT, button_color="#15263c", cursor="hand2",
            font=ctk.CTkFont(size=10),
        ).pack(side="left", padx=10)

        # Export & PDF Buttons
        ctk.CTkButton(
            bar, text="📄  Generate PDF Report", width=155, height=30, corner_radius=8,
            fg_color="#38bdf8", hover_color="#fbbf24", text_color=TEXT_WHITE,
            border_width=1, border_color="#38bdf8", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self.generate_pdf_report_action,
        ).pack(side="right", padx=12)

        ctk.CTkButton(
            bar, text="↓  Export CSV", width=100, height=30, corner_radius=8,
            fg_color="#15263c", hover_color="#0f2e24", text_color="#10b981",
            border_width=1, border_color="#10b981", cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self.export_history_csv,
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            bar, text="🔄  Refresh", width=80, height=30, corner_radius=8,
            fg_color="#15263c", hover_color="#15263c", border_width=1, border_color="#15263c",
            text_color=TEXT_LIGHT, cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self._refresh_history_tab,
        ).pack(side="right", padx=4)

        # History table container
        table_card = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=14,
                                  border_width=1, border_color=PANEL_BORDER)
        table_card.grid(row=1, column=0, sticky="nsew")
        table_card.grid_columnconfigure(0, weight=1)
        table_card.grid_rowconfigure(1, weight=1)

        # Table Header
        th = ctk.CTkFrame(table_card, fg_color="#15263c", corner_radius=8, height=34)
        th.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        th.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        headers = ["Timestamp", "Object Class", "Confidence", "Event Type", "Severity Status", "Source", "Photo Evidence"]
        for idx, h in enumerate(headers):
            ctk.CTkLabel(th, text=h, text_color=TEXT_MUTED, font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=idx, padx=4, pady=6)

        # Scrollable table body
        self.history_table_scroll = ctk.CTkScrollableFrame(table_card, fg_color="transparent")
        self.history_table_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.history_table_scroll.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

    def _refresh_history_tab(self) -> None:
        if not hasattr(self, "history_table_scroll"):
            return

        for child in self.history_table_scroll.winfo_children():
            child.destroy()

        rows = self.store.recent_events(120)
        filt = self.history_filter_var.get()

        displayed = 0
        for row in rows:
            stamp, name, confidence, event_type, warning_status, source = row[:6]
            snapshot_file = row[6] if len(row) > 6 else ""

            if filt != "All Events" and event_type != filt:
                continue

            try:
                t_str = datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                t_str = stamp

            normal_bg = PANEL_ALT if displayed % 2 == 0 else "#15263c"
            r_frame = ctk.CTkFrame(self.history_table_scroll, fg_color=normal_bg, corner_radius=6,
                                   border_width=1, border_color=normal_bg)
            r_frame.pack(fill="x", pady=2)
            r_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

            color = ALERT_RED if "CRITICAL" in warning_status or event_type == "BLIND-SPOT ENTRY" else (
                ADVISORY_AMBER if "ADVISORY" in warning_status else SAFE_GREEN
            )

            ctk.CTkLabel(r_frame, text=t_str, text_color=TEXT_MUTED, font=ctk.CTkFont(size=9)).grid(row=0, column=0, padx=4, pady=5)
            ctk.CTkLabel(r_frame, text=name, text_color=TEXT_WHITE, font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=1, padx=4, pady=5)
            ctk.CTkLabel(r_frame, text=f"{confidence:.0%}", text_color=TEXT_MUTED, font=ctk.CTkFont(size=9)).grid(row=0, column=2, padx=4, pady=5)
            ctk.CTkLabel(r_frame, text=event_type, text_color=color, font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=3, padx=4, pady=5)
            ctk.CTkLabel(r_frame, text=warning_status, text_color=color, font=ctk.CTkFont(size=9)).grid(row=0, column=4, padx=4, pady=5)
            ctk.CTkLabel(r_frame, text=source[:18], text_color=TEXT_MUTED, font=ctk.CTkFont(size=9)).grid(row=0, column=5, padx=4, pady=5)

            # Snapshot button
            if snapshot_file and (SNAPSHOTS_DIR / snapshot_file).exists():
                snap_btn = ctk.CTkButton(
                    r_frame, text="📷 View", width=62, height=20, corner_radius=4,
                    fg_color="#38bdf8", hover_color="#fbbf24", text_color=TEXT_WHITE,
                    cursor="hand2", font=ctk.CTkFont(size=8, weight="bold"),
                    command=lambda s=snapshot_file, t=t_str, n=name: self._show_snapshot_viewer(s, t, n),
                )
                snap_btn.grid(row=0, column=6, padx=4, pady=5)
            else:
                ctk.CTkLabel(r_frame, text="—", text_color=TEXT_MUTED, font=ctk.CTkFont(size=9)).grid(row=0, column=6, padx=4, pady=5)

            # Row hover transition
            make_card_interactive(r_frame, normal_bg, "#15263c", normal_bg, "#1b304c")
            displayed += 1

        if displayed == 0:
            ctk.CTkLabel(self.history_table_scroll, text="No events matching the selected filter",
                         text_color=TEXT_MUTED, font=ctk.CTkFont(size=11)).pack(pady=20)

    def _show_snapshot_viewer(self, snapshot_filename: str, timestamp_str: str, object_name: str) -> None:
        """Modal viewer popup to inspect black-box incident snapshot photo evidence."""
        snap_path = SNAPSHOTS_DIR / snapshot_filename
        if not snap_path.exists():
            messagebox.showwarning("File Missing", f"Snapshot {snapshot_filename} was not found on disk.")
            return

        win = ctk.CTkToplevel(self)
        win.title(f"DRISHTI Incident Evidence · {snapshot_filename}")
        win.geometry("860x640")
        win.configure(fg_color=BG_MAIN)
        win.attributes("-topmost", True)

        header = ctk.CTkFrame(win, fg_color=PANEL_BG, corner_radius=0, height=48)
        header.pack(fill="x", padx=12, pady=10)
        ctk.CTkLabel(header, text=f"📷  Incident Photo Vault: {object_name} at {timestamp_str}",
                     text_color=TEXT_WHITE, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=14)

        try:
            pil_img = Image.open(snap_path)
            pil_img.thumbnail((820, 500), Image.Resampling.LANCZOS)
            ctk_snap = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            lbl = ctk.CTkLabel(win, text="", image=ctk_snap, fg_color="#060c14", corner_radius=10)
            lbl.image = ctk_snap  # Keep reference
            lbl.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        except Exception as exc:
            ctk.CTkLabel(win, text=f"Could not render image: {exc}", text_color=ALERT_RED).pack(pady=30)

        ctk.CTkButton(win, text="Close Viewer", width=100, height=28, fg_color="#15263c",
                      command=win.destroy, cursor="hand2").pack(pady=(0, 10))

    def export_history_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export DRISHTI Detection History",
            defaultextension=".csv",
            initialfile=f"drishti_blind_spot_events_{datetime.now():%Y%m%d_%H%M%S}.csv",
            filetypes=[("CSV spreadsheet", "*.csv")],
        )
        if path:
            try:
                count = self.store.export_csv(path)
                messagebox.showinfo("Export Complete", f"Successfully exported {count} detection events to:\n{Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Export Failed", f"Could not write CSV:\n{exc}")

    # ==========================================
    # VIEW 4: SETTINGS TAB
    # ==========================================
    def _build_settings_view(self) -> None:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.tabs["Settings"] = view
        view.grid_columnconfigure((0, 1), weight=1)

        # Section 1: Video & Camera Source + Auto-Scanner
        s1 = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=PANEL_BORDER)
        s1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=8)

        ctk.CTkLabel(s1, text="📷  Camera Hardware & Video Source", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(14, 10))

        # Camera dropdown + Auto-Scan Button
        f_cam = ctk.CTkFrame(s1, fg_color="transparent")
        f_cam.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(f_cam, text="Webcam Device Index:", text_color=TEXT_LIGHT).pack(side="left")

        self.camera_var = tk.StringVar(value="Camera 0")
        self.camera_menu = ctk.CTkOptionMenu(
            f_cam, variable=self.camera_var, values=["Camera 0", "Camera 1", "Camera 2"],
            command=self._on_camera_select, width=120, height=30, fg_color=PANEL_ALT, cursor="hand2",
        )
        self.camera_menu.pack(side="right")

        f_scan = ctk.CTkFrame(s1, fg_color="transparent")
        f_scan.pack(fill="x", padx=16, pady=4)
        btn_scan = ctk.CTkButton(
            f_scan, text="🔍  Auto-Scan Connected Cameras", width=220, height=28, corner_radius=6,
            fg_color="#15263c", hover_color="#15263c", text_color=TEXT_ACCENT, cursor="hand2",
            font=ctk.CTkFont(size=10, weight="bold"), command=self._scan_cameras_action,
        )
        btn_scan.pack(side="left")

        # Video file upload
        f_vid = ctk.CTkFrame(s1, fg_color="transparent")
        f_vid.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(f_vid, text="Custom Video File:", text_color=TEXT_LIGHT).pack(side="left")
        ctk.CTkButton(f_vid, text="Browse Video...", width=120, height=30, cursor="hand2",
                      fg_color="#15263c", hover_color="#15263c", command=self.upload_video).pack(side="right")

        # Looping toggle
        f_loop = ctk.CTkFrame(s1, fg_color="transparent")
        f_loop.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(f_loop, text="Loop Video Playback:", text_color=TEXT_LIGHT).pack(side="left")
        self.loop_switch = ctk.CTkSwitch(f_loop, text="", command=self._toggle_loop, cursor="hand2")
        if self.loop_video:
            self.loop_switch.select()
        self.loop_switch.pack(side="right")

        # Section 2: Audio Alert Settings
        s2 = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=PANEL_BORDER)
        s2.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=8)

        ctk.CTkLabel(s2, text="🔊  ADAS Audio Alert Settings", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(14, 10))

        # Mute toggle
        f_mute = ctk.CTkFrame(s2, fg_color="transparent")
        f_mute.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(f_mute, text="Speaker Alarm Active:", text_color=TEXT_LIGHT).pack(side="left")
        self.sound_switch = ctk.CTkSwitch(f_mute, text="", command=self.toggle_mute, cursor="hand2")
        self.sound_switch.select()
        self.sound_switch.pack(side="right")

        # Volume slider
        f_vol = ctk.CTkFrame(s2, fg_color="transparent")
        f_vol.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(f_vol, text="Alarm Volume:", text_color=TEXT_LIGHT).pack(side="left")
        self.vol_slider = ctk.CTkSlider(f_vol, from_=0, to=1, command=self.audio.set_volume, cursor="hand2",
                                        progress_color=ACCENT_ORANGE, button_color=ACCENT_ORANGE, width=140)
        self.vol_slider.set(0.60)
        self.vol_slider.pack(side="right")

        # Test Alarm Sound
        f_test = ctk.CTkFrame(s2, fg_color="transparent")
        f_test.pack(fill="x", padx=16, pady=8)
        ctk.CTkButton(f_test, text="🔔 Test Stage 1 (Advisory)", width=150, height=28, cursor="hand2",
                      fg_color="#15263c", hover_color="#15263c", command=self.audio.play_advisory).pack(side="left", padx=(0, 6))
        ctk.CTkButton(f_test, text="🚨 Test Stage 2 (Critical)", width=150, height=28, cursor="hand2",
                      fg_color="#15263c", hover_color="#631a28", text_color="#fda4af", command=self.audio.play_critical).pack(side="left")

        # Section 3: Interactive Polygon Zone Editor
        s3 = ctk.CTkFrame(view, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=PANEL_BORDER)
        s3.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        ctk.CTkLabel(s3, text="📐  Interactive Zone Vertex Editor", text_color=TEXT_WHITE,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            s3,
            text="Toggle on polygon editing to visually drag vertices directly on the live camera canvas.\nClick on the video to add a vertex, or right-click near a vertex to delete it.",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11), justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self.btn_toggle_edit = ctk.CTkButton(
            s3, text="✎  Enable Interactive Polygon Dragging", width=250, height=34, corner_radius=8,
            fg_color="#15263c", hover_color="#1b304c", text_color=TEXT_WHITE, cursor="hand2",
            border_width=1, border_color="#1b304c",
            font=ctk.CTkFont(size=11, weight="bold"), command=self.toggle_zone_edit,
        )
        self.btn_toggle_edit.pack(anchor="w", padx=16, pady=(0, 12))

    def _scan_cameras_action(self) -> None:
        """Scans hardware indices and populates the camera selector with live devices."""
        cams = scan_available_cameras(6)
        cam_strings = [f"Camera {idx}" for idx in cams]
        self.camera_menu.configure(values=cam_strings)
        if cam_strings:
            self.camera_var.set(cam_strings[0])
        messagebox.showinfo("Hardware Scan Complete", f"Found {len(cams)} active camera(s):\n{', '.join(cam_strings)}")

    def _toggle_loop(self) -> None:
        self.loop_video = bool(self.loop_switch.get())
        if self.worker:
            self.worker.loop_video = self.loop_video

    def _on_camera_select(self, val: str) -> None:
        if self.running:
            self.stop()
        self.video_path = None
        self.loop_video = False
        self.source_name = val
        self.feed_source_badge.configure(text=val)

    def toggle_zone_edit(self) -> None:
        self.zone_editing = not self.zone_editing
        self.drag_vertex = None
        if self.zone_editing:
            self.btn_toggle_edit.configure(text="✓  Finish Editing Polygon", fg_color="#6a2536", hover_color="#883045")
            self._switch_tab("Live Monitor")
            messagebox.showinfo("Zone Editor Active", "You can now drag polygon points on the camera view.\nClick to add points, right-click a point to delete.")
        else:
            self.btn_toggle_edit.configure(text="✎  Enable Interactive Polygon Dragging", fg_color="#15263c", hover_color="#1b304c")

    # ==========================================
    # VIEW 5: ABOUT TAB
    # ==========================================
    def _build_about_view(self) -> None:
        view = ctk.CTkFrame(self.content_container, fg_color=PANEL_BG, corner_radius=14,
                            border_width=1, border_color=PANEL_BORDER)
        self.tabs["About"] = view
        view.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(view, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(scroll, text="DRISHTI · Intelligent Blind-Spot Warning System",
                     text_color=ACCENT_ORANGE, font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(scroll, text="Dedicated to Safer Roads & Stronger India · Heavy Vehicle Road Safety AI",
                     text_color=TEXT_LIGHT, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(0, 16))

        desc = (
            "DRISHTI is an educational computer-vision demonstration that detects road users inside configurable "
            "image-space warning zones. Its simulated left/right zones do not measure or eliminate real truck blind spots.\n\n"
            "Demonstration Capabilities:\n"
            "• Dual-Zone Image Analysis: Simultaneous left and right example regions over the video feed.\n"
            "• Simulated ADAS Controls: Demonstrates how turn-signal and speed inputs could affect warning priority; "
            "these controls are not connected to a real vehicle.\n"
            "• Simulated Speed Filter: Demonstrates a stationary mute rule and does not measure vehicle speed.\n"
            "• Image-Space Motion Estimates: Shows pixel movement trends; it does not estimate real-world speed or distance.\n"
            "• Low-Light Contrast Mode: Applies image enhancement, but does not guarantee night or adverse-weather detection.\n"
            "• Zone-Entry Snapshots: Saves annotated images when a tracked object enters an example warning region.\n"
            "• Optional Voice Alerts: Speaks the detected object class and image-side zone on a new entry.\n"
            "• Fully Local Execution: 100% on-device processing without cloud latency or external telemetry leaks."
        )
        ctk.CTkLabel(scroll, text=desc, text_color=TEXT_LIGHT, font=ctk.CTkFont(size=11), justify="left").pack(anchor="w", pady=(0, 20))

        disclaimer = (
            "Educational & Research Notice:\n"
            "This software is a demonstration tool and algorithm prototype. The warning zones in this software are "
            "image-space screen coordinate overlays. They are not calibrated to specific physical truck dimensions, "
            "chassis geometry, or optical lens distortion. Do not use this demo in a live operating vehicle for safety decisions."
        )
        d_card = ctk.CTkFrame(scroll, fg_color="#15263c", corner_radius=10, border_width=1, border_color="#1b304c")
        d_card.pack(fill="x", pady=10)
        ctk.CTkLabel(d_card, text=disclaimer, text_color="#fde68a", font=ctk.CTkFont(size=10), justify="left").pack(padx=14, pady=10)

    # ==========================================
    # RUNTIME DETECTION & POLLING
    # ==========================================
    def toggle_start(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self, source_override=None, performance_mode_override: str | None = None) -> None:
        if self.running:
            return
        source = source_override if source_override is not None else (
            self.video_path if self.video_path else int(self.camera_var.get().split()[-1])
        )
        self.messages = queue.Queue(maxsize=4)
        self.session_id = uuid.uuid4().hex
        self.known_tracks.clear()
        self.zone_tracks.clear()
        self.alert_active = False
        self.alert_is_critical = False

        self.worker = DetectionWorker(
            source,
            self.confidence_val,
            self.zones,
            self.messages,
            loop_video=self.loop_video,
            zone_mode=self.zone_mode,
            performance_mode=performance_mode_override or ("Detail" if source_override == DEMO_VIDEO else "Fast"),
        )
        self.worker.night_mode = self.night_mode
        self.worker.start()
        self.running = True
        self.paused = False
        self.btn_start.configure(text="⏹  Stop", fg_color="#ef4444", hover_color="#ef4444", border_color="#ef4444", text_color=TEXT_WHITE)
        self.btn_pause.configure(text="⏸  Pause")
        self.feed_source_badge.configure(text=f"CONNECTING · {self.source_name[:24]}")

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
        self.worker = None
        self.running = False
        self.paused = False
        self.alert_active = False
        self.alert_is_critical = False
        self._clear_zone_tracks()
        self.audio.stop()
        self.btn_start.configure(text="▶  Start", fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE_HOVER, border_color="#fbbf24", text_color="#080f1b")
        self.btn_pause.configure(text="⏸  Pause")
        self.fps_label.configure(text="AI 0.0 · UI 0.0 FPS")
        self._last_ui_frame_time = 0.0
        self._ui_fps = 0.0
        self._last_audio_signature = None
        self._set_alert("NORMAL", "No road users in warning zone", is_critical=False)

    def toggle_pause(self) -> None:
        if not self.worker:
            return
        self.paused = not self.paused
        (self.worker.pause if self.paused else self.worker.resume)()
        self.btn_pause.configure(text="▶  Resume" if self.paused else "⏸  Pause")

    def run_demo(self) -> None:
        if not DEMO_VIDEO.exists():
            messagebox.showerror("Demo Video Missing", "The bundled video assets/demo_road.webm was not found.")
            return
        if self.running:
            self.stop()
        self.video_path = DEMO_VIDEO
        self.source_name = "DRISHTI Traffic Demo"
        self.loop_video = True
        self.feed_source_badge.configure(text="DRISHTI Road Demo")
        self.start(source_override=DEMO_VIDEO, performance_mode_override="Detail")

    def upload_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a Road Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.webm"), ("All files", "*.*")],
        )
        if path:
            self.stop()
            self.video_path = Path(path)
            self.loop_video = False
            self.source_name = self.video_path.name
            self.feed_source_badge.configure(text=self.source_name[:24])
            self.start()

    def toggle_mute(self) -> None:
        muted = not self.audio.muted
        self.audio.set_muted(muted)
        self.alert_speaker_btn.configure(text="🔇" if muted else "🔊")

    def _poll(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message.get("kind")
                if not self.running and kind in {"status", "preview", "frame", "error", "ended", "looped"}:
                    continue
                if kind == "status":
                    text = message.get("text", "")
                    if "ready" in text.lower() or "connected" in text.lower():
                        self.feed_source_badge.configure(text=f"{self.source_name[:24]}")
                elif kind == "frame":
                    self._render(message)
                elif kind == "preview":
                    # Show the selected source immediately while YOLO initializes in the worker.
                    self._render({"frame": message["frame"], "detections": []})
                elif kind == "error":
                    self.stop()
                    messagebox.showerror("Monitoring Error", message.get("text", "Unknown capture error"))
                elif kind == "ended":
                    self.stop()
        except queue.Empty:
            pass
        self.after(30, self._poll)

    # ==========================================
    # FRAME RENDERING & ADAS OVERLAYS
    # ==========================================
    def _render(self, message: dict) -> None:
        frame = message["frame"]
        detections: list[Detection] = message["detections"]
        self.frame_size = (frame.shape[1], frame.shape[0])
        # Draw Left Zone, Right Zone, Mirror Box, Motion Vectors, and Detections
        self._draw_zones_and_detections(frame, detections)

        # Resize in OpenCV first so the UI only converts the smaller display image.
        widget_w = max(10, self.video_view.winfo_width() - 8)
        widget_h = max(10, self.video_view.winfo_height() - 8)
        frame_h, frame_w = frame.shape[:2]
        scale = min(widget_w / max(frame_w, 1), widget_h / max(frame_h, 1))
        display_size = (max(1, int(frame_w * scale)), max(1, int(frame_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        display_bgr = cv2.resize(frame, display_size, interpolation=interpolation)
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self.display_geometry = (
            (self.video_view.winfo_width() - pil.width) / 2,
            (self.video_view.winfo_height() - pil.height) / 2,
            pil.width / max(1, self.frame_size[0]),
        )
        self._video_image = ImageTk.PhotoImage(pil)
        self.video_view.configure(image=self._video_image, text="")

        # Track management and zone entry evaluation
        key_prefix = int(message.get("loop_index", 0))
        track_keys = {(key_prefix, d.track_id) for d in detections}
        detections_by_key = {(key_prefix, d.track_id): d for d in detections}

        # New object detected
        for key, detection in detections_by_key.items():
            if key not in self.known_tracks:
                self.known_tracks.add(key)
                self._record_event(detection, "OBJECT DETECTED", "NORMAL")

        # Zone occupancy
        current_zone = {key: detections_by_key[key] for key in track_keys if detections_by_key[key].in_zone}
        entering = current_zone.keys() - self.zone_tracks.keys()
        leaving = self.zone_tracks.keys() - current_zone.keys()

        for key in entering:
            detection = current_zone[key]
            side = "right" if "right" in detection.zone_name.lower() else "left"
            z_label = f"{side.title()} Zone"
            snap_path = self._capture_incident_snapshot(frame, detection)
            self._record_event(
                detection, f"Blind spot detected – {z_label}", "BLIND-SPOT ALERT",
                is_alert=True, snapshot_path=snap_path,
            )
            self.warning_events += 1
            self.audio.speak_warning(detection.name, side)

        for key in leaving:
            previous = self.zone_tracks[key]
            self._record_event(previous, "Zone cleared", "NORMAL")

        self.zone_tracks = current_zone
        self.alert_active = bool(current_zone)

        # Evaluate Two-Stage ADAS Alarm (Turn signal correlation + Speed filter)
        self._evaluate_two_stage_alarm(new_entry=bool(entering))

        # Separate model throughput from the rate the desktop actually renders frames.
        now = time.perf_counter()
        if self._last_ui_frame_time:
            ui_instant = 1.0 / max(now - self._last_ui_frame_time, 1e-6)
            self._ui_fps = ui_instant if not self._ui_fps else self._ui_fps * 0.75 + ui_instant * 0.25
        self._last_ui_frame_time = now
        fps = message.get("inference_fps", 0)
        skipped = message.get("frames_skipped", 0)
        self.fps_label.configure(text=f"AI {fps:.1f} · UI {self._ui_fps:.1f} FPS · skip {skipped}")

        # Update 3 Metric Cards with Deltas
        self._update_metrics(detections)

    def _evaluate_two_stage_alarm(self, new_entry: bool = False) -> None:
        """Core ADAS Decision Engine: Correlates Turn Signals + Blind Zones + Speed."""
        if not self.alert_active:
            self.alert_is_critical = False
            self.audio.stop()
            self._last_audio_signature = None
            self._set_alert("NORMAL", "No road users in warning zone", is_critical=False)
            return

        zones_with_objs = {d.zone_name.lower() for d in self.zone_tracks.values()}
        has_left = any("left" in z for z in zones_with_objs)
        has_right = any("right" in z for z in zones_with_objs)

        # Check if Turn Signal indicates intention to turn toward an occupied zone
        critical_turn = (has_right and self.indicator_right) or (has_left and self.indicator_left)

        # Check Stationary Noise Filter (< 5 km/h)
        is_stationary = self.vehicle_speed < 5
        road_users = sorted({
            f"{d.name} on {'right' if 'right' in d.zone_name.lower() else 'left'} side"
            for d in self.zone_tracks.values()
        })
        occupants = ", ".join(road_users)

        if critical_turn:
            self.alert_is_critical = True
            which_side = "RIGHT" if (has_right and self.indicator_right) else "LEFT"
            title = f"CRITICAL: {which_side} TURN SIGNAL ACTIVE!"
            detail = f"COLLISION THREAT: {occupants} inside the warning zone."
            signature = ("critical", has_left, has_right, is_stationary)
            if is_stationary:
                detail += " [AUDIO MUTED AT 0 KM/H]"
                self.audio.stop()
            elif new_entry or signature != self._last_audio_signature:
                self.audio.play_critical()
            self._last_audio_signature = signature
            self._set_alert(title, detail, is_critical=True)
        else:
            self.alert_is_critical = False
            which_side = "Both Sides" if (has_left and has_right) else ("Right Zone" if has_right else "Left Zone")
            title = f"ADVISORY: ROAD USER IN {which_side.upper()}"
            detail = f"{occupants}. Turn signal is OFF."
            signature = ("advisory", has_left, has_right, is_stationary)
            if is_stationary:
                detail += " [AUDIO MUTED AT 0 KM/H]"
                self.audio.stop()
            elif new_entry or signature != self._last_audio_signature:
                self.audio.play_advisory()
            self._last_audio_signature = signature
            self._set_alert(title, detail, is_critical=False)

    def _capture_incident_snapshot(self, frame: np.ndarray, detection: Detection) -> str:
        """Captures a high-resolution black-box photo evidence frame."""
        now_ts = time.time()
        last_t = self._last_snapshot_time.get(detection.track_id, 0.0)
        if now_ts - last_t < 3.0:
            return ""

        self._last_snapshot_time[detection.track_id] = now_ts
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incident_{stamp}_ID{detection.track_id}_{detection.name.replace(' ', '_')}.jpg"
        save_path = SNAPSHOTS_DIR / filename

        try:
            # Watermark the incident snapshot with speed and timestamp
            annotated = frame.copy()
            h, w = annotated.shape[:2]
            watermark = f"DRISHTI ADAS INCIDENT · {datetime.now():%Y-%m-%d %H:%M:%S} · SPEED {self.vehicle_speed} km/h · ID {detection.track_id} {detection.name}"
            cv2.rectangle(annotated, (0, h - 30), (w, h), (10, 15, 25), -1)
            cv2.putText(annotated, watermark, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(save_path), annotated)
            return filename
        except Exception:
            return ""

    def _draw_zones_and_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Check occupancy
        left_occupied = any(d.in_zone and "left" in d.zone_name.lower() for d in detections)
        right_occupied = any(d.in_zone and "right" in d.zone_name.lower() for d in detections)

        # 1. Left Blind Zone (Red Overlay if alert, Cyan/Blue if safe)
        if self.zone_mode in ("left", "both") and len(self.zones.get("left", [])) >= 3:
            pts_l = np.array([[int(x * w), int(y * h)] for x, y in self.zones["left"]], dtype=np.int32)
            color_l = (40, 40, 235) if (left_occupied and self.indicator_left) else (
                (20, 140, 235) if left_occupied else (50, 50, 200)
            )
            cv2.fillPoly(overlay, [pts_l], color_l)
            cv2.polylines(frame, [pts_l], True, (60, 60, 245) if left_occupied else (40, 160, 240), 2, cv2.LINE_AA)

            # Left Zone Label Badge
            lx, ly = pts_l[0]
            badge_text = "Left Blind Zone"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (lx + 5, ly + 5), (lx + tw + 18, ly + th + 16), (30, 30, 180), -1)
            cv2.putText(frame, badge_text, (lx + 11, ly + th + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            if left_occupied:
                ax, ay = int(0.12 * w), int(0.48 * h)
                alert_badge = "! CRITICAL BLIND SPOT ALERT!" if self.indicator_left else "! ADVISORY: ROAD USER IN ZONE"
                (atw, ath), _ = cv2.getTextSize(alert_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                badge_bg = (20, 20, 220) if self.indicator_left else (15, 120, 220)
                cv2.rectangle(frame, (ax - 4, ay - ath - 8), (ax + atw + 14, ay + 8), badge_bg, -1)
                cv2.rectangle(frame, (ax - 4, ay - ath - 8), (ax + atw + 14, ay + 8), (255, 255, 255), 1)
                cv2.putText(frame, alert_badge, (ax + 4, ay - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Right Blind Zone (Amber / Orange Overlay)
        if self.zone_mode in ("right", "both") and len(self.zones.get("right", [])) >= 3:
            pts_r = np.array([[int(x * w), int(y * h)] for x, y in self.zones["right"]], dtype=np.int32)
            color_r = (30, 40, 240) if (right_occupied and self.indicator_right) else (
                (20, 140, 225) if right_occupied else (20, 110, 180)
            )
            cv2.fillPoly(overlay, [pts_r], color_r)
            cv2.polylines(frame, [pts_r], True, (30, 160, 245), 2, cv2.LINE_AA)

            # Right Zone Label Badge
            rx, ry = pts_r[0]
            badge_text = "Right Blind Zone"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (rx - tw - 18, ry + 5), (rx - 5, ry + th + 16), (20, 140, 225), -1)
            cv2.putText(frame, badge_text, (rx - tw - 12, ry + th + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 20, 30), 1, cv2.LINE_AA)

            if right_occupied:
                ax, ay = int(0.60 * w), int(0.48 * h)
                alert_badge = "! CRITICAL BLIND SPOT ALERT!" if self.indicator_right else "! ADVISORY: ROAD USER IN ZONE"
                (atw, ath), _ = cv2.getTextSize(alert_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                badge_bg = (20, 20, 220) if self.indicator_right else (15, 120, 220)
                cv2.rectangle(frame, (ax - 4, ay - ath - 8), (ax + atw + 14, ay + 8), badge_bg, -1)
                cv2.rectangle(frame, (ax - 4, ay - ath - 8), (ax + atw + 14, ay + 8), (255, 255, 255), 1)
                cv2.putText(frame, alert_badge, (ax + 4, ay - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlays with smooth alpha
        alpha = 0.28 if (left_occupied or right_occupied) else 0.18
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # 3. Futuristic HUD Corner Reticles (L-brackets on camera corners)
        hud_col = (210, 235, 255)
        arm = 18
        cv2.line(frame, (14, 14), (14 + arm, 14), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (14, 14), (14, 14 + arm), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 14, 14), (w - 14 - arm, 14), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 14, 14), (w - 14, 14 + arm), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (14, h - 14), (14 + arm, h - 14), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (14, h - 14), (14, h - 14 - arm), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 14, h - 14), (w - 14 - arm, h - 14), hud_col, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 14, h - 14), (w - 14, h - 14 - arm), hud_col, 2, cv2.LINE_AA)

        # 4. Side Mirror Reference Indicator (Green box on left mirror area)
        mx1, my1, mx2, my2 = int(0.12 * w), int(0.38 * h), int(0.24 * w), int(0.56 * h)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (80, 220, 120), 2, cv2.LINE_AA)

        # 5. Detected Road Users Bounding Boxes with Motion Vectors
        for d in detections:
            x1, y1, x2, y2 = d.box
            is_alert = d.in_zone
            is_crit = is_alert and (
                ("right" in d.zone_name.lower() and self.indicator_right) or
                ("left" in d.zone_name.lower() and self.indicator_left)
            )

            box_col = (30, 30, 240) if is_crit else ((20, 140, 240) if is_alert else (210, 160, 30))

            # Outer subtle glow
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 100, 100) if is_crit else (255, 200, 80), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2, cv2.LINE_AA)

            # Motion Vector Arrow (Visualizing vehicle closing velocity)
            vx, vy = d.velocity
            if d.speed_px > 1.2:
                cx, cy = d.center
                tip_x = int(cx + max(-50, min(50, vx * 5)))
                tip_y = int(cy + max(-50, min(50, vy * 5)))
                cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y), (0, 255, 255), 2, tipLength=0.35)

            # Bounding box tag with motion state
            tag = f"{d.name} {d.confidence:.0%}"
            if d.motion_state != "STEADY":
                tag += f" · {d.motion_state}"
            if is_crit:
                tag += " · CRITICAL"
            elif is_alert:
                tag += " · ADVISORY"

            (ttw, tth), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ty = max(0, y1 - tth - 8)
            cv2.rectangle(frame, (x1, ty), (x1 + ttw + 10, ty + tth + 8), box_col, -1)
            cv2.putText(frame, tag, (x1 + 5, ty + tth + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (255, 255, 255) if (is_alert or is_crit) else (10, 20, 30), 1, cv2.LINE_AA)

        # 6. Interactive vertex handles when in editing mode
        if self.zone_editing:
            for zname, zpts in self.zones.items():
                for vx, vy in zpts:
                    px, py = int(vx * w), int(vy * h)
                    cv2.circle(frame, (px, py), 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 9, (245, 158, 11), 2, cv2.LINE_AA)

    # ==========================================
    # METRICS, ALERTS & TIMELINE LOGIC
    # ==========================================
    def _set_alert(self, state: str, details: str, is_critical: bool = False) -> None:
        if is_critical:
            self.alert_card.configure(fg_color=ALERT_RED_BG, border_color=ALERT_RED_BORDER)
            self.alert_title.configure(text=state, text_color=ALERT_RED)
            self.alert_sub.configure(text=details, text_color="#fda4af")
            self.alert_icon_label.configure(text="⚠️")
        elif "ADVISORY" in state:
            self.alert_card.configure(fg_color=ADVISORY_AMBER_BG, border_color=ADVISORY_AMBER_BORDER)
            self.alert_title.configure(text=state, text_color=ADVISORY_AMBER)
            self.alert_sub.configure(text=details, text_color="#fde68a")
            self.alert_icon_label.configure(text="⚠️")
        else:
            self.alert_card.configure(fg_color=SAFE_GREEN_BG, border_color=SAFE_GREEN_BORDER)
            self.alert_title.configure(text="ALL CLEAR", text_color=SAFE_GREEN)
            self.alert_sub.configure(text="No road users in warning zone", text_color="#a7f3d0")
            self.alert_icon_label.configure(text="🛡️")

    def _update_metrics(self, detections: list[Detection]) -> None:
        num_veh = sum(d.name in VEHICLES for d in detections)
        num_ped = sum(d.name == "Person" for d in detections)
        num_two = sum(d.name in TWO_WHEELERS for d in detections)

        d_veh = max(0, num_veh - self.last_metrics["vehicles"])
        d_ped = max(0, num_ped - self.last_metrics["pedestrians"])
        d_two = max(0, num_two - self.last_metrics["two_wheelers"])

        self.last_metrics = {"vehicles": num_veh, "pedestrians": num_ped, "two_wheelers": num_two}

        self.card_vehicles["val"].configure(text=str(num_veh))
        self.card_vehicles["delta"].configure(text=f"↑ +{d_veh}" if d_veh > 0 else "—")

        self.card_pedestrians["val"].configure(text=str(num_ped))
        self.card_pedestrians["delta"].configure(text=f"↑ +{d_ped}" if d_ped > 0 else "—")

        self.card_two_wheelers["val"].configure(text=str(num_two))
        self.card_two_wheelers["delta"].configure(text=f"↑ +{d_two}" if d_two > 0 else "—")

    def _record_event(self, detection: Detection, event_type: str, status: str,
                      is_alert: bool = False, snapshot_path: str = "") -> None:
        self.store.add_event(
            self.session_id, detection.track_id, detection.name,
            detection.confidence, event_type, status, self.source_name,
            snapshot_path=snapshot_path,
        )
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self._add_timeline_entry(stamp, event_type, is_alert, snapshot_path)

    def _add_timeline_entry(self, stamp: str, event_text: str, is_alert: bool = False, snapshot_path: str = "") -> None:
        if self.timeline_empty is not None and self.timeline_empty.winfo_exists():
            self.timeline_empty.destroy()
            self.timeline_empty = None

        row = ctk.CTkFrame(self.timeline_scroll, fg_color="transparent", corner_radius=6,
                           border_width=1, border_color="#15263c")
        row.pack(fill="x", pady=2, padx=4)
        row.grid_columnconfigure(1, weight=1)

        # Dot or alert badge
        if is_alert:
            icon_lbl = ctk.CTkLabel(row, text="⚠️", font=ctk.CTkFont(size=11))
            icon_lbl.grid(row=0, column=0, padx=(0, 6))
            text_color = ALERT_RED
        else:
            icon_lbl = ctk.CTkLabel(row, text="●", text_color=TEXT_ACCENT, font=ctk.CTkFont(size=9))
            icon_lbl.grid(row=0, column=0, padx=(2, 8))
            text_color = TEXT_LIGHT

        # Event Description
        full_text = f"{stamp}  {event_text}"
        ctk.CTkLabel(row, text=full_text, text_color=text_color,
                     font=ctk.CTkFont(size=10, weight="bold" if is_alert else "normal")).grid(row=0, column=1, sticky="w")

        # Snapshot indicator pill
        if snapshot_path:
            snap_tag = ctk.CTkLabel(row, text="📷 Snap", text_color="#38bdf8", fg_color="#15263c",
                                    corner_radius=4, font=ctk.CTkFont(size=8, weight="bold"))
            snap_tag.grid(row=0, column=2, padx=4)

        # Row interactive hover highlight
        hover_bg = "#15263c" if is_alert else "#15263c"
        hover_border = ALERT_RED if is_alert else "#1b304c"
        make_card_interactive(row, "transparent", hover_bg, "#15263c", hover_border)

        # Keep timeline clean (max 50 rows)
        children = self.timeline_scroll.winfo_children()
        if len(children) > 50:
            children[0].destroy()

    def _clear_zone_tracks(self) -> None:
        for detection in self.zone_tracks.values():
            self._record_event(detection, "Zone cleared", "NORMAL")
        self.zone_tracks.clear()
        self.alert_active = False
        self.alert_is_critical = False

    # ==========================================
    # INTERACTIVE ZONE DRAGGING
    # ==========================================
    def _widget_to_normalized(self, event) -> tuple[float, float] | None:
        if not self.display_geometry or not all(self.frame_size):
            return None
        ox, oy, scale = self.display_geometry
        x = (event.x - ox) / scale / self.frame_size[0]
        y = (event.y - oy) / scale / self.frame_size[1]
        if 0 <= x <= 1 and 0 <= y <= 1:
            return (x, y)
        return None

    def _zone_press(self, event) -> None:
        if not self.zone_editing:
            return
        pt = self._widget_to_normalized(event)
        if pt is None:
            return
        ox, oy, scale = self.display_geometry
        w, h = self.frame_size
        px, py = event.x, event.y

        candidates = []
        for zname, pts in self.zones.items():
            for idx, (x, y) in enumerate(pts):
                dist_sq = (ox + x * w * scale - px) ** 2 + (oy + y * h * scale - py) ** 2
                candidates.append((dist_sq, zname, idx))

        best = min(candidates, default=(10**9, "", -1))
        if best[0] <= 24**2:
            self.drag_vertex = (best[1], best[2])
        else:
            target_zone = "right" if pt[0] > 0.5 else "left"
            self.zones[target_zone].append(pt)
            self.drag_vertex = (target_zone, len(self.zones[target_zone]) - 1)
            if self.worker:
                self.worker.zones = {k: list(v) for k, v in self.zones.items()}

    def _zone_drag(self, event) -> None:
        if not self.zone_editing or not self.drag_vertex:
            return
        pt = self._widget_to_normalized(event)
        if pt is not None:
            zname, idx = self.drag_vertex
            self.zones[zname][idx] = pt
            if self.worker:
                self.worker.zones = {k: list(v) for k, v in self.zones.items()}

    def _zone_release(self, _event) -> None:
        self.drag_vertex = None

    def _zone_remove_vertex(self, event) -> None:
        if not self.zone_editing or not self.display_geometry:
            return
        ox, oy, scale = self.display_geometry
        w, h = self.frame_size
        px, py = event.x, event.y

        candidates = []
        for zname, pts in self.zones.items():
            if len(pts) > 3:
                for idx, (x, y) in enumerate(pts):
                    dist_sq = (ox + x * w * scale - px) ** 2 + (oy + y * h * scale - py) ** 2
                    candidates.append((dist_sq, zname, idx))

        best = min(candidates, default=(10**9, "", -1))
        if best[0] <= 24**2:
            zname, idx = best[1], best[2]
            self.zones[zname].pop(idx)
            if self.worker:
                self.worker.zones = {k: list(v) for k, v in self.zones.items()}

    def _close(self) -> None:
        self.stop()
        self.destroy()


if __name__ == "__main__":
    app = BlindSpotApp()
    if "--demo" in sys.argv:
        app.after(300, app.run_demo)
    app.mainloop()




