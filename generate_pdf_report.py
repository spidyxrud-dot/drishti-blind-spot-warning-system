"""DRISHTI Project Safety & Architecture PDF Report Generator.

Generates a comprehensive, executive-ready PDF report covering:
1. Executive Summary & Heavy Vehicle Blind Spot Problem.
2. Current Prototype Features & Capabilities (Software-Only Prototype).
3. System Architecture & Detection Pipeline.
4. Embedded Visual Proof & Screenshots.
5. The Next Step: Phase 2 Hardware Integration Roadmap (ESP32/Arduino mirror LEDs,
   CAN-bus, haptic vibration, radar-camera fusion).
Written in simple, clear, and accessible language.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_PDF = BASE_DIR / "DRISHTI_Blind_Spot_System_Report.pdf"


class NumberedCanvas(canvas.Canvas):
    """Adds professional running header and page numbering."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Don't draw header on cover page (page 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "DRISHTI · Intelligent Blind-Spot Warning System · Prototype Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer on all pages
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, footer_text)
        self.drawString(54, 32, "Confidential & Proprietary · Safer Roads, Stronger India")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        self.restoreState()


def build_pdf_report(output_path: Path | str = OUTPUT_PDF) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0f2942")     # Deep navy
    SECONDARY = colors.HexColor("#f59e0b")   # Saffron / Amber
    TEXT_DARK = colors.HexColor("#1e293b")   # Slate 800
    TEXT_MUTED = colors.HexColor("#475569")  # Slate 600
    BG_CARD = colors.HexColor("#f8fafc")     # Light slate
    BORDER_COL = colors.HexColor("#e2e8f0")
    ALERT_RED = colors.HexColor("#dc2626")
    SAFE_GREEN = colors.HexColor("#16a34a")

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=PRIMARY,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        textColor=SECONDARY,
        spaceAfter=18,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=TEXT_DARK,
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=TEXT_DARK,
        leftIndent=14,
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=14,
        textColor=PRIMARY,
    )

    story = []

    # =========================================================================
    # COVER / HEADER BANNER
    # =========================================================================
    logo_path = ASSETS_DIR / "drishti_logo.png"
    if logo_path.exists():
        story.append(Image(str(logo_path), width=4.8 * inch, height=0.55 * inch))
        story.append(Spacer(1, 14))

    story.append(Paragraph("DRISHTI: Intelligent Blind-Spot Warning System", title_style))
    story.append(Paragraph("Advanced Computer Vision & ADAS Safety Prototype for Heavy Commercial Vehicles", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceAfter=14))

    meta_table_data = [
        [
            Paragraph("<b>Document Version:</b> 1.0 (Prototype Evaluation)", body_style),
            Paragraph(f"<b>Date:</b> {datetime.now():%d %B %Y}", body_style),
        ],
        [
            Paragraph("<b>System Core:</b> YOLOv8n + ByteTrack + Dual-Zone Vision", body_style),
            Paragraph("<b>Target Domain:</b> Heavy Trucks, Buses, Commercial Fleets", body_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[240, 260])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COL),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Executive Summary Card
    summary_text = (
        "<b>Executive Summary:</b><br/>"
        "Large commercial vehicles such as freight trucks, tankers, and transit buses suffer from dangerous blind spots "
        "along their left and right sides. Smaller vulnerable road users—especially two-wheelers, cyclists, and pedestrians—"
        "frequently disappear from mirror sightlines. <b>DRISHTI</b> is an intelligent, software-powered safety system "
        "that uses real-time computer vision to monitor vehicle blind zones, track surrounding traffic, correlate "
        "turn signal indicators, and deliver immediate visual and audible warnings to prevent side-swipe and turn collisions."
    )
    card_table = Table([[Paragraph(summary_text, body_style)]], colWidths=[500])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e0f2fe")),
        ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#0284c7")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 16))

    # =========================================================================
    # SECTION 1: THE REAL-WORLD PROBLEM
    # =========================================================================
    story.append(Paragraph("1. The Heavy Vehicle Blind-Spot Crisis", h1_style))
    p1 = (
        "Every year, thousands of fatal accidents occur when heavy trucks make left or right turns across lanes occupied "
        "by motorcycles or bicycles. The driver sits high in the cab, and traditional side mirrors have physical optical limits "
        "creating significant invisible cones (often spanning 2 to 4 meters wide beside the trailer). "
        "In developing urban road conditions, two-wheelers regularly weave past heavy vehicles precisely through these blind zones. "
        "A reliable system must distinguish between normal traffic and imminent collision hazards without overwhelming the driver with false alarms."
    )
    story.append(Paragraph(p1, body_style))
    story.append(Spacer(1, 10))

    # Problem Matrix Table
    prob_data = [
        ["Blind Zone Region", "Hazard Level", "Real-World Vulnerability", "DRISHTI Solution"],
        ["Right Side Flank", "Extreme", "Motorcyclists overtaking on the right or filtering at intersections", "Right Blind Zone polygonal tracking"],
        ["Left Side / Mirror", "High", "Pedestrians and cyclists crossing near the front-left cab corner", "Left Blind Zone & mirror reference box"],
        ["Intersection Turns", "Critical", "Truck turning into a lane where a two-wheeler is traveling parallel", "Two-Stage Turn Signal collision escalation"],
        ["Night & Heavy Rain", "High", "Glare and darkness blinding conventional optical mirrors", "CLAHE adaptive contrast enhancement"],
    ]
    prob_table = Table(prob_data, colWidths=[110, 75, 175, 140])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))

    # =========================================================================
    # SECTION 2: WHAT WE HAVE BUILT (CURRENT PROTOTYPE)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. What We Have Built (Current Software Prototype)", h1_style))
    p2 = (
        "The current DRISHTI prototype is a fully functioning, on-device software ADAS application. "
        "It runs locally on Windows without needing any cloud connection, specialized AI servers, or external hardware. "
        "Here are the core components built into this prototype:"
    )
    story.append(Paragraph(p2, body_style))

    features = [
        ("High-Speed Object Tracking (YOLOv8n + ByteTrack)",
         "Identifies and tracks Vehicles (Cars, Trucks, Buses), Pedestrians, and Two-Wheelers (Motorcycles, Bicycles) simultaneously with unique persistent track IDs."),

        ("Dynamic Dual-Zone Image Geometry",
         "Monitors both Left and Right blind zones concurrently with user-adjustable polygon boundaries, lateral width scaling (50% to 100%), and sensitivity sliders."),

        ("Automotive Two-Stage Warning Logic (UN ECE R151 Standard)",
         "<b>Stage 1 (Advisory - Amber):</b> Visual notification when a vehicle is in the blind spot but turn signals are OFF.<br/>"
         "<b>Stage 2 (Critical Emergency - Flashing Red):</b> Triggers rapid double-beep sirens and high-contrast HUD warnings when the driver activates the turn signal toward an occupied blind zone."),

        ("Vehicle Speed & Stationary Noise Suppression",
         "Includes vehicle speed simulation (0–90 km/h). When the truck is stationary at a traffic light (< 5 km/h), noisy false audible alarms are automatically silenced."),

        ("Motion Velocity Vectors & Direction of Approach",
         "Calculates frame-by-frame velocity vectors (Δx, Δy) to distinguish vehicles closing in from behind ('APPROACHING') versus vehicles moving away ('RECEDING')."),

        ("Night & Adverse Weather Vision Enhancement (CLAHE)",
         "One-click adaptive histogram equalization and gamma correction to cut through headlight glare, heavy rain, and nighttime shadows."),

        ("Black-Box Incident Snapshot Recording",
         "Automatically captures high-resolution photo evidence of every blind-spot alert and saves it to a local snapshot vault with one-click review in the History tab."),

        ("Full-Screen Driver Kiosk Mode (F11)",
         "A simplified, distraction-free in-cabin HUD mode displaying only live video, mirror indicators, speed, and emergency alerts."),
    ]

    for title, desc in features:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 12))

    # Sample Prototype Screenshot / Visual Box
    sample_img_path = ASSETS_DIR / "rendered_feed_sample.png"
    if sample_img_path.exists():
        story.append(Paragraph("<b>Figure 1: Live Dual-Zone Detection Feed with Real-Time Bounding Boxes and HUD</b>", h2_style))
        story.append(Image(str(sample_img_path), width=5.5 * inch, height=3.4 * inch))
        story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 3: SYSTEM ARCHITECTURE & TELEMETRY
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Technical Architecture & Verification", h1_style))
    p3 = (
        "The system employs a decoupled, multi-threaded producer-consumer architecture. "
        "The computer vision inference pipeline runs on a dedicated worker thread, ensuring the user interface "
        "and audio warning system remain buttery-smooth with zero frame drops."
    )
    story.append(Paragraph(p3, body_style))

    arch_data = [
        ["Layer", "Module", "Responsibility", "Key Performance Metric"],
        ["Input", "OpenCV Video Engine", "Webcam indices 0–5 or recorded MP4/WebM video feed", "DirectShow low-latency capture"],
        ["Vision AI", "Ultralytics YOLOv8n", "Deep learning road user feature extraction", "640px / 416px Fast inference"],
        ["Tracking", "ByteTrack Algorithm", "Associates bounding boxes across frames", "IoU fallback, persistent IDs"],
        ["Spatial Math", "cv2.pointPolygonTest", "Ray-casting point-in-polygon zone containment", "< 0.5 ms evaluation time"],
        ["Decision", "Two-Stage State Machine", "Correlates Turn Indicators + Speed + Zones", "Instant Stage 1 / Stage 2 dispatch"],
        ["Alert Audio", "Pygame Frequency Synth", "Multi-frequency advisory chimes & pulsed sirens", "Non-blocking audio thread"],
        ["Data Vault", "SQLite 3 Database", "Logs every distinct safety event and snapshot path", "Indexed timeline, CSV export"],
    ]
    arch_table = Table(arch_data, colWidths=[65, 110, 205, 120])
    arch_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 14))

    # Full Product Reference Visual
    banner_img_path = ASSETS_DIR / "ref_banner.png"
    if banner_img_path.exists():
        story.append(Paragraph("<b>Figure 2: DRISHTI Initiative Emblem — Dedicated to Safer Indian Highways</b>", h2_style))
        story.append(Image(str(banner_img_path), width=4.5 * inch, height=1.6 * inch))
        story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 4: THE NEXT STEP (PHASE 2 HARDWARE ROADMAP)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. The Next Step: Phase 2 Physical Hardware Integration", h1_style))
    p4 = (
        "Now that the complete software intelligence, ADAS logic, and alert state machine are validated, "
        "the next phase transitions this prototype from a desktop demonstration to an <b>in-vehicle physical hardware installation</b>. "
        "The following hardware modules are planned for the Phase 2 build:"
    )
    story.append(Paragraph(p4, body_style))

    hw_steps = [
        ("1. Physical Exterior Mirror Warning LEDs (Microcontroller Relay)",
         "Mount high-intensity amber/red LED warning pods on or inside the truck's exterior side-mirror housings. "
         "An onboard ESP32 or Arduino microcontroller connects via USB Serial or Wi-Fi to receive immediate "
         "<code>ALERT_LEFT_ON</code> or <code>ALERT_RIGHT_CRITICAL</code> commands, illuminating the mirror warning light right in the driver's peripheral vision."),

        ("2. Tactile Haptic Steering & Seat Vibration",
         "Visual alerts can be missed during complex maneuvers. By connecting eccentric rotating mass (ERM) vibration motors "
         "to the driver's seat cushion or steering wheel, the system delivers direct tactile feedback corresponding to the side of the impending hazard."),

        ("3. Direct Vehicle CAN-Bus (J1939 / OBD-II) Integration",
         "Replace simulated turn signals and sliders with physical vehicle data. Using an MCP2515 CAN-bus transceiver, "
         "the system will directly read the truck's actual turn indicator stalk position, steering wheel angle, and transmission speedometer."),

        ("4. Radar-Camera Sensor Fusion (77GHz mmWave Radar)",
         "Combine the wide-angle camera with an automotive 77GHz millimeter-wave radar sensor mounted on the trailer side skirt. "
         "Cameras provide rich object classification (distinguishing bikes from cars), while radar provides millimeter-accurate distance "
         "and velocity even through blinding sandstorms, dense smog, or heavy spray."),

        ("5. Automotive Ruggedized Edge AI Unit",
         "Port the YOLOv8 and tracking pipeline onto an industrial, fanless, IP67 sealed edge AI module (such as an NVIDIA Jetson Orin Nano "
         "or NXP S32G vehicle computer) powered directly from the truck's 12V / 24V DC battery system with wide temperature tolerance (-20°C to +70°C)."),
    ]

    for step_title, step_desc in hw_steps:
        story.append(Paragraph(f"• <b>{step_title}:</b> {step_desc}", bullet_style))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 14))

    # Hardware Implementation Comparison Table
    hw_table_data = [
        ["Subsystem", "Current Phase 1 (Software Prototype)", "Phase 2 (Physical Hardware Target)"],
        ["Turn Signals", "Keyboard hotkeys (A/D) & onscreen buttons", "Direct truck CAN-bus indicator stalk sensor"],
        ["Speed Telemetry", "Interactive slider (0–90 km/h) & presets", "Vehicle Speed Sensor (VSS) / GPS pulse reader"],
        ["Visual Alert", "Onscreen HUD glowing red alert cards", "Amber/Red LED pods inside exterior side mirrors"],
        ["Audible Alert", "Laptop/PC speaker synthesized frequencies", "Cabin buzzer + haptic seat vibration actuator"],
        ["Sensors", "USB / built-in webcam or road video", "Automotive IP69K HDR camera + 77GHz mmWave radar"],
        ["Compute Unit", "Local Python / Windows PC", "Automotive fanless edge computer (Jetson / NXP)"],
    ]
    hw_table = Table(hw_table_data, colWidths=[100, 195, 205])
    hw_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(hw_table)
    story.append(Spacer(1, 16))

    # =========================================================================
    # SECTION 5: CONCLUSION & ACTION PLAN
    # =========================================================================
    story.append(Paragraph("5. Summary & Immediate Next Steps", h1_style))
    concl = (
        "<b>Conclusion:</b> The Phase 1 prototype successfully demonstrates that intelligent, vision-based blind-spot monitoring "
        "can be deployed efficiently with zero lag and high reliability. The software ADAS logic cleanly handles two-stage alert "
        "escalation, motion vectors, and stationary filtering.<br/><br/>"
        "<b>Recommended Immediate Actions:</b><br/>"
        "1. Present and validate the current software prototype using the bundled looping road demo.<br/>"
        "2. Review the incident snapshot log to evaluate real-world bounding box accuracy across various road user types.<br/>"
        "3. Procure the Phase 2 hardware bill of materials: ESP32 development board, dual 12V high-brightness amber LED pods, "
        "and an MCP2515 CAN-bus interface module for vehicle test-bench integration."
    )
    story.append(Paragraph(concl, body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    return str(output_path)


if __name__ == "__main__":
    generated_file = build_pdf_report()
    print(f"Successfully generated PDF report at: {generated_file}")
