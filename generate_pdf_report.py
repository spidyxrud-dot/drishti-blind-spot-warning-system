"""Create the current DRISHTI project overview PDF."""
from pathlib import Path
import cv2

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
    KeepTogether, HRFlowable,
)

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
OUT_DIR = BASE / "output" / "pdf"
TMP = BASE / "tmp" / "pdfs"
OUT = OUT_DIR / "DRISHTI_Project_Overview.pdf"
LANDING = "https://spidyxrud-dot.github.io/drishti-blind-spot-warning-system/"
REPO = "https://github.com/spidyxrud-dot/drishti-blind-spot-warning-system"
VIDEO_SOURCE = "https://commons.wikimedia.org/wiki/File:Garbsen_2019.webm"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"

NAVY = colors.HexColor("#102139")
INK = colors.HexColor("#17263A")
MUTED = colors.HexColor("#5E6B79")
ORANGE = colors.HexColor("#F3A20B")
CREAM = colors.HexColor("#F7F2E8")
PALE = colors.HexColor("#EEF2F5")
GREEN = colors.HexColor("#14805E")
RED = colors.HexColor("#B63B32")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=29, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=7))
styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=12, leading=18, textColor=MUTED, spaceAfter=7))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=NAVY, spaceBefore=3, spaceAfter=10))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.3, leading=14, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.6, leading=10.5, textColor=MUTED, spaceAfter=3))
styles.add(ParagraphStyle(name="Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=ORANGE, spaceAfter=6))
styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=8.1, leading=11, textColor=INK))
styles.add(ParagraphStyle(name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.2, leading=11, textColor=colors.white))
styles.add(ParagraphStyle(name="Link", parent=styles["BodyText"], fontSize=9, leading=14, textColor=NAVY))

def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])

def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h-9*mm, w, 9*mm, fill=1, stroke=0)
    canvas.setFillColor(ORANGE)
    canvas.rect(0, h-9*mm, 3*mm, 9*mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(16*mm, h-6*mm, "DRISHTI  /  PROJECT OVERVIEW")
    canvas.setStrokeColor(colors.HexColor("#D8DEE5"))
    canvas.line(16*mm, 15*mm, w-16*mm, 15*mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(16*mm, 10*mm, "Rudra Rahul Rajgure  ·  Educational software prototype")
    canvas.drawRightString(w-16*mm, 10*mm, f"{doc.page}")
    canvas.restoreState()

def extract_frame():
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "road_demo_frame.jpg"
    cap = cv2.VideoCapture(str(ASSETS / "demo_road.webm"))
    if not cap.isOpened():
        raise RuntimeError("Could not open bundled demo video")
    cap.set(cv2.CAP_PROP_POS_MSEC, 7000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not extract a frame from bundled demo video")
    h, w = frame.shape[:2]
    # Draw restrained example zones so the image communicates the project, not calibration.
    overlay = frame.copy()
    cv2.fillPoly(overlay, [__import__("numpy").array([[0,int(h*.28)],[int(w*.22),int(h*.23)],[int(w*.29),h],[0,h]])], (20,105,225))
    cv2.fillPoly(overlay, [__import__("numpy").array([[int(w*.78),int(h*.22)],[w,int(h*.28)],[w,h],[int(w*.72),h]])], (20,105,225))
    frame = cv2.addWeighted(overlay, .20, frame, .80, 0)
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path

def card(text, bg=PALE, width=85*mm):
    t = Table([[P(text, "Cell")]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1),bg), ("BOX",(0,0),(-1,-1),.7,colors.HexColor("#D6DEE7")),
        ("LEFTPADDING",(0,0),(-1,-1),9), ("RIGHTPADDING",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    return t

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = extract_frame()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=18*mm, bottomMargin=20*mm, title="DRISHTI Project Overview", author="Rudra Rahul Rajgure", subject="Software-only blind-spot warning prototype")
    story = []
    story += [Spacer(1, 5*mm), P("COMPUTER VISION · HEAVY VEHICLE DEMONSTRATION", "Kicker"), P("DRISHTI", "CoverTitle"), P("Intelligent Blind-Spot Warning System for Heavy Vehicles", "Sub")]
    banner = ASSETS / "india_banner.png"
    if banner.exists():
        img = Image(str(banner), width=178*mm, height=31*mm, kind="proportional")
        story += [img, Spacer(1, 4*mm)]
    else:
        story += [Spacer(1, 8*mm)]
    story.append(P("PROJECT OVERVIEW", "Kicker"))
    story.append(P("A local desktop prototype that detects road users in camera or recorded video, tracks them across frames, and highlights configurable warning regions with visual, sound, and optional spoken alerts.", "Bodyx"))
    story.append(Spacer(1, 2*mm))
    img = Image(str(frame), width=177*mm, height=99.6*mm)
    story.append(img)
    story.append(P("Bundled real-road demonstration footage with illustrative overlays. The polygons are image-space examples and do not represent calibrated truck blind spots.", "Smallx"))
    info = Table([[P("PREPARED BY", "Kicker"), P("PROJECT LINKS", "Kicker")],
                  [P("<b>Rudra Rahul Rajgure</b><br/>Educational software prototype<br/>15 November 2026", "Bodyx"),
                   P(f'<link href="{LANDING}"><u>Open the landing page</u></link><br/><link href="{REPO}"><u>View the GitHub repository</u></link>', "Link")]], colWidths=[72*mm, 105*mm])
    info.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CREAM),("BOX",(0,0),(-1,-1),.8,colors.HexColor("#E7DCC8")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story += [Spacer(1, 4*mm), info, PageBreak()]

    story += [P("01  /  WHAT THE PROJECT DOES", "Kicker"), P("A software-only road-scene system", "H1x"),
        P("DRISHTI runs on a Windows computer and analyzes either a connected camera or a recorded clip. The dashboard shows detected objects, confidence, image-space zones, current alert state, recent events, and performance indicators.")]
    flow = Table([[P("VIDEO INPUT", "CellHead"), P("DETECTION + TRACKING", "CellHead"), P("ZONE + RESPONSE", "CellHead")],
                  [P("Webcam or video file<br/>MP4 · AVI · MOV · WebM", "Cell"), P("OpenCV + YOLOv8n<br/>ByteTrack IDs", "Cell"), P("Editable polygons<br/>Visual · tone · speech", "Cell")]], colWidths=[59*mm]*3)
    flow.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(-1,-1),PALE),("GRID",(0,0),(-1,-1),.5,colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [flow, Spacer(1, 4*mm), P("Road-user classes", "H2x"), P("The detector is configured for person, bicycle, motorcycle, car, bus, and truck. Bounding boxes show the class label and confidence score. Tracking associates detections over processed frames so entries can be recorded as distinct events rather than a new event on every frame."), P("Review and records", "H2x"), P("SQLite stores detection and alert history. The dashboard includes timeline/history views, incident snapshots, object summaries, charts, and CSV export. Speed indicators distinguish AI inference speed from displayed UI update speed; recorded-video frame skipping is used to limit playback lag."), P("Inputs and local processing", "H2x"), P("The video and model run locally. Computer speakers provide the warning tone, and optional Windows speech can announce the object and its left/right image-space side. Mute, volume, video controls, camera selection, and zone presets are provided in the desktop UI."), PageBreak()]

    story += [P("02  /  WARNING BEHAVIOR", "Kicker"), P("How the two stages work", "H1x"), P("The stage labels describe application logic for demonstration. Turn indicators and vehicle speed are simulated dashboard inputs; the software is not connected to truck signals or vehicle systems.")]
    stages = Table([[P("STAGE 1  ·  ADVISORY", "CellHead"), P("STAGE 2  ·  SIMULATED PRIORITY", "CellHead")],
                    [P("A tracked road user’s bounding-box centre enters an enabled image-space warning zone. The UI marks the object and records a new zone-entry event. The dashboard uses an advisory state when the matching indicator is off.", "Cell"), P("When the simulated left or right turn indicator matches an occupied zone, the dashboard raises the warning to a critical red state and may play the critical warning tone. This is a priority demonstration, not a vehicle-integrated collision alarm.", "Cell")]], colWidths=[88.5*mm,88.5*mm])
    stages.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),ORANGE),("BACKGROUND",(1,0),(1,0),RED),("BACKGROUND",(0,1),(-1,-1),PALE),("GRID",(0,0),(-1,-1),1,colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [stages, Spacer(1, 4*mm), P("What “blind spot” means here", "H2x"), P("A zone is a polygon drawn on the video image. Its location depends on the camera view, image framing, and current preset. It is not measured in metres and has not been calibrated to a truck model, camera pose, mirrors, or road geometry. An object inside this polygon is a demo-zone entry, not proof that the object is physically inside a real truck blind spot."), P("Current limitations", "H2x"), P("The included road clip is useful for demonstrating playback and overlays, but it is not a labeled evaluation set. Detection results can vary with lighting, distance, occlusion, camera motion, inference speed, and the selected confidence threshold. Dropping stale video frames reduces lag but can miss brief entries. No accuracy, safety certification, or real-vehicle performance claim is made."), P("This prototype is entirely software-based", "H2x"), P("No microcontroller, external sensor, physical LED, or physical buzzer is required. The next project stage should improve measured evidence and usability before any real-world safety use is considered."), PageBreak()]

    story += [P("03  /  PROJECT STAGES", "Kicker"), P("Where it is now, and what comes next", "H1x"), P("Current stage · Desktop software prototype", "H2x"), P("The working application demonstrates local video input, YOLOv8n detection, ByteTrack object association, editable image-space polygons, simulated turn-indicator priority, computer audio and optional speech, event logging, snapshots, analytics, and history export. The landing page provides project information and a downloadable repository ZIP; it does not run the Python application in a browser."), P("Second stage · Validation and refinement", "H2x")]
    second = [
        [P("1. Collect and label", "CellHead"), P("Representative road videos covering road users, weather, lighting, camera angles, and occlusion; label a held-out set with clear class and zone-entry annotations.", "Cell")],
        [P("2. Measure", "CellHead"), P("Report precision, recall, missed entries, false alerts, alert delay, and performance on target computers. Keep the test set separate from model tuning.", "Cell")],
        [P("3. Review zones", "CellHead"), P("Test the image-space interaction and document camera placement assumptions. Only qualified vehicle-safety work can establish physical calibration.", "Cell")],
        [P("4. Improve", "CellHead"), P("Use findings to tune tracking, frame handling, warning timing, speech, and operator controls. Repeat validation after changes and publish the limitations with results.", "Cell")],
    ]
    tab = Table(second, colWidths=[34*mm,143*mm])
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),NAVY),("BACKGROUND",(1,0),(1,-1),PALE),("GRID",(0,0),(-1,-1),.8,colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [tab, Spacer(1, 3*mm), P("The project currently includes an evaluation command and a COCO128 smoke baseline. COCO128 is not held out for this project and cannot establish real-world reliability. A future validation report should publish the actual dataset, split, configuration, errors, and performance measurements rather than infer accuracy from a demo video."), PageBreak()]

    story += [P("04  /  GETTING STARTED", "Kicker"), P("Run the desktop demo", "H1x"), P("The public landing page is a guide and download entry point. To run the actual detector, download or clone the repository on Windows, extract it, and start the bundled launcher. The first model download requires internet access if the YOLO weights are not already present.")]
    steps = ["Download and extract the project ZIP from the GitHub repository.", "On Windows, double-click <b>run.bat</b>, or use <b>run.bat --demo</b> to start the bundled road clip.", "Choose a camera or upload an MP4, AVI, MOV, or WebM recording.", "Choose a zone preset, adjust the confidence and audio settings, then start detection.", "Use History or Analytics to review events; export a CSV when needed."]
    for i, s in enumerate(steps, 1):
        story.append(P(f"<font color='#F3A20B'><b>{i:02d}</b></font>  {s}"))
    story += [Spacer(1, 2*mm), P("Project links", "H2x")]
    links = Table([[P(f'<link href="{LANDING}"><u>DRISHTI Landing Page</u></link>', "Link"), P(f'<link href="{REPO}"><u>GitHub Repository</u></link>', "Link")]], colWidths=[88.5*mm,88.5*mm])
    links.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CREAM),("BOX",(0,0),(-1,-1),.7,colors.HexColor("#E7DCC8")),("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    story += [links, P("Demo video and reuse", "H2x"), P(f'The bundled clip is a 480p transcode of <link href="{VIDEO_SOURCE}"><u>Garbsen 2019.webm</u></link>, created by C.Suthorn and shared under <link href="{LICENSE}"><u>CC BY-SA 4.0</u></link>. Full credit is provided in the repository at assets/ATTRIBUTION.md. This source clip documents a road-safety pilot; it is not a DRISHTI test result and the project is not endorsed by the creator.'), P("Project boundaries", "H2x"), P("DRISHTI is an educational research and demonstration prototype. It is software-only, not an in-vehicle product. Its image overlays are not physically calibrated and must not be used to make driving or safety decisions."), Spacer(1, 2*mm), HRFlowable(width="100%", thickness=.8, color=ORANGE), Spacer(1, 2*mm), P("Safer Roads, Stronger India  ·  Prepared for project presentation by Rudra Rahul Rajgure", "Smallx")]
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT)

if __name__ == "__main__":
    build()
