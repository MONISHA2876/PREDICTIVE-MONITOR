"""
gui/styles.py  —  redesigned dark industrial theme
Matches the sketch: gauge meters, status dots, graphs, left sidebar.
"""

# ── Palette ──────────────────────────────────────────────────────────
BG_PRIMARY      = "#0D1117"   # near-black main background
BG_CARD         = "#161B22"   # sensor panel cards
BG_SIDEBAR      = "#0D1117"   # left sidebar
BG_HEADER       = "#161B22"   # top bar
BORDER_COLOR    = "#30363D"   # subtle borders
BORDER_ACCENT   = "#21262D"   # inner borders

ACCENT_BLUE     = "#1F6FEB"   # primary action blue
ACCENT_BLUE_DIM = "#1A4F9C"   # dimmed blue

TEXT_PRIMARY    = "#E6EDF3"   # bright white text
TEXT_SECONDARY  = "#8B949E"   # muted labels
TEXT_DIM        = "#484F58"   # very muted

# Status colours (used in Python code too)
STATUS_NORMAL   = "#3FB950"   # vivid green
STATUS_WARNING  = "#D29922"   # amber
STATUS_CRITICAL = "#F85149"   # vivid red
STATUS_IDLE     = "#484F58"   # grey

# Gauge / graph accent colours per sensor
TEMP_COLOR      = "#FF7B72"   # warm red-orange
PRES_COLOR      = "#79C0FF"   # sky blue
FLOW_COLOR      = "#56D364"   # green

GRAPH_ACTUAL    = "#58A6FF"   # actual line
GRAPH_PREDICTED = "#FF7B72"   # predicted line (warm contrast)

# ── QSS stylesheet ────────────────────────────────────────────────────
MAIN_STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    border: none;
}}

/* ── Top bar ─────────────────────────────────────────────────────── */
#TopBar {{
    background-color: {BG_HEADER};
    border-bottom: 1px solid {BORDER_COLOR};
}}

#AppTitle {{
    color: {TEXT_PRIMARY};
    font-size: 14px;
    font-weight: bold;
    letter-spacing: 2px;
}}

#SubTitle {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    letter-spacing: 1px;
}}

/* ── Sidebar ─────────────────────────────────────────────────────── */
#Sidebar {{
    background-color: {BG_SIDEBAR};
    border-right: 1px solid {BORDER_COLOR};
}}

#SectionLabel {{
    color: {TEXT_DIM};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 4px 0px 2px 0px;
}}

/* ── Sensor panel card ───────────────────────────────────────────── */
#SensorPanel {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {ACCENT_BLUE};
    color: {TEXT_PRIMARY};
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 11px;
    font-weight: bold;
    text-align: left;
}}
QPushButton:hover {{
    background-color: #388BFD;
}}
QPushButton:pressed {{
    background-color: {ACCENT_BLUE_DIM};
}}
QPushButton:disabled {{
    background-color: {BG_CARD};
    color: {TEXT_DIM};
    border: 1px solid {BORDER_COLOR};
}}
QPushButton#DangerButton {{
    background-color: #3D1A1A;
    color: {STATUS_CRITICAL};
    border: 1px solid #5C2020;
}}
QPushButton#DangerButton:hover {{
    background-color: #5C2020;
}}
QPushButton#SuccessButton {{
    background-color: #1A3D2B;
    color: {STATUS_NORMAL};
    border: 1px solid #2D6A40;
}}
QPushButton#SuccessButton:hover {{
    background-color: #2D6A40;
}}

/* ── Combo box ───────────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_PRIMARY};
    font-size: 11px;
}}
QComboBox:hover {{ border-color: {ACCENT_BLUE}; }}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {ACCENT_BLUE};
    color: {TEXT_PRIMARY};
}}

/* ── Event log ───────────────────────────────────────────────────── */
QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
    font-size: 10px;
    color: {TEXT_PRIMARY};
    font-family: "Consolas", "Courier New", monospace;
}}
QListWidget::item {{
    padding: 4px 6px;
    border-bottom: 1px solid {BORDER_ACCENT};
    border-radius: 0px;
}}
QListWidget::item:selected {{
    background-color: {BORDER_COLOR};
}}

/* ── Progress bar ────────────────────────────────────────────────── */
QProgressBar {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    background: {BG_CARD};
    text-align: center;
    height: 8px;
    font-size: 9px;
    color: {TEXT_SECONDARY};
}}
QProgressBar::chunk {{
    background-color: {ACCENT_BLUE};
    border-radius: 2px;
}}

/* ── Scrollbars ──────────────────────────────────────────────────── */
QScrollBar:vertical {{
    width: 6px;
    background: transparent;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_COLOR};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    height: 6px;
    background: transparent;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_COLOR};
    border-radius: 3px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Splitter ────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {BORDER_COLOR};
    width: 1px;
    height: 1px;
}}

/* ── Tooltips ────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
}}
"""
