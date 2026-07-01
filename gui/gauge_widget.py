"""
gui/gauge_widget.py

Custom analogue gauge widget drawn entirely with QPainter.

Displays:
  - A semicircular arc dial (like a speedometer / analogue meter)
  - Colour zones: green (0-33%), amber (33-66%), red (66-100%)
  - A needle that points to the current anomaly score
  - Numeric value text in the centre
  - The sensor's current real value above the dial
  - A status LED indicator (circle) that is green/amber/red
"""

from __future__ import annotations
import math
from typing import Optional

from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QColor, QConicalGradient, QFont, QLinearGradient,
    QPainter, QPainterPath, QPen, QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from gui.styles import (
    STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL, STATUS_IDLE,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD, BORDER_COLOR,
)
from model.anomaly_detector import AnomalyLevel


class GaugeWidget(QWidget):
    """
    Analogue gauge meter + status LED.

    Args:
        title:       Sensor name shown above the gauge.
        unit:        Physical unit (°C, bar, L/min).
        color:       Accent colour for the needle and title.
    """

    # Dial sweeps from 225° to -45° (total 270°) — standard gauge layout
    _START_ANGLE  = 225   # degrees (Qt: 0 = 3 o'clock, CCW positive)
    _SWEEP_ANGLE  = 270   # total sweep

    def __init__(
        self,
        title: str,
        unit: str,
        color: str = "#58A6FF",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._unit  = unit
        self._color = QColor(color)

        self._actual_value: float   = 0.0
        self._predicted_value: float = 0.0
        self._anomaly_pct: float    = 0.0
        self._inference_ms: float   = 0.0
        self._model_name: str       = "—"
        self._level: AnomalyLevel   = AnomalyLevel.NORMAL
        self._is_idle: bool         = True

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(180, 200)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update_values(
        self,
        actual: float,
        predicted: float,
        anomaly_pct: float,
        level: AnomalyLevel,
        inference_ms: float,
        model_name: str,
    ) -> None:
        self._actual_value    = actual
        self._predicted_value = predicted
        self._anomaly_pct     = max(0.0, min(100.0, anomaly_pct))
        self._level           = level
        self._inference_ms    = inference_ms
        self._model_name      = model_name
        self._is_idle         = False
        self.update()

    def set_idle(self) -> None:
        self._is_idle     = True
        self._anomaly_pct = 0.0
        self._level       = AnomalyLevel.NORMAL
        self.update()

    # ------------------------------------------------------------------ #
    # Paint                                                                #
    # ------------------------------------------------------------------ #

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        w, h = self.width(), self.height()
        cx = w / 2

        # ── LED dot (top-right corner) ───────────────────────────────
        led_r = 8
        led_x = w - led_r - 10
        led_y = led_r + 10
        self._draw_led(painter, led_x, led_y, led_r)

        # ── Sensor title (top-left) ──────────────────────────────────
        title_font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(10, 0, w - 30, 28, Qt.AlignLeft | Qt.AlignVCenter,
                         self._title.upper())

        # ── Actual value (large, below title) ────────────────────────
        val_text = f"{self._actual_value:.1f} {self._unit}" if not self._is_idle else "— " + self._unit
        val_font = QFont("Segoe UI", 14, QFont.Bold)
        painter.setFont(val_font)
        painter.setPen(self._color)
        painter.drawText(0, 22, w, 30, Qt.AlignCenter, val_text)

        # ── Gauge dial ────────────────────────────────────────────────
        gauge_top   = 56
        gauge_size  = min(w - 20, h - gauge_top - 60)
        gauge_rect  = QRectF(
            cx - gauge_size / 2, gauge_top,
            gauge_size, gauge_size
        )
        self._draw_gauge(painter, gauge_rect)

        # ── Predicted + inference row (below gauge) ───────────────────
        info_y = int(gauge_rect.bottom()) + 6
        small_font = QFont("Segoe UI", 7)
        painter.setFont(small_font)
        painter.setPen(QColor(TEXT_SECONDARY))

        pred_text = f"Pred: {self._predicted_value:.1f}" if not self._is_idle else "Pred: —"
        ms_text   = f"{self._inference_ms:.1f} ms"     if not self._is_idle else "—"

        painter.drawText(0, info_y, w // 2, 16, Qt.AlignCenter, pred_text)
        painter.drawText(w // 2, info_y, w // 2, 16, Qt.AlignCenter, ms_text)

        # ── Model name ────────────────────────────────────────────────
        model_font = QFont("Segoe UI", 7)
        painter.setFont(model_font)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(0, info_y + 16, w, 14,
                         Qt.AlignCenter, self._model_name)

        painter.end()

    # ------------------------------------------------------------------ #
    # Drawing helpers                                                      #
    # ------------------------------------------------------------------ #

    def _draw_led(self, p: QPainter, cx: float, cy: float, r: float) -> None:
        """Draw the coloured status LED circle."""
        if self._is_idle:
            base = QColor(STATUS_IDLE)
        else:
            base = {
                AnomalyLevel.NORMAL:   QColor(STATUS_NORMAL),
                AnomalyLevel.WARNING:  QColor(STATUS_WARNING),
                AnomalyLevel.CRITICAL: QColor(STATUS_CRITICAL),
            }[self._level]

        # Outer glow
        glow = QRadialGradient(cx, cy, r * 2.5)
        glow_color = QColor(base)
        glow_color.setAlpha(60)
        glow.setColorAt(0, glow_color)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx, cy), r * 2.5, r * 2.5)

        # Main circle with radial gradient (3D effect)
        grad = QRadialGradient(cx - r * 0.25, cy - r * 0.25, r)
        bright = QColor(base)
        bright = bright.lighter(140)
        grad.setColorAt(0, bright)
        grad.setColorAt(1, base.darker(120))
        p.setBrush(grad)
        p.setPen(QPen(base.darker(150), 1))
        p.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_gauge(self, p: QPainter, rect: QRectF) -> None:
        """Draw the complete analogue gauge into rect."""
        cx = rect.center().x()
        cy = rect.center().y()
        r  = rect.width() / 2

        # ── Background ring ──────────────────────────────────────────
        p.setPen(QPen(QColor(BORDER_COLOR), 2))
        p.setBrush(QColor(BG_CARD).darker(115))
        p.drawEllipse(rect)

        # ── Colour arc (track) ───────────────────────────────────────
        arc_width = max(10, r * 0.18)
        pen_rect  = rect.adjusted(arc_width / 2, arc_width / 2,
                                  -arc_width / 2, -arc_width / 2)

        segments = [
            (0,   33,  QColor(STATUS_NORMAL)),
            (33,  66,  QColor(STATUS_WARNING)),
            (66,  100, QColor(STATUS_CRITICAL)),
        ]
        for seg_start, seg_end, colour in segments:
            angle_start = self._START_ANGLE - (seg_start / 100) * self._SWEEP_ANGLE
            angle_span  = -((seg_end - seg_start) / 100) * self._SWEEP_ANGLE
            colour.setAlpha(90)
            pen = QPen(colour, arc_width, Qt.SolidLine, Qt.FlatCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(pen_rect,
                      int(angle_start * 16),
                      int(angle_span  * 16))

        # ── Active fill arc (shows current anomaly level) ────────────
        if not self._is_idle and self._anomaly_pct > 0:
            active_colour = self._level_colour()
            active_colour.setAlpha(220)
            active_pen = QPen(active_colour, arc_width,
                              Qt.SolidLine, Qt.FlatCap)
            p.setPen(active_pen)
            active_span = -(self._anomaly_pct / 100) * self._SWEEP_ANGLE
            p.drawArc(pen_rect,
                      int(self._START_ANGLE * 16),
                      int(active_span * 16))

        # ── Tick marks ───────────────────────────────────────────────
        for i in range(11):
            pct   = i * 10
            angle = math.radians(self._START_ANGLE - (pct / 100) * self._SWEEP_ANGLE)
            is_major = (i % 5 == 0)
            tick_len = r * (0.16 if is_major else 0.10)
            inner_r  = r - arc_width - tick_len - 4
            outer_r  = r - arc_width - 4
            x1 = cx + inner_r * math.cos(angle)
            y1 = cy - inner_r * math.sin(angle)
            x2 = cx + outer_r * math.cos(angle)
            y2 = cy - outer_r * math.sin(angle)
            tick_color = QColor(TEXT_SECONDARY if is_major else "#30363D")
            p.setPen(QPen(tick_color, 1.5 if is_major else 0.8))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Major tick labels (0, 50, 100)
            if i in (0, 5, 10):
                label_r = inner_r - 12
                lx = cx + label_r * math.cos(angle) - 10
                ly = cy - label_r * math.sin(angle) - 7
                lbl_font = QFont("Segoe UI", 6)
                p.setFont(lbl_font)
                p.setPen(QColor(TEXT_SECONDARY))
                p.drawText(QRectF(lx, ly, 20, 14),
                           Qt.AlignCenter, str(pct))

        # ── Needle ───────────────────────────────────────────────────
        needle_pct   = self._anomaly_pct if not self._is_idle else 0.0
        needle_angle = math.radians(
            self._START_ANGLE - (needle_pct / 100) * self._SWEEP_ANGLE
        )
        needle_len = r - arc_width - 8
        tip_x = cx + needle_len * math.cos(needle_angle)
        tip_y = cy - needle_len * math.sin(needle_angle)

        # Shadow
        shadow_pen = QPen(QColor(0, 0, 0, 80), 3, Qt.SolidLine, Qt.RoundCap)
        p.setPen(shadow_pen)
        p.drawLine(QPointF(cx + 1, cy + 1), QPointF(tip_x + 1, tip_y + 1))

        # Main needle
        needle_colour = self._level_colour() if not self._is_idle else QColor(STATUS_IDLE)
        needle_pen = QPen(needle_colour, 2.5, Qt.SolidLine, Qt.RoundCap)
        p.setPen(needle_pen)
        p.drawLine(QPointF(cx, cy), QPointF(tip_x, tip_y))

        # Needle pivot circle
        pivot_r = r * 0.07
        grad = QRadialGradient(cx, cy, pivot_r)
        grad.setColorAt(0, QColor(TEXT_PRIMARY))
        grad.setColorAt(1, QColor(BORDER_COLOR))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), pivot_r, pivot_r)

        # ── Centre text: anomaly % ───────────────────────────────────
        pct_font = QFont("Segoe UI", int(r * 0.16), QFont.Bold)
        p.setFont(pct_font)
        colour = self._level_colour() if not self._is_idle else QColor(STATUS_IDLE)
        p.setPen(colour)
        pct_text = f"{self._anomaly_pct:.1f}%" if not self._is_idle else "—"
        text_rect = QRectF(cx - r * 0.4, cy + r * 0.15, r * 0.8, r * 0.3)
        p.drawText(text_rect, Qt.AlignCenter, pct_text)

        # Sub-label "ANOMALY"
        lbl_font = QFont("Segoe UI", int(r * 0.085))
        p.setFont(lbl_font)
        p.setPen(QColor(TEXT_SECONDARY))
        lbl_rect = QRectF(cx - r * 0.4, cy + r * 0.35, r * 0.8, r * 0.2)
        p.drawText(lbl_rect, Qt.AlignCenter, "ANOMALY")

    def _level_colour(self) -> QColor:
        return {
            AnomalyLevel.NORMAL:   QColor(STATUS_NORMAL),
            AnomalyLevel.WARNING:  QColor(STATUS_WARNING),
            AnomalyLevel.CRITICAL: QColor(STATUS_CRITICAL),
        }[self._level]

    def sizeHint(self) -> QSize:
        return QSize(220, 240)
