#!/usr/bin/env python3
"""
main_gui.py
===========
TBRGS — Traffic-Based Route Guidance System
PyQt5 GUI with Folium / OpenStreetMap route visualisation.

Dependencies
------------
    pip install PyQt5 PyQtWebEngine folium scikit-learn torch

Usage
-----
    python main_gui.py
"""

import sys
import os
import math
import itertools
import heapq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit,
    QFrame, QMessageBox, QSizePolicy, QGroupBox, QScrollArea,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon

import folium

from routing import TrafficGraph, calculate_travel_time, yen_k_shortest_paths, build_graph_from_scats, DATA_PATH, predict_flows


# ===========================================================================
# Mock SCATS network — Melbourne suburban intersections (Boroondara /
# Stonnington area).  In production this is replaced by the full CSV.
# Format: scats_id -> (latitude, longitude, street_label)
# ===========================================================================


ROUTE_COLORS = ["red", "blue", "green", "purple", "orange"]
ROUTE_COLOR_HEX = {
    "red":    "#e74c3c",
    "blue":   "#2980b9",
    "green":  "#27ae60",
    "purple": "#8e44ad",
    "orange": "#e67e22",
}
CONNECTION_RADIUS_KM = 3.2   # link two sites if they are within this distance
MAX_NEIGHBOURS       = 4     # maximum outgoing edges per node

MAP_OUTPUT = os.path.join(SCRIPT_DIR, "map.html")


# ===========================================================================
# Graph helpers
# ===========================================================================


try:
    _real_graph = build_graph_from_scats(DATA_PATH)
    REAL_SCATS = {
        nid: (lat, lon, f"Intersection {nid}") 
        for nid, (lat, lon) in _real_graph.nodes.items()
    }
except Exception as e:
    print(f"Warning: Could not load real data: {e}")
    REAL_SCATS = {}

# ===========================================================================
# Background worker thread
# ===========================================================================

class RoutingWorker(QThread):
    """Run Yen's K-shortest paths in a background thread so the GUI stays
    responsive during computation."""

    finished = pyqtSignal(list)   # emits List[(path, travel_time_min)]
    error    = pyqtSignal(str)

    def __init__(self, origin: int, destination: int, model_name: str):
        super().__init__()
        self.origin      = origin
        self.destination = destination
        self.model_name  = model_name

    def run(self) -> None:
        try:
            
            graph = build_graph_from_scats(DATA_PATH)
            flow_map = predict_flows(self.model_name)

            if self.origin not in graph.nodes:
                avail = sorted(graph.nodes.keys())
                self.error.emit(
                    f"Origin SCATS {self.origin} not in network.\n"
                    f"Available sites: {avail}"
                )
                return

            if self.destination not in graph.nodes:
                avail = sorted(graph.nodes.keys())
                self.error.emit(
                    f"Destination SCATS {self.destination} not in network.\n"
                    f"Available sites: {avail}"
                )
                return

            routes = yen_k_shortest_paths(
                graph, flow_map,
                self.origin, self.destination,
                K=5,
            )

            if not routes:
                self.error.emit(
                    "No path found between the selected SCATS sites.\n"
                    "Try a different origin/destination pair."
                )
                return

            self.finished.emit(routes)

        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


# ===========================================================================
# Folium map generation
# ===========================================================================

def generate_map(routes: list, origin: int, destination: int) -> str:
    """
    Plot up to 5 routes on an OpenStreetMap base using folium.

    - Each route is drawn as a coloured PolyLine (red, blue, green, purple,
      orange) with a tooltip showing path and estimated travel time.
    - All SCATS nodes are marked: green = origin, red = destination, blue = other.
    - An HTML legend is injected into the bottom-left corner.
    - Map is saved to MAP_OUTPUT and its absolute path is returned.
    """
    involved = {n for path, _ in routes for n in path}
    lats = [REAL_SCATS[n][0] for n in involved if n in REAL_SCATS]
    lons = [REAL_SCATS[n][1] for n in involved if n in REAL_SCATS]
    center = (
        sum(lats) / len(lats) if lats else -37.858,
        sum(lons) / len(lons) if lons else 145.070,
    )

    fmap = folium.Map(
        location=list(center),
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # ---- Route polylines -------------------------------------------------
    for rank, (path, tt) in enumerate(routes):
        color  = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
        coords = [
            (REAL_SCATS[n][0], REAL_SCATS[n][1])
            for n in path if n in REAL_SCATS
        ]
        if len(coords) < 2:
            continue

        label = " → ".join(map(str, path))
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=6 if rank == 0 else 4,
            opacity=0.95 if rank == 0 else 0.75,
            dash_array=None if rank == 0 else ("10 5" if rank % 2 else None),
            tooltip=f"Route {rank + 1} [{color}]  |  {label}  |  {tt:.1f} min",
            popup=folium.Popup(
                f"<div style='font-family:sans-serif'>"
                f"<b>Route {rank + 1}</b> "
                f"<span style='color:{ROUTE_COLOR_HEX[color]}'>({color})</span><br>"
                f"<b>Path:</b> {label}<br>"
                f"<b>Est. travel time:</b> {tt:.2f} min"
                f"</div>",
                max_width=320,
            ),
        ).add_to(fmap)

        # Small direction arrows along the route
        for i in range(len(coords) - 1):
            mid_lat = (coords[i][0] + coords[i + 1][0]) / 2
            mid_lon = (coords[i][1] + coords[i + 1][1]) / 2
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=f'<div style="color:{ROUTE_COLOR_HEX[color]};'
                         f'font-size:14px;font-weight:bold;">▶</div>',
                    icon_size=(16, 16),
                    icon_anchor=(8, 8),
                ),
            ).add_to(fmap)

    # ---- SCATS site markers ----------------------------------------------
    for nid, (lat, lon, label) in REAL_SCATS.items():
        if nid == origin:
            icon = folium.Icon(color="green", icon="play", prefix="fa")
        elif nid == destination:
            icon = folium.Icon(color="red", icon="flag-checkered", prefix="fa")
        else:
            icon = folium.Icon(color="cadetblue", icon="map-marker", prefix="fa")

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<div style='font-family:sans-serif'>"
                f"<b>SCATS {nid}</b><br>{label}</div>",
                max_width=220,
            ),
            tooltip=f"SCATS {nid} — {label}",
            icon=icon,
        ).add_to(fmap)

    # ---- Legend ----------------------------------------------------------
    legend_rows = "".join(
        f'<tr><td><span style="color:{ROUTE_COLOR_HEX[ROUTE_COLORS[i]]};'
        f'font-size:18px;">&#9644;</span></td>'
        f'<td style="padding-left:6px">Route {i + 1} — '
        f'{routes[i][1]:.1f} min</td></tr>'
        for i in range(len(routes))
    )
    legend_html = f"""
    <div style="
        position: fixed; bottom: 36px; left: 36px; z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 12px 16px; border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;">
      <b style="font-size:14px">📍 Route Legend</b>
      <table style="margin-top:6px;border-collapse:collapse">
        {legend_rows}
      </table>
      <div style="margin-top:8px;font-size:12px;color:#555">
        <span style="color:green">●</span> Origin (SCATS {origin}) &nbsp;
        <span style="color:red">●</span> Destination (SCATS {destination})
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(MAP_OUTPUT)
    return os.path.realpath(MAP_OUTPUT)


# ===========================================================================
# Main Window
# ===========================================================================

class TBRGSWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TBRGS — Traffic-Based Route Guidance System")
        self.setMinimumSize(1380, 820)
        self._worker: RoutingWorker | None = None
        self._build_ui()
        self._show_default_map()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_map_panel(), stretch=1)

        self.setStyleSheet(_STYLESHEET)

    def _build_left_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setFixedWidth(370)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 26, 22, 18)
        layout.setSpacing(0)

        # --- Title ----------------------------------------------------------
        title = QLabel("🚦 TBRGS")
        title.setObjectName("appTitle")
        sub   = QLabel("Traffic-Based Route Guidance System")
        sub.setObjectName("appSubtitle")
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(12)
        layout.addWidget(_make_hline())
        layout.addSpacing(10)

        # --- SCATS site quick-reference ------------------------------------
        ref_lbl = QLabel("SCATS NETWORK NODES")
        ref_lbl.setObjectName("sectionLabel")
        layout.addWidget(ref_lbl)
        layout.addSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedHeight(148)
        scroll.setObjectName("scatsScroll")

        inner = QWidget()
        inner.setObjectName("scatsInner")
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(2)
        for nid, (lat, lon, lbl) in REAL_SCATS.items():
            row = QLabel(
                f"<b style='color:#3498db'>{nid}</b>"
                f"<span style='color:#95a5a6'> — {lbl}</span>"
            )
            row.setObjectName("scatsRow")
            row.setTextFormat(Qt.RichText)
            iv.addWidget(row)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        layout.addSpacing(10)
        layout.addWidget(_make_hline())
        layout.addSpacing(12)

        # --- Route config group --------------------------------------------
        grp = QGroupBox("Route Configuration")
        grp.setObjectName("configGroup")
        gl  = QVBoxLayout(grp)
        gl.setSpacing(8)

        gl.addWidget(_form_label("Origin SCATS Site"))
        self.origin_input = QLineEdit("2000")
        self.origin_input.setPlaceholderText("e.g. 970")
        self.origin_input.setObjectName("formInput")
        gl.addWidget(self.origin_input)

        gl.addWidget(_form_label("Destination SCATS Site"))
        self.dest_input = QLineEdit("970")
        self.dest_input.setPlaceholderText("e.g. 5000")
        self.dest_input.setObjectName("formInput")
        gl.addWidget(self.dest_input)

        gl.addWidget(_form_label("Prediction Model"))
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelCombo")
        self.model_combo.addItems(["LSTM", "GRU", "Transformer"])
        self.model_combo.setToolTip(
            "Select the trained deep-learning model to predict traffic flow.\n"
            "Falls back to mock flows (300 veh/hr) if weights are not found."
        )
        gl.addWidget(self.model_combo)

        layout.addWidget(grp)
        layout.addSpacing(12)

        # --- Find button ---------------------------------------------------
        self.find_btn = QPushButton("🔍   Find Top 5 Routes")
        self.find_btn.setObjectName("findBtn")
        self.find_btn.setFixedHeight(46)
        self.find_btn.setCursor(Qt.PointingHandCursor)
        self.find_btn.clicked.connect(self._on_find_clicked)
        layout.addWidget(self.find_btn)
        layout.addSpacing(14)

        # --- Results area --------------------------------------------------
        res_lbl = QLabel("ROUTE RESULTS")
        res_lbl.setObjectName("sectionLabel")
        layout.addWidget(res_lbl)
        layout.addSpacing(4)

        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText(
            "Route results will appear here.\n\n"
            "Select an origin, destination, and model,\n"
            "then click 'Find Top 5 Routes'."
        )
        layout.addWidget(self.results_text, stretch=1)
        layout.addSpacing(8)

        # --- Status bar ----------------------------------------------------
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setObjectName("statusLabel")
        layout.addWidget(self.status_lbl)

        return panel

    def _build_map_panel(self) -> QWidget:
        container = QWidget()
        container.setObjectName("mapContainer")
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Thin top bar
        bar = QFrame()
        bar.setObjectName("mapBar")
        bar.setFixedHeight(34)
        bl  = QHBoxLayout(bar)
        bl.setContentsMargins(14, 0, 14, 0)
        map_title = QLabel("🗺  OpenStreetMap — Route Visualisation")
        map_title.setObjectName("mapBarLabel")
        bl.addWidget(map_title)
        bl.addStretch()
        hint = QLabel("Click a route on the map for details")
        hint.setObjectName("mapBarHint")
        bl.addWidget(hint)
        vl.addWidget(bar)

        self.map_view = QWebEngineView()
        self.map_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self.map_view, stretch=1)

        return container

    # ------------------------------------------------------------------
    # Default map (no routes)
    # ------------------------------------------------------------------

    def _show_default_map(self) -> None:
        fmap = folium.Map(
            location=[-37.858, 145.070],
            zoom_start=14,
            tiles="OpenStreetMap",
        )
        for nid, (lat, lon, lbl) in REAL_SCATS.items():
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(
                    f"<b>SCATS {nid}</b><br>{lbl}", max_width=220
                ),
                tooltip=f"SCATS {nid}",
                icon=folium.Icon(color="cadetblue", icon="map-marker", prefix="fa"),
            ).add_to(fmap)

        intro_html = """
        <div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
                    z-index:9999;background:rgba(255,255,255,0.9);
                    padding:18px 28px;border-radius:12px;text-align:center;
                    box-shadow:0 4px 16px rgba(0,0,0,0.15);
                    font-family:'Segoe UI',Arial,sans-serif;">
          <div style="font-size:28px">🚦</div>
          <div style="font-size:16px;font-weight:bold;margin-top:6px">
            TBRGS Route Map
          </div>
          <div style="font-size:13px;color:#666;margin-top:4px">
            Select origin &amp; destination, then click<br>
            <b>Find Top 5 Routes</b> to visualise routes.
          </div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(intro_html))
        fmap.save(MAP_OUTPUT)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.realpath(MAP_OUTPUT)))

    # ------------------------------------------------------------------
    # Slot: Find Routes button
    # ------------------------------------------------------------------

    def _on_find_clicked(self) -> None:
        try:
            origin = int(self.origin_input.text().strip())
            dest   = int(self.dest_input.text().strip())
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input",
                "Origin and Destination must be integer SCATS numbers.\n"
                f"Available: {sorted(REAL_SCATS.keys())}"
            )
            return

        if origin == dest:
            QMessageBox.warning(
                self, "Invalid Input",
                "Origin and Destination must be different sites."
            )
            return

        model = self.model_combo.currentText().lower()

        self.find_btn.setEnabled(False)
        self.results_text.clear()
        self._set_status("⏳  Computing routes …", "#f39c12")

        self._worker = RoutingWorker(origin, dest, model)
        self._worker.finished.connect(self._on_routes_ready)
        self._worker.error.connect(self._on_routing_error)
        self._worker.start()

    # ------------------------------------------------------------------
    # Slot: Routes computed
    # ------------------------------------------------------------------

    def _on_routes_ready(self, routes: list) -> None:
        self.find_btn.setEnabled(True)

        origin = int(self.origin_input.text().strip())
        dest   = int(self.dest_input.text().strip())
        model  = self.model_combo.currentText()

        origin_lbl = REAL_SCATS.get(origin, ("", "", str(origin)))[2]
        dest_lbl   = REAL_SCATS.get(dest,   ("", "", str(dest)))[2]

        # ---- Text results ------------------------------------------------
        COLOR_DOTS = {
            "red":    "🔴", "blue":   "🔵", "green":  "🟢",
            "purple": "🟣", "orange": "🟠",
        }
        lines = [
            f"  Origin      : SCATS {origin}",
            f"                {origin_lbl}",
            f"  Destination : SCATS {dest}",
            f"                {dest_lbl}",
            f"  Model       : {model}",
            f"  Routes found: {len(routes)}",
            "",
            "  ─────────────────────────────────────",
        ]
        for i, (path, tt) in enumerate(routes):
            col  = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            dot  = COLOR_DOTS.get(col, "●")
            hops = len(path) - 1
            lines += [
                f"",
                f"  {dot}  Route {i + 1}  [{col.upper()}]",
                f"     Path  : {' → '.join(map(str, path))}",
                f"     Hops  : {hops}",
                f"     Time  : {tt:.2f} min",
            ]
        self.results_text.setPlainText("\n".join(lines))

        # ---- Folium map --------------------------------------------------
        self._set_status("🗺  Rendering map …", "#3498db")
        try:
            path_html = generate_map(routes, origin, dest)
            self.map_view.setUrl(QUrl.fromLocalFile(path_html))
            self._set_status(
                f"✅  {len(routes)} route(s) found — map updated.",
                "#27ae60",
            )
        except Exception as exc:
            self._set_status(f"Map render error: {exc}", "#e74c3c")

    # ------------------------------------------------------------------
    # Slot: Routing error
    # ------------------------------------------------------------------

    def _on_routing_error(self, msg: str) -> None:
        self.find_btn.setEnabled(True)
        self._set_status("❌  Routing failed.", "#e74c3c")
        QMessageBox.critical(self, "Routing Error", msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, color: str = "#7f8c8d") -> None:
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")


# ===========================================================================
# Widget helpers
# ===========================================================================

def _make_hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setObjectName("hLine")
    return line


def _form_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("formLabel")
    return lbl


# ===========================================================================
# Stylesheet
# ===========================================================================

_STYLESHEET = """
/* ── Global ─────────────────────────────────────────── */
QMainWindow, QWidget      { background: #f0f2f5; font-family: 'Segoe UI', Arial, sans-serif; }
QScrollBar:vertical       { width: 6px; background: transparent; }
QScrollBar::handle:vertical { background: #4a6080; border-radius: 3px; min-height: 20px; }

/* ── Left panel ──────────────────────────────────────── */
#leftPanel                { background: #1c2b3a; }
#appTitle                 { font-size: 28px; font-weight: bold; color: #3498db; margin: 0; }
#appSubtitle              { font-size: 11px; color: #7f8c8d; margin-bottom: 2px; }
#hLine                    { background: #2c3e50; border: none; max-height: 1px; }
#sectionLabel             { font-size: 10px; font-weight: bold; color: #7f8c8d;
                             letter-spacing: 1.5px; }
#scatsScroll              { background: transparent; border: none; }
#scatsInner               { background: transparent; }
#scatsRow                 { font-size: 11px; padding: 1px 0; color: #bdc3c7; }

/* ── Config group ────────────────────────────────────── */
#configGroup              { border: 1px solid #2c3e50; border-radius: 8px;
                             margin-top: 6px; padding-top: 8px; color: #ecf0f1;
                             font-size: 13px; font-weight: bold; }
#configGroup::title       { subcontrol-origin: margin; left: 10px;
                             color: #3498db; padding: 0 4px; }
#formLabel                { font-size: 12px; color: #bdc3c7; margin-top: 4px; }
#formInput, QLineEdit     { background: #243447; color: #ecf0f1;
                             border: 1px solid #2c3e50; border-radius: 5px;
                             padding: 7px 10px; font-size: 13px; }
#formInput:focus          { border: 1px solid #3498db; }
#modelCombo, QComboBox    { background: #243447; color: #ecf0f1;
                             border: 1px solid #2c3e50; border-radius: 5px;
                             padding: 7px 10px; font-size: 13px; }
QComboBox::drop-down      { border: none; width: 20px; }
QComboBox::down-arrow     { image: none; }
QComboBox QAbstractItemView { background: #243447; color: #ecf0f1;
                               selection-background-color: #3498db; border: none; }

/* ── Find button ─────────────────────────────────────── */
#findBtn                  { background: #3498db; color: white; font-size: 14px;
                             font-weight: bold; border: none; border-radius: 8px; }
#findBtn:hover            { background: #2980b9; }
#findBtn:pressed          { background: #1a6fa1; }
#findBtn:disabled         { background: #2c3e50; color: #4a6080; }

/* ── Results text ────────────────────────────────────── */
#resultsText              { background: #172535; color: #ecf0f1; border: none;
                             border-radius: 6px; font-family: 'Courier New', monospace;
                             font-size: 12px; padding: 8px; }
#statusLabel              { font-size: 11px; color: #7f8c8d; }

/* ── Map panel ───────────────────────────────────────── */
#mapContainer             { background: #f0f2f5; }
#mapBar                   { background: #2c3e50; }
#mapBarLabel              { color: #ecf0f1; font-size: 13px; font-weight: bold; }
#mapBarHint               { color: #7f8c8d; font-size: 11px; font-style: italic; }

/* ── Message boxes ───────────────────────────────────── */
QMessageBox               { background: #1c2b3a; color: #ecf0f1; }
QMessageBox QPushButton   { background: #3498db; color: white; border-radius: 4px;
                             padding: 6px 16px; font-weight: bold; }
QMessageBox QPushButton:hover { background: #2980b9; }
"""


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = TBRGSWindow()
    window.show()

    sys.exit(app.exec_())
