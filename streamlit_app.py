"""
Analog Modulation Lab
=====================
Interactive Streamlit demonstrator for analog amplitude-modulation families:

    * Conventional AM (DSB-LC)  - envelope / rectifier / coherent detection
    * DSB-SC                    - coherent detection
    * SSB (USB / LSB)           - Hilbert-transform (phasing) method + coherent detection
    * QAM                       - two independent messages on quadrature carriers

Run:
    pip install streamlit numpy scipy plotly
    streamlit run am_modulation_lab.py
"""

import base64
import html
import inspect

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal

# ----------------------------------------------------------------------------
# Simulation grid
#   200 ms record  ->  5 Hz FFT bin spacing (very sharp spectral lines)
# ----------------------------------------------------------------------------
FS = 400_000                     # sample rate [Hz]
T_SIM = 0.2                      # simulated duration [s]
N = int(FS * T_SIM)
t = np.arange(N) / FS
ZPAD = 4                         # FFT zero-padding factor (interpolates peaks smoothly)

C_MSG, C_MSG2 = "#2563eb", "#7c3aed"
C_TX, C_ENV = "#dc2626", "#f59e0b"
C_RX, C_OUT = "#64748b", "#059669"

SCHEMES = ["Conventional AM", "DSB-SC", "SSB", "QAM"]


# ----------------------------------------------------------------------------
# Streamlit version helpers (use_container_width -> width="stretch")
# ----------------------------------------------------------------------------
def _stretch(fn):
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def show(fig):
    st.plotly_chart(fig, **_stretch(st.plotly_chart))


def show_table(data):
    st.dataframe(data, hide_index=True, **_stretch(st.dataframe))


def show_svg(svg):
    b64 = base64.b64encode(svg.encode("utf-8")).decode()
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" '
        'style="width:100%;max-width:920px;display:block;margin:0 auto 0.5rem auto;"/>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------------
def snap(f):
    """Snap a frequency to the FFT grid so every message is exactly periodic
    in the simulation window (leakage-free spectra, clean Hilbert transform)."""
    return max(1, round(f * T_SIM)) / T_SIM


def make_message(kind, f, fc):
    """Return (message normalised to unit peak, message bandwidth W in Hz)."""
    f = snap(f)
    if kind == "Sine":
        x, W = np.sin(2 * np.pi * f * t), f
    elif kind == "Two-tone":
        f2 = snap(2.5 * f)
        x = 0.6 * np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f2 * t)
        W = f2
    else:  # band-limited Square / Triangle
        raw = (signal.square(2 * np.pi * f * t) if kind == "Square"
               else signal.sawtooth(2 * np.pi * f * t, width=0.5))
        W = min(5 * f, fc / 3)
        sos = signal.butter(6, W, fs=FS, output="sos")
        x = signal.sosfiltfilt(sos, raw)
    return x / np.max(np.abs(x)), W


def hilbert_tf(x):
    """Hilbert transform (quadrature version of x)."""
    return np.imag(signal.hilbert(x))


def modulate(scheme, m1, m2, p):
    """Return (s(t), ideal envelope)."""
    Ac, mu, fc = p["Ac"], p["mu"], p["fc"]
    c, s_ = np.cos(2 * np.pi * fc * t), np.sin(2 * np.pi * fc * t)
    if scheme == "Conventional AM":
        env = Ac * (1 + mu * m1)
        return env * c, env
    if scheme == "DSB-SC":
        return Ac * m1 * c, Ac * m1
    if scheme == "SSB":
        mh = hilbert_tf(m1)
        sign = 1 if p["sideband"] == "USB" else -1
        s = 0.5 * Ac * (m1 * c - sign * mh * s_)
        return s, 0.5 * Ac * np.hypot(m1, mh)
    # QAM
    return Ac * (m1 * c + m2 * s_), Ac * np.hypot(m1, m2)


def add_noise(s, snr_db):
    rng = np.random.default_rng(7)
    p_n = np.mean(s ** 2) / 10 ** (snr_db / 10)
    return s + rng.normal(0, np.sqrt(p_n), s.size)


# ----------------------------------------------------------------------------
# Demodulation
# ----------------------------------------------------------------------------
def lpf(x, fcut):
    sos = signal.butter(6, fcut, fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x)


def detector_stage(scheme, r, p):
    """Detector output(s) BEFORE the low-pass filter (list of 1 or 2 arrays).
    Envelope / rectifier: |.| of the input.  Coherent: mixer product 2*r*LO."""
    if scheme == "Conventional AM":
        det = p["detector"]
        if det.startswith("Envelope"):
            return [np.abs(signal.hilbert(r))]
        if det.startswith("Rectifier"):
            return [np.abs(r) * np.pi / 2]
    th = 2 * np.pi * (p["fc"] + p["df"]) * t + np.radians(p["phi"])
    if scheme == "QAM":
        return [2 * r * np.cos(th), 2 * r * np.sin(th)]
    return [2 * r * np.cos(th)]


def am_detect(r, p):
    """Raw AM detector output AFTER the LPF but BEFORE the DC-blocking stage (volts).
    Envelope / rectifier detectors give Ac*|1 + mu*m(t)| (never negative);
    the coherent detector gives the signed Ac*(1 + mu*m(t))."""
    return lpf(detector_stage("Conventional AM", r, p)[0], p["cutoff"])


def demodulate(scheme, r, p):
    """Return calibrated recovered message(s): (y1, y2 or None)."""
    Ac, mu, cut = p["Ac"], p["mu"], p["cutoff"]
    stage = detector_stage(scheme, r, p)
    if scheme == "Conventional AM":
        raw = lpf(stage[0], cut)
        return (raw - np.mean(raw)) / (Ac * mu), None      # DC block + scale
    if scheme == "DSB-SC":
        return lpf(stage[0], cut) / Ac, None
    if scheme == "SSB":
        return 2 * lpf(stage[0], cut) / Ac, None
    return lpf(stage[0], cut) / Ac, lpf(stage[1], cut) / Ac


def pipeline(scheme, m1, m2, p):
    s, env = modulate(scheme, m1, m2, p)
    r = add_noise(s, p["snr"]) if p["noise"] else s
    y1, y2 = demodulate(scheme, r, p)
    return s, env, r, y1, y2


def fidelity(y, m):
    """SNR (dB) of the recovered message relative to the original."""
    k = int(0.05 * N)
    e = y[k:-k] - m[k:-k]
    return float(np.clip(10 * np.log10(np.sum(m[k:-k] ** 2) / (np.sum(e ** 2) + 1e-12)), -20, 60))


def _messages(scheme, kind, fm, fc):
    m1, _ = make_message(kind, fm, fc)
    m2 = make_message(kind, 1.5 * fm, fc)[0] if scheme == "QAM" else np.zeros(N)
    return m1, m2


@st.cache_data(show_spinner=False)
def sweep_phase(scheme, kind, fm, p):
    m1, m2 = _messages(scheme, kind, fm, p["fc"])
    phis = np.arange(-180, 181, 10)
    fids = [fidelity(pipeline(scheme, m1, m2, {**p, "phi": float(ph)})[3], m1) for ph in phis]
    return phis, fids


@st.cache_data(show_spinner=False)
def sweep_mu(scheme, kind, fm, p):
    m1, m2 = _messages(scheme, kind, fm, p["fc"])
    mus = np.arange(0.1, 2.01, 0.1)
    fids = [fidelity(pipeline(scheme, m1, m2, {**p, "mu": float(x)})[3], m1) for x in mus]
    return mus, fids


# ----------------------------------------------------------------------------
# Spectrum helpers
# ----------------------------------------------------------------------------
WINDOWS = {"Hann": signal.windows.hann,
           "Blackman-Harris": signal.windows.blackmanharris,
           "Rectangular": signal.windows.boxcar}


def spectrum(x, window="Hann"):
    """Single-sided amplitude spectrum, zero-padded for smooth, sharp peaks."""
    w = WINDOWS[window](x.size)
    nfft = ZPAD * x.size
    X = np.fft.rfft(x * w, n=nfft)
    return np.fft.rfftfreq(nfft, 1 / FS), 2 * np.abs(X) / w.sum()


def spec_view(x, lo, hi, window, yscale):
    """Spectrum cropped to [lo, hi] Hz. Returns (kHz axis, dB or linear amplitude)."""
    f, a = spectrum(x, window)
    m = (f >= lo) & (f <= hi)
    y = 20 * np.log10(np.maximum(a[m], 1e-6)) if yscale == "dB" else a[m]
    return f[m] / 1e3, y


def band_power(x, f_lo, f_hi):
    X = np.fft.rfft(x * signal.windows.hann(x.size))
    f = np.fft.rfftfreq(x.size, 1 / FS)
    return float(np.sum(np.abs(X[(f >= f_lo) & (f < f_hi)]) ** 2)) + 1e-18


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def style(fig, height, **kw):
    layout = dict(
        template="plotly_white", height=height,
        margin=dict(l=55, r=20, t=45, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified")
    layout.update(kw)                      # caller overrides win (no duplicate kwargs)
    fig.update_layout(**layout)
    return fig


def legend_key(row):
    return "legend" if row == 1 else f"legend{row}"


def add(fig, trace, row):
    """Add a trace to a subplot row and attach it to that row's own legend."""
    try:
        trace.legend = legend_key(row)
    except (ValueError, AttributeError):        # Plotly < 5.15: fall back to one legend
        pass
    fig.add_trace(trace, row=row, col=1)


def side_legends(fig, rows):
    """Place one legend per subplot in the right margin (never over a plot)."""
    try:
        upd = {}
        for r in range(1, rows + 1):
            dom = fig.layout["yaxis" if r == 1 else f"yaxis{r}"].domain
            upd[legend_key(r)] = dict(orientation="v", x=1.01, y=dom[1], xanchor="left",
                                      yanchor="top", bgcolor="rgba(0,0,0,0)")
        fig.update_layout(**upd)
    except Exception:
        pass


def multi_style(fig, rows, row_h=200, extra=90, **kw):
    style(fig, row_h * rows + extra, margin=dict(l=55, r=235, t=45, b=40), **kw)
    side_legends(fig, rows)
    return fig


def shade_overmod(fig, mask, tm, rows, label_row):
    for j, (x0, x1) in enumerate(segments(mask, tm)):
        for rr in rows:
            kw = (dict(annotation_text="1 + μm < 0", annotation_position="top left")
                  if (j == 0 and rr == label_row) else {})
            fig.add_vrect(x0=x0, x1=x1, fillcolor="#ef4444", opacity=0.10, line_width=0,
                          row=rr, col=1, **kw)


def segments(mask, x):
    """(x_start, x_end) of every contiguous True run in a boolean mask."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return [(x[a], x[b - 1]) for a, b in zip(idx[::2], idx[1::2])]


def shade_bands(fig, row, scheme, fc, W, sb):
    lo, hi = (fc - W, fc), (fc, fc + W)
    if scheme == "SSB":
        keep, supp = (hi, lo) if sb == "USB" else (lo, hi)
        fig.add_vrect(x0=keep[0] / 1e3, x1=keep[1] / 1e3, fillcolor=C_OUT, opacity=0.13,
                      line_width=0, row=row, col=1)
        fig.add_vrect(x0=supp[0] / 1e3, x1=supp[1] / 1e3, fillcolor="#94a3b8", opacity=0.10,
                      line_width=0, row=row, col=1)
    else:
        fig.add_vrect(x0=lo[0] / 1e3, x1=hi[1] / 1e3, fillcolor=C_MSG, opacity=0.07,
                      line_width=0, row=row, col=1)
    fig.add_vline(x=fc / 1e3, line_dash="dot", line_color="#475569", row=row, col=1)


# ----------------------------------------------------------------------------
# Block-diagram engine (pure SVG, theme independent)
# ----------------------------------------------------------------------------
KINDS = {                      # fill, stroke
    "proc": ("#dbeafe", "#2563eb"),   # gain / generic processing
    "osc": ("#fef3c7", "#d97706"),    # oscillators
    "filt": ("#dcfce7", "#16a34a"),   # filters
    "nl": ("#f3e8ff", "#7c3aed"),     # non-linear / Hilbert
    "sync": ("#e2e8f0", "#475569"),   # synchronisation
}
DW = 850                       # diagram width


class Diagram:
    def __init__(self, h):
        self.w, self.h, self.e = DW, h, []

    def _t(self, x, y, txt, size=13, weight="400", anchor="middle", style="normal", color="#0f172a"):
        self.e.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
                      f'text-anchor="{anchor}" font-style="{style}" fill="{color}">{html.escape(txt)}</text>')

    def label(self, x, y, txt, anchor="middle"):
        self._t(x, y, txt, 16, "700", anchor, "italic")

    def note(self, x, y, txt, anchor="start", size=12, weight="700", color="#334155"):
        self._t(x, y, txt, size, weight, anchor, "normal", color)

    def box(self, cx, cy, w, h, title, sub="", kind="proc"):
        fill, stroke = KINDS[kind]
        self.e.append(f'<rect x="{cx - w / 2:g}" y="{cy - h / 2:g}" width="{w}" height="{h}" rx="8" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
        if sub:
            self._t(cx, cy - 4, title, 13, "700")
            self._t(cx, cy + 14, sub, 11, "400", color="#475569")
        else:
            self._t(cx, cy + 5, title, 13, "700")
        return dict(l=(cx - w / 2, cy), r=(cx + w / 2, cy), t=(cx, cy - h / 2), b=(cx, cy + h / 2))

    def _circle(self, cx, cy, r=16):
        self.e.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="#0f172a" stroke-width="1.8"/>')
        return dict(l=(cx - r, cy), r=(cx + r, cy), t=(cx, cy - r), b=(cx, cy + r))

    def mult(self, cx, cy):
        a = self._circle(cx, cy)
        d = 9
        self.e.append(f'<path d="M{cx - d},{cy - d} L{cx + d},{cy + d} M{cx - d},{cy + d} L{cx + d},{cy - d}" '
                      'stroke="#0f172a" stroke-width="1.8"/>')
        return a

    def add(self, cx, cy):
        a = self._circle(cx, cy)
        d = 9
        self.e.append(f'<path d="M{cx - d},{cy} L{cx + d},{cy} M{cx},{cy - d} L{cx},{cy + d}" '
                      'stroke="#0f172a" stroke-width="1.8"/>')
        return a

    def dot(self, x, y):
        self.e.append(f'<circle cx="{x}" cy="{y}" r="3.6" fill="#334155"/>')

    def line(self, *pts):
        p = " ".join(f"{x:g},{y:g}" for x, y in pts)
        self.e.append(f'<polyline points="{p}" fill="none" stroke="#334155" stroke-width="1.7"/>')

    def arrow(self, *pts, label=None, lpos=None):
        p = " ".join(f"{x:g},{y:g}" for x, y in pts)
        self.e.append(f'<polyline points="{p}" fill="none" stroke="#334155" stroke-width="1.7" '
                      'marker-end="url(#ah)"/>')
        if label:
            if lpos is None:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                lpos = ((x0 + x1) / 2, (y0 + y1) / 2 - 8)
            self._t(lpos[0], lpos[1], label, 11, "400", "middle", "italic", "#334155")

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
                '<defs><marker id="ah" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
                '<path d="M0,0 L10,4 L0,8 z" fill="#334155"/></marker></defs>'
                f'<rect x="1" y="1" width="{self.w - 2}" height="{self.h - 2}" rx="14" fill="#ffffff" '
                'stroke="#cbd5e1"/>' + "".join(self.e) + "</svg>")


def _osc(fc, amp=None, fn="cos", phi=0, df=0):
    f = (fc + df) / 1e3
    a = f"{amp:g}·" if amp else ""
    ph = f"{phi:+d}°" if phi else ""
    return f"{a}{fn}(2π·{f:g}k·t{ph})"


# ---- modulators -------------------------------------------------------------
def _mod_am(p):
    d = Diagram(270)
    d.label(32, 116, "m(t)")
    g = d.box(190, 110, 110, 50, "Gain", f"μ = {p['mu']:.2f}", "proc")
    d.arrow((54, 110), g["l"])
    s = d.add(340, 110)
    d.arrow(g["r"], s["l"], label="μ·m(t)")
    dc = d.box(340, 205, 90, 40, "DC bias", "1", "proc")
    d.arrow(dc["t"], s["b"])
    x = d.mult(540, 110)
    d.arrow(s["r"], x["l"], label="1 + μ·m(t)")
    osc = d.box(540, 205, 190, 50, "Carrier oscillator", _osc(p["fc"], p["Ac"]), "osc")
    d.arrow(osc["t"], x["b"])
    d.arrow(x["r"], (690, 110))
    d.label(705, 116, "s(t)", "start")
    return d.svg()


def _mod_dsb(p):
    d = Diagram(260)
    d.label(32, 96, "m(t)")
    lp = d.box(180, 90, 120, 50, "Message LPF", "limits BW to W", "filt")
    d.arrow((54, 90), lp["l"])
    x = d.mult(400, 90)
    d.arrow(lp["r"], x["l"])
    osc = d.box(400, 190, 190, 50, "Carrier oscillator", _osc(p["fc"], p["Ac"]), "osc")
    d.arrow(osc["t"], x["b"])
    bp = d.box(600, 90, 130, 50, "Band-pass", "centred at fc", "filt")
    d.arrow(x["r"], bp["l"], label="m·cos ωct")
    d.arrow(bp["r"], (750, 90))
    d.label(765, 96, "s(t)", "start")
    return d.svg()


def _quadrature_core(d, x, sc_sub, cos_sub, top_y, mid_y, sh_y, bot_y):
    """Oscillator + 90° shifter stack feeding two multipliers (returns mult anchors)."""
    x1 = d.mult(x, top_y)
    osc = d.box(x, mid_y, 170, 46, "Carrier oscillator", cos_sub, "osc")
    sh = d.box(x, sh_y, 110, 34, "−90° shift", "", "sync")
    x2 = d.mult(x, bot_y)
    d.arrow(osc["t"], x1["b"])
    d.arrow(osc["b"], sh["t"])
    d.arrow(sh["b"], x2["t"])
    d.note(x + 14, sh_y + 30, sc_sub, size=11, weight="400", color="#475569")
    return x1, x2, osc


def _mod_ssb(p):
    d = Diagram(285)
    usb = p["sideband"] == "USB"
    d.label(32, 151, "m(t)")
    d.line((54, 145), (80, 145))
    d.dot(80, 145)
    x1, x2, _ = _quadrature_core(d, 330, "sin(2π·fc·t)", _osc(p["fc"], p["Ac"]), 45, 110, 170, 240)
    d.line((80, 145), (80, 45))
    d.arrow((80, 45), x1["l"], label="m(t)", lpos=(200, 37))
    d.line((80, 145), (80, 240))
    hb = d.box(190, 240, 110, 46, "Hilbert", "−90°, all freq.", "nl")
    d.arrow((80, 240), hb["l"])
    d.arrow(hb["r"], x2["l"], label="m̂(t)", lpos=(280, 232))
    sm = d.add(570, 142)
    d.arrow(x1["r"], (570, 45), sm["t"], label="m(t)·cos ωct", lpos=(458, 37))
    d.arrow(x2["r"], (570, 240), sm["b"], label="m̂(t)·sin ωct", lpos=(458, 232))
    d.note(584, 116, "+")
    d.note(584, 184, f"{'−' if usb else '+'}  ({p['sideband']})")
    g = d.box(705, 142, 90, 46, "Gain", "1/2", "proc")
    d.arrow(sm["r"], g["l"])
    d.arrow(g["r"], (800, 142))
    d.label(810, 148, "s(t)", "start")
    return d.svg()


def _mod_qam(p):
    d = Diagram(285)
    x1, x2, _ = _quadrature_core(d, 360, "sin(2π·fc·t)", _osc(p["fc"], p["Ac"]), 45, 110, 170, 240)
    d.label(32, 51, "m₁(t)")
    lp1 = d.box(175, 45, 100, 44, "LPF", "W", "filt")
    d.arrow((58, 45), lp1["l"])
    d.arrow(lp1["r"], x1["l"])
    d.label(32, 246, "m₂(t)")
    lp2 = d.box(175, 240, 100, 44, "LPF", "W", "filt")
    d.arrow((58, 240), lp2["l"])
    d.arrow(lp2["r"], x2["l"])
    sm = d.add(600, 142)
    d.arrow(x1["r"], (600, 45), sm["t"], label="m₁·cos ωct", lpos=(480, 37))
    d.arrow(x2["r"], (600, 240), sm["b"], label="m₂·sin ωct", lpos=(480, 232))
    d.arrow(sm["r"], (730, 142))
    d.label(745, 148, "s(t)", "start")
    return d.svg()


# ---- demodulators -----------------------------------------------------------
def _plain_chain(blocks):
    n = len(blocks)
    step = (DW - 170) / n
    d = Diagram(170)
    d.label(32, 91, "r(t)")
    prev = (54, 85)
    for i, (ti, su, k) in enumerate(blocks):
        b = d.box(75 + step * (i + 0.5), 85, 124, 56, ti, su, k)
        d.arrow(prev, b["l"])
        prev = b["r"]
    d.arrow(prev, (prev[0] + 40, 85))
    d.label(prev[0] + 50, 91, "y(t)", "start")
    return d.svg()


def _coherent_single(lo_sub, rec, post):
    d = Diagram(240)
    d.label(32, 101, "r(t)")
    x = d.mult(305, 95)
    d.arrow((54, 95), x["l"])
    d.dot(100, 95)
    pll = d.box(100, 195, 150, 50, rec[0], rec[1], "sync")
    d.arrow((100, 95), pll["t"])
    lo = d.box(305, 195, 175, 50, "Local oscillator", lo_sub, "osc")
    d.arrow(pll["r"], lo["l"])
    d.arrow(lo["t"], x["b"])
    prev = x["r"]
    for i, (ti, su, k) in enumerate(post):
        b = d.box(440 + 130 * i, 95, 100, 52, ti, su, k)
        d.arrow(prev, b["l"])
        prev = b["r"]
    d.arrow(prev, (prev[0] + 38, 95))
    d.label(prev[0] + 48, 101, "y(t)", "start")
    return d.svg()


def _demod_qam(p):
    d = Diagram(285)
    lo_sub = _osc(p["fc"], None, "cos", int(p["phi"]), p["df"])
    d.label(32, 151, "r(t)")
    d.line((54, 145), (85, 145))
    d.dot(85, 145)
    x1, x2, osc = _quadrature_core(d, 360, "sin(2π·fc·t + φ)", lo_sub, 45, 110, 170, 240)
    d.line((85, 145), (85, 45))
    d.arrow((85, 45), x1["l"])
    d.line((85, 145), (85, 240))
    d.arrow((85, 240), x2["l"])
    d.dot(85, 110)
    pll = d.box(185, 110, 100, 50, "Carrier", "recovery", "sync")
    d.arrow((85, 110), pll["l"])
    d.arrow(pll["r"], osc["l"])
    cut = f"{p['cutoff']:.0f} Hz"
    la = d.box(520, 45, 100, 44, "LPF", cut, "filt")
    d.arrow(x1["r"], la["l"])
    d.arrow(la["r"], (640, 45))
    d.label(654, 51, "y₁(t) ≈ m₁", "start")
    lb = d.box(520, 240, 100, 44, "LPF", cut, "filt")
    d.arrow(x2["r"], lb["l"])
    d.arrow(lb["r"], (640, 240))
    d.label(654, 246, "y₂(t) ≈ m₂", "start")
    return d.svg()


def am_demod_svgs(p):
    cut = f"{p['cutoff']:.0f} Hz"
    scale = "1/(Ac·μ)"
    return {
        "Envelope detector (ideal)": _plain_chain([
            ("Envelope", "|x + j·x̂|", "nl"), ("LPF", cut, "filt"),
            ("DC block", "removes carrier", "proc"), ("Scale", scale, "proc")]),
        "Rectifier + LPF (diode detector)": _plain_chain([
            ("Rectifier", "|r(t)|", "nl"), ("LPF", cut, "filt"),
            ("Gain", "π/(2·Ac·μ)", "proc"), ("DC block", "removes carrier", "proc")]),
        "Coherent (synchronous)": _coherent_single(
            _osc(p["fc"], None, "cos", int(p["phi"]), p["df"]),
            ("Carrier recovery", "locks to carrier"),
            [("LPF", cut, "filt"), ("DC block", "removes carrier", "proc"), ("Gain", scale, "proc")]),
    }


def block_modulator(scheme, p):
    return {"Conventional AM": _mod_am, "DSB-SC": _mod_dsb, "SSB": _mod_ssb, "QAM": _mod_qam}[scheme](p)


def block_demodulator(scheme, p):
    cut = f"{p['cutoff']:.0f} Hz"
    lo = _osc(p["fc"], None, "cos", int(p["phi"]), p["df"])
    if scheme == "Conventional AM":
        return am_demod_svgs(p)[p["detector"]]
    if scheme == "DSB-SC":
        return _coherent_single(lo, ("Costas loop", "recovers fc, φ"),
                                [("LPF", cut, "filt"), ("Gain", "1/Ac", "proc")])
    if scheme == "SSB":
        return _coherent_single(lo, ("BFO / pilot", "tuned to fc"),
                                [("LPF", cut, "filt"), ("Gain", "4/Ac", "proc")])
    return _demod_qam(p)


DEMOD_EQ = {
    "Envelope detector (ideal)": r"y(t)=\mathrm{LPF}\{|r(t)+j\hat r(t)|\}-\text{DC}\;\propto\; m(t)",
    "Rectifier + LPF (diode detector)": r"y(t)=\tfrac{\pi}{2}\,\mathrm{LPF}\{|r(t)|\}-\text{DC}\;\propto\; m(t)\quad(\mu\le 1)",
    "Coherent (synchronous)": r"y(t)=\mathrm{LPF}\{2\,r(t)\cos(2\pi f_c t+\varphi)\}-\text{DC}\;\propto\; m(t)",
    "DSB-SC": r"y(t)=\mathrm{LPF}\{2\,s(t)\cos(2\pi f_c t+\varphi)\}/A_c=m(t)\cos\varphi",
    "SSB": r"y(t)=\mathrm{LPF}\{4\,s(t)\cos(2\pi f_c t+\varphi)\}/A_c=m(t)\cos\varphi\pm\hat m(t)\sin\varphi",
    "QAM": r"y_1=m_1\cos\varphi-m_2\sin\varphi,\qquad y_2=m_2\cos\varphi+m_1\sin\varphi",
}


# ----------------------------------------------------------------------------
# Static theory text
# ----------------------------------------------------------------------------
THEORY = {
    "Conventional AM": dict(
        eq=[r"s(t)=A_c\,[1+\mu\,m(t)]\cos(2\pi f_c t),\quad |m(t)|\le 1",
            r"S(f)=\tfrac{A_c}{2}\delta(f\mp f_c)+\tfrac{A_c\mu}{4}\left[M(f- f_c)+M(f+f_c)\right]",
            r"B_T=2W,\qquad \eta=\frac{\mu^2\langle m^2\rangle}{1+\mu^2\langle m^2\rangle}\;\le\;33\%\ \text{(for a sine, }\mu=1)"],
        txt="The carrier is transmitted along with the sidebands, so the message is the *envelope* "
            "and a very simple diode/RC envelope detector recovers it. The price is power efficiency: "
            "most of the power sits in the carrier, which carries no information. For μ > 1 the "
            "envelope crosses zero (over-modulation) and an envelope detector distorts the output; "
            "a coherent detector still works."),
    "DSB-SC": dict(
        eq=[r"s(t)=A_c\,m(t)\cos(2\pi f_c t)",
            r"S(f)=\tfrac{A_c}{2}\left[M(f-f_c)+M(f+f_c)\right]",
            r"y(t)=\mathrm{LPF}\{2\,s(t)\cos(2\pi f_c t+\varphi)\}=A_c\,m(t)\cos\varphi"],
        txt="The carrier is suppressed, so 100 % of the power carries information. The envelope is "
            "|m(t)| and the carrier phase flips by 180° at every zero crossing, therefore an envelope "
            "detector cannot be used: a synchronous (coherent) detector with a phase-locked local "
            "oscillator is required. A phase error φ scales the output by cos φ (a null at 90°)."),
    "SSB": dict(
        eq=[r"s(t)=\tfrac{A_c}{2}\left[m(t)\cos(2\pi f_c t)\mp\hat m(t)\sin(2\pi f_c t)\right]\quad(\text{upper}/\text{lower})",
            r"\hat m(t)=\mathcal{H}\{m(t)\},\qquad B_T=W",
            r"y(t)=\mathrm{LPF}\{4\,s(t)\cos(2\pi f_c t+\varphi)/A_c\}=m(t)\cos\varphi\pm\hat m(t)\sin\varphi"],
        txt="Only one sideband is transmitted, halving the bandwidth relative to DSB. The phasing "
            "method used here forms the quadrature term with a Hilbert transformer. Phase error "
            "rotates the recovered spectrum (phase distortion), while a *frequency* error shifts every "
            "component by Δf, the familiar 'Donald-Duck' effect on speech."),
    "QAM": dict(
        eq=[r"s(t)=A_c\left[m_1(t)\cos(2\pi f_c t)+m_2(t)\sin(2\pi f_c t)\right]",
            r"y_I=m_1\cos\varphi-m_2\sin\varphi,\qquad y_Q=m_2\cos\varphi+m_1\sin\varphi",
            r"B_T=2W\ \text{(shared by two messages)}"],
        txt="Two independent messages share the same spectrum by riding on carriers 90° apart, giving "
            "the spectral efficiency of SSB with the simplicity of DSB. The price is strict phase "
            "synchronisation: any phase error φ leaks part of one channel into the other "
            "(co-channel / crosstalk interference)."),
}

COMPARE_TABLE = {
    "Property": ["Transmitted bandwidth", "Carrier transmitted", "Power efficiency",
                 "Receiver", "Sync. required", "Typical use"],
    "Conventional AM": ["2W", "Yes", "Low (≤ 33 % at μ=1, sine)", "Envelope detector (very simple)",
                        "No", "Broadcast MW/SW radio"],
    "DSB-SC": ["2W", "No", "100 %", "Coherent + carrier recovery", "Phase & frequency",
               "Subcarriers, stereo FM difference signal"],
    "SSB": ["W", "No", "100 %", "Coherent + BFO", "Phase & frequency (tolerant to phase)",
            "HF voice, amateur radio"],
    "QAM": ["2W (for 2 messages)", "No", "100 %", "Two coherent branches", "Phase & frequency (strict)",
            "NTSC/PAL colour, digital QAM basis"],
}


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Analog Modulation Lab", page_icon="📡", layout="wide")
    st.markdown("""
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      .hero {background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 100%); color:#fff;
             padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}
      .hero h1 {margin:0; font-size: 1.7rem; color:#fff;}
      .hero p {margin:.25rem 0 0 0; color:#cbd5e1; font-size:.95rem;}
      div[data-testid="stMetric"] {background: rgba(148,163,184,.12); border:1px solid rgba(148,163,184,.25);
             padding: .7rem .9rem; border-radius: 12px;}
    </style>
    <div class="hero"><h1>📡 Analog Modulation Lab</h1>
    <p>Conventional AM · DSB-SC · SSB · QAM — modulation, spectra, block diagrams and coherent / non-coherent demodulation</p></div>
    """, unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    sb = st.sidebar
    sb.header("Controls")
    scheme = sb.radio("Modulation scheme", SCHEMES)

    sb.subheader("Carrier")
    Ac = sb.slider("Carrier amplitude Aᴄ", 0.5, 2.0, 1.0, 0.1)
    fc = sb.slider("Carrier frequency fᴄ (kHz)", 8, 40, 10, 1) * 1000.0

    sb.subheader("Message")
    kind = sb.selectbox("Waveform", ["Sine", "Two-tone", "Triangle", "Square"])
    fm = sb.slider("Fundamental fₘ (Hz)", 100, 1000, 500, 100)

    mu, detector, sideband = 0.7, "Coherent (synchronous)", "USB"
    if scheme == "Conventional AM":
        sb.subheader("AM settings")
        mu = sb.slider("Modulation index μ", 0.05, 2.0, 0.7, 0.05)
        detector = sb.selectbox("Detector", ["Envelope detector (ideal)",
                                             "Rectifier + LPF (diode detector)",
                                             "Coherent (synchronous)"])
    if scheme == "SSB":
        sb.subheader("SSB settings")
        sideband = sb.radio("Sideband", ["USB", "LSB"], horizontal=True)

    sb.subheader("Receiver")
    coherent = not (scheme == "Conventional AM" and not detector.startswith("Coherent"))
    phi = sb.slider("LO phase error φ (°)", -180, 180, 0, 5, disabled=not coherent)
    df = sb.slider("LO frequency error Δf (Hz)", -300, 300, 0, 5, disabled=not coherent)
    if coherent:
        cut_k = sb.slider("LPF cutoff (× message BW)", 1.1, 5.0, 1.5, 0.1)
        cut_cap = 0.7 * fc
    else:
        cut_k = sb.slider("Detector RC cutoff (× message BW)", 1.1, 10.0, 6.0, 0.5,
                          help="A real diode detector's RC network is set well above the message "
                               "bandwidth. A tight cutoff smooths away the distortion that "
                               "over-modulation produces.")
        cut_cap = 0.35 * fc

    sb.subheader("Channel")
    noise = sb.toggle("Add AWGN", value=False)
    snr = sb.slider("Input SNR (dB)", 0, 40, 25, 1, disabled=not noise)

    sb.subheader("Display")
    periods = sb.slider("Message periods shown", 1, 8, 3)
    yscale = sb.radio("Spectrum scale", ["dB", "Linear"], horizontal=True)
    win = sb.selectbox("FFT window", list(WINDOWS), index=0,
                       help="Hann: sharp lines, low leakage. Blackman-Harris: deepest floor. "
                            "Rectangular: ideal lines for exactly periodic signals.")
    span = sb.slider("Spectrum half-span around fᴄ (kHz)", 0.5, fc / 1e3, min(4.0, fc / 1e3), 0.5)

    # ---------------- Signals ----------------
    m1, W1 = make_message(kind, fm, fc)
    if scheme == "QAM":
        m2, W2 = make_message(kind, 1.5 * fm, fc)
    else:
        m2, W2 = np.zeros(N), 0.0
    W = max(W1, W2)
    cutoff = min(cut_k * W, cut_cap)

    p = dict(Ac=float(Ac), mu=float(mu), fc=float(fc), sideband=sideband, detector=detector,
             phi=int(phi) if coherent else 0, df=int(df) if coherent else 0,
             cutoff=float(cutoff), noise=bool(noise), snr=float(snr))
    s, env, r, y1, y2 = pipeline(scheme, m1, m2, p)

    # ---------------- KPIs ----------------
    Pm = float(np.mean(m1 ** 2))
    eta = mu ** 2 * Pm / (1 + mu ** 2 * Pm) if scheme == "Conventional AM" else 1.0
    bw = {"Conventional AM": 2, "DSB-SC": 2, "SSB": 1, "QAM": 2}[scheme] * W
    fid1 = fidelity(y1, m1)
    fid = fid1 if y2 is None else 0.5 * (fid1 + fidelity(y2, m2))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transmission bandwidth", f"{bw / 1e3:.2f} kHz")
    k2.metric("Power efficiency η", f"{100 * eta:.1f} %")
    k3.metric("Transmitted power ⟨s²⟩", f"{np.mean(s ** 2):.3f}")
    k4.metric("Recovery fidelity", f"{fid:.1f} dB")

    if scheme == "Conventional AM" and mu > 1 and not detector.startswith("Coherent"):
        st.warning("Over-modulation (μ > 1): this detector outputs |1 + μ·m(t)|, so wherever the "
                   "envelope would go negative (red bands in the plots) the output is folded — i.e. "
                   "inverted — instead of following the message. Switch to *Coherent* to recover it "
                   "correctly, or lower μ.")
    if coherent and (phi != 0 or df != 0):
        st.info(f"Local-oscillator mismatch active (φ = {phi}°, Δf = {df} Hz) — watch the recovered "
                "signal and the *Analysis* tab.")

    tab_mod, tab_dem, tab_a, tab_c, tab_th = st.tabs(
        ["📡 Modulator", "📻 Demodulator", "🔬 Analysis", "⚖️ Compare schemes", "📘 Theory"])

    # common display helpers
    i0 = int(0.010 * FS)
    n_win = min(int(periods / snap(fm) * FS), N - i0 - int(0.005 * FS))
    sl = slice(i0, i0 + n_win)
    tm = t[sl] * 1e3
    is_am = scheme == "Conventional AM"
    two = scheme == "QAM"
    lo_hi = (0.0, min(fc, max(4 * W, 3000)))
    tx_rng = (max(0.0, fc - span * 1e3), fc + span * 1e3)
    ylab = "Magnitude (dB)" if yscale == "dB" else "Amplitude"
    db_range = [-120, 5] if yscale == "dB" else None
    color_key = ("Colour code: 🟨 oscillators · 🟦 gain / processing · 🟩 filters · "
                 "🟪 non-linear / Hilbert · ⬜ carrier synchronisation. Values shown (fᴄ, μ, φ, Δf, "
                 "LPF cutoff) follow the sidebar settings.")
    res_note = (f"Spectral resolution: {1 / T_SIM:.0f} Hz bins, {ZPAD}× zero-padded "
                f"({1 / (T_SIM * ZPAD):.2f} Hz interpolation), {win} window.")

    # =====================================================================
    # MODULATOR TAB: block diagram + signals + spectrum
    # =====================================================================
    with tab_mod:
        st.subheader(f"{scheme} modulator")
        show_svg(block_modulator(scheme, p))
        st.latex(THEORY[scheme]["eq"][0])
        st.caption(color_key)

        st.markdown("#### Signals")
        ttl = ["Messages m₁(t), m₂(t)" if two else
               ("Message m(t) and its Hilbert transform m̂(t)" if scheme == "SSB" else "Message m(t)"),
               "Carrier(s)" if (two or scheme == "SSB") else "Carrier c(t)",
               "Modulated signal s(t) with envelope"]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                            subplot_titles=ttl)
        add(fig, go.Scatter(x=tm, y=m1[sl], name="m₁(t)" if two else "m(t)",
                            line=dict(color=C_MSG, width=2)), 1)
        if two:
            add(fig, go.Scatter(x=tm, y=m2[sl], name="m₂(t)", line=dict(color=C_MSG2, width=2)), 1)
        if scheme == "SSB":
            add(fig, go.Scatter(x=tm, y=hilbert_tf(m1)[sl], name="m̂(t)",
                                line=dict(color=C_MSG2, width=1.6, dash="dash")), 1)
        add(fig, go.Scatter(x=tm, y=Ac * np.cos(2 * np.pi * fc * t[sl]), name="Aᴄ cos ωᴄt",
                            line=dict(color=C_RX, width=1.2)), 2)
        if two or scheme == "SSB":
            add(fig, go.Scatter(x=tm, y=Ac * np.sin(2 * np.pi * fc * t[sl]), name="Aᴄ sin ωᴄt",
                                line=dict(color=C_ENV, width=1.2)), 2)
        add(fig, go.Scatter(x=tm, y=s[sl], name="s(t)", line=dict(color=C_TX, width=1)), 3)
        add(fig, go.Scatter(x=tm, y=env[sl], name="Envelope ±", line=dict(color=C_ENV, width=2, dash="dash")), 3)
        add(fig, go.Scatter(x=tm, y=-env[sl], name="Envelope −", showlegend=False,
                            line=dict(color=C_ENV, width=2, dash="dash")), 3)
        if is_am and mu > 1:
            shade_overmod(fig, (1 + mu * m1[sl]) < 0, tm, (3,), 3)
        fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude")
        multi_style(fig, 3)
        show(fig)

        st.markdown("#### Spectrum")
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                            subplot_titles=["Message spectrum", "Transmitted spectrum"])
        fx, fy = spec_view(m1, *lo_hi, win, yscale)
        add(fig, go.Scatter(x=fx, y=fy, name="m₁" if two else "m(t)", line=dict(color=C_MSG, width=1.3)), 1)
        if two:
            fx, fy = spec_view(m2, *lo_hi, win, yscale)
            add(fig, go.Scatter(x=fx, y=fy, name="m₂", line=dict(color=C_MSG2, width=1.3)), 1)
        fx, fy = spec_view(s, *tx_rng, win, yscale)
        add(fig, go.Scatter(x=fx, y=fy, name="s(t)", line=dict(color=C_TX, width=1.3)), 2)
        shade_bands(fig, 2, scheme, fc, W, sideband)
        fig.update_xaxes(range=[lo_hi[0] / 1e3, lo_hi[1] / 1e3], title_text="Frequency (kHz)", row=1, col=1)
        fig.update_xaxes(range=[tx_rng[0] / 1e3, tx_rng[1] / 1e3], title_text="Frequency (kHz)", row=2, col=1)
        fig.update_yaxes(title_text=ylab)
        if db_range:
            fig.update_yaxes(range=db_range)
        multi_style(fig, 2, row_h=230, extra=90)
        show(fig)
        st.caption(("Shaded: occupied sidebands around fᴄ (dotted line). " if scheme != "SSB" else
                    f"Green = transmitted {sideband}; grey = suppressed sideband; dotted line = fᴄ. ")
                   + res_note)

    # =====================================================================
    # DEMODULATOR TAB: block diagram + signals + spectrum
    # =====================================================================
    with tab_dem:
        st.subheader(f"{scheme} demodulator" + (f" — {detector}" if is_am else ""))
        show_svg(block_demodulator(scheme, p))
        st.latex(DEMOD_EQ[detector if is_am else scheme])
        st.caption(color_key)
        if is_am:
            with st.expander("Compare all three AM detector structures"):
                for name, svg in am_demod_svgs(p).items():
                    st.markdown(f"**{name}**")
                    show_svg(svg)
                    st.latex(DEMOD_EQ[name])
                st.caption("A hardware diode detector is a diode followed by an RC network: the diode "
                           "rectifies, the RC low-pass tracks the envelope, and a series capacitor "
                           "blocks the DC term.")

        stage = detector_stage(scheme, r, p)
        st.markdown("#### Signals")
        if is_am:
            raw = am_detect(r, p)
            ttl = ["Received signal r(t)",
                   ("Detector output (before DC block) — envelope, never negative"
                    if not detector.startswith("Coherent") else
                    "Coherent detector output (before DC block) — signed"),
                   "After DC block: recovered message"]
        elif two:
            ttl = ["Received signal r(t)", "In-phase output vs m₁(t)", "Quadrature output vs m₂(t)"]
        else:
            ttl = ["Received signal r(t)", "Mixer output before / after LPF", "Recovered message"]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.10, subplot_titles=ttl)
        add(fig, go.Scatter(x=tm, y=r[sl], name="r(t)" + (" + noise" if noise else ""),
                            line=dict(color=C_TX, width=1)), 1)
        add(fig, go.Scatter(x=tm, y=env[sl], name="Tx envelope ±", line=dict(color=C_ENV, width=1.6, dash="dash")), 1)
        add(fig, go.Scatter(x=tm, y=-env[sl], name="Tx envelope −", showlegend=False,
                            line=dict(color=C_ENV, width=1.6, dash="dash")), 1)
        if is_am:
            add(fig, go.Scatter(x=tm, y=env[sl], name="Aᴄ[1+μm] (ideal)",
                                line=dict(color=C_ENV, width=1.6, dash="dot")), 2)
            add(fig, go.Scatter(x=tm, y=raw[sl], name="Detector out", line=dict(color=C_OUT, width=2.4)), 2)
            fig.add_hline(y=0, line_color="#94a3b8", line_width=1, row=2, col=1)
            add(fig, go.Scatter(x=tm, y=m1[sl], name="Original m(t)",
                                line=dict(color=C_MSG, width=1.5, dash="dot")), 3)
            add(fig, go.Scatter(x=tm, y=y1[sl], name="Recovered", line=dict(color=C_OUT, width=2.2)), 3)
            if mu > 1:
                shade_overmod(fig, (1 + mu * m1[sl]) < 0, tm, (1, 2, 3), 1)
        elif two:
            add(fig, go.Scatter(x=tm, y=m1[sl], name="Original m₁", line=dict(color=C_MSG, width=1.5, dash="dot")), 2)
            add(fig, go.Scatter(x=tm, y=y1[sl], name="Recovered I", line=dict(color=C_OUT, width=2.2)), 2)
            add(fig, go.Scatter(x=tm, y=m2[sl], name="Original m₂", line=dict(color=C_MSG2, width=1.5, dash="dot")), 3)
            add(fig, go.Scatter(x=tm, y=y2[sl], name="Recovered Q", line=dict(color=C_OUT, width=2.2)), 3)
        else:
            add(fig, go.Scatter(x=tm, y=stage[0][sl], name="Mixer output", line=dict(color="#cbd5e1", width=1)), 2)
            add(fig, go.Scatter(x=tm, y=lpf(stage[0], cutoff)[sl], name="After LPF",
                                line=dict(color=C_OUT, width=2.2)), 2)
            add(fig, go.Scatter(x=tm, y=m1[sl], name="Original m(t)", line=dict(color=C_MSG, width=1.5, dash="dot")), 3)
            add(fig, go.Scatter(x=tm, y=y1[sl], name="Recovered", line=dict(color=C_OUT, width=2.2)), 3)
        fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude")
        multi_style(fig, 3)
        show(fig)

        st.markdown("#### Spectrum")
        stg_ttl = ("Detector stage before LPF" if is_am else "Mixer output before LPF") + \
                  " — LPF cutoff marked"
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.11,
                            subplot_titles=["Received spectrum", stg_ttl, "Recovered-message spectrum"])
        fx, fy = spec_view(r, *tx_rng, win, yscale)
        add(fig, go.Scatter(x=fx, y=fy, name="r(t)", line=dict(color=C_TX, width=1.3)), 1)
        shade_bands(fig, 1, scheme, fc, W, sideband)
        stg_rng = (0.0, 2.4 * fc)
        for j, x_ in enumerate(stage):
            fx, fy = spec_view(x_, *stg_rng, win, yscale)
            nm = ("Mixer I" if j == 0 else "Mixer Q") if two else ("Detector stage" if is_am else "Mixer")
            add(fig, go.Scatter(x=fx, y=fy, name=nm,
                                line=dict(color=C_MSG if j == 0 else C_MSG2, width=1.3)), 2)
        fig.add_vline(x=cutoff / 1e3, line_dash="dash", line_color=C_ENV, row=2, col=1)
        fx, fy = spec_view(m1, *lo_hi, win, yscale)
        add(fig, go.Scatter(x=fx, y=fy, name="Original", line=dict(color=C_MSG, width=1.2, dash="dot")), 3)
        fx, fy = spec_view(y1, *lo_hi, win, yscale)
        add(fig, go.Scatter(x=fx, y=fy, name="Recovered I" if two else "Recovered",
                            line=dict(color=C_OUT, width=1.3)), 3)
        if two:
            fx, fy = spec_view(y2, *lo_hi, win, yscale)
            add(fig, go.Scatter(x=fx, y=fy, name="Recovered Q", line=dict(color=C_MSG2, width=1.3)), 3)
        fig.update_xaxes(range=[tx_rng[0] / 1e3, tx_rng[1] / 1e3], title_text="Frequency (kHz)", row=1, col=1)
        fig.update_xaxes(range=[stg_rng[0] / 1e3, stg_rng[1] / 1e3], title_text="Frequency (kHz)", row=2, col=1)
        fig.update_xaxes(range=[lo_hi[0] / 1e3, lo_hi[1] / 1e3], title_text="Frequency (kHz)", row=3, col=1)
        fig.update_yaxes(title_text=ylab)
        if db_range:
            fig.update_yaxes(range=db_range)
        multi_style(fig, 3, row_h=230, extra=90)
        show(fig)
        st.caption("Dashed amber line = LPF cutoff. For coherent detectors the mixer output contains the "
                   "message at baseband plus an image around 2fᴄ, which the low-pass filter removes. " + res_note)

    # ---------------- Analysis ----------------
    with tab_a:
        left, right = st.columns(2)

        with left:
            if scheme == "Conventional AM":
                st.subheader("Trapezoidal pattern")
                fig = go.Figure(go.Scatter(x=m1[sl], y=s[sl], mode="lines",
                                           line=dict(color=C_TX, width=0.8)))
                fig.update_xaxes(title_text="m(t)")
                fig.update_yaxes(title_text="s(t)")
                style(fig, 380, hovermode="closest")
                show(fig)
                st.caption("A straight-edged trapezoid ⇒ linear modulation. Its parallel sides give "
                           "μ = (A − B)/(A + B); for μ > 1 the pattern turns into two triangles.")
                Pc = Ac ** 2 / 2
                Ps = Ac ** 2 * mu ** 2 * Pm / 2
                fig = go.Figure(go.Bar(x=["Carrier", "Sidebands"], y=[Pc, Ps],
                                       marker_color=[C_RX, C_TX],
                                       text=[f"{100 * Pc / (Pc + Ps):.1f}%", f"{100 * Ps / (Pc + Ps):.1f}%"],
                                       textposition="outside"))
                fig.update_yaxes(title_text="Power")
                style(fig, 280, title="Power split", hovermode="closest")
                show(fig)

            elif scheme == "DSB-SC":
                st.subheader("Why an envelope detector fails")
                e = np.abs(signal.hilbert(s))
                tm = t[sl] * 1e3
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tm, y=Ac * m1[sl], name="Ac·m(t)", line=dict(color=C_MSG, width=2)))
                fig.add_trace(go.Scatter(x=tm, y=e[sl], name="Envelope of s(t)",
                                         line=dict(color=C_ENV, width=2, dash="dash")))
                fig.update_xaxes(title_text="Time (ms)")
                style(fig, 380)
                show(fig)
                st.caption("The envelope of a DSB-SC wave is |m(t)|: the sign (a 180° carrier phase "
                           "reversal) is lost, so a coherent detector is mandatory.")

            elif scheme == "SSB":
                st.subheader("Sideband suppression")
                pu = band_power(s, fc, fc + 1.3 * W)
                pl = band_power(s, fc - 1.3 * W, fc)
                sup = 10 * np.log10(max(pu, pl) / min(pu, pl))
                fig = go.Figure(go.Bar(x=["Lower (LSB)", "Upper (USB)"],
                                       y=[10 * np.log10(pl / max(pu, pl)), 10 * np.log10(pu / max(pu, pl))],
                                       marker_color=[C_RX, C_TX]))
                fig.update_yaxes(title_text="Relative power (dB)")
                style(fig, 380, hovermode="closest")
                show(fig)
                st.caption(f"Sideband suppression ≈ {sup:.0f} dB from the ideal Hilbert phasing network.")

            else:  # QAM
                st.subheader("I/Q recovery (Lissajous)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m1[sl], y=m2[sl], name="Original (m₁, m₂)",
                                         line=dict(color=C_MSG, width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=y1[sl], y=y2[sl], name="Recovered (I, Q)",
                                         line=dict(color=C_OUT, width=2)))
                fig.update_xaxes(title_text="In-phase", scaleanchor="y", scaleratio=1)
                fig.update_yaxes(title_text="Quadrature")
                style(fig, 380, hovermode="closest")
                show(fig)
                st.caption("A phase error rotates and shrinks the recovered figure (crosstalk between "
                           "I and Q); a frequency error makes it spin continuously.")

        with right:
            if scheme == "Conventional AM":
                st.subheader("Fidelity vs modulation index")
                mus, fids = sweep_mu(scheme, kind, fm, p)
                fig = go.Figure(go.Scatter(x=mus, y=fids, mode="lines+markers", name="Sweep",
                                           line=dict(color=C_OUT, width=2.4)))
                fig.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8", annotation_text="μ = 1")
                fig.add_trace(go.Scatter(x=[mu], y=[fid], mode="markers", name="Current",
                                         marker=dict(color=C_TX, size=13, symbol="diamond")))
                fig.update_xaxes(title_text="Modulation index μ")
                fig.update_yaxes(title_text="Fidelity (dB)")
                style(fig, 380, hovermode="closest")
                show(fig)
                st.caption(f"Detector: **{detector}**. Envelope-type detectors collapse once μ > 1; "
                           "coherent detection is unaffected.")
            else:
                st.subheader("Fidelity vs LO phase error")
                phis, fids = sweep_phase(scheme, kind, fm, p)
                fig = go.Figure(go.Scatter(x=phis, y=fids, mode="lines", name="Sweep",
                                           line=dict(color=C_OUT, width=2.4)))
                fig.add_trace(go.Scatter(x=[phi], y=[fid1], mode="markers", name="Current",
                                         marker=dict(color=C_TX, size=13, symbol="diamond")))
                fig.update_xaxes(title_text="Phase error φ (°)", tickvals=list(range(-180, 181, 45)))
                fig.update_yaxes(title_text="Fidelity (dB)")
                style(fig, 380, hovermode="closest")
                show(fig)
                msg = {"DSB-SC": "Output ∝ cos φ: full recovery at 0°, a null at ±90°, polarity inversion at 180°.",
                       "SSB": "Amplitude is preserved, but the output becomes m·cos φ ± m̂·sin φ — pure phase distortion.",
                       "QAM": "The I output is m₁ cos φ − m₂ sin φ, so quadrature crosstalk grows quickly with φ."}[scheme]
                st.caption(msg)

    # ---------------- Compare ----------------
    with tab_c:
        st.subheader("Same message, four schemes")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=SCHEMES)
        m2c, _ = make_message(kind, 1.5 * fm, fc)
        colors = [C_TX, C_MSG, C_OUT, C_MSG2]
        rng = (max(0.0, fc - 2.2 * W), fc + 2.2 * W)
        for i, sc in enumerate(SCHEMES, start=1):
            sig, _ = modulate(sc, m1, m2c, p)
            fx, fy = spec_view(sig, *rng, win, yscale)
            fig.add_trace(go.Scatter(x=fx, y=fy, showlegend=False, line=dict(color=colors[i - 1], width=1.3)), i, 1)
            fig.add_vline(x=fc / 1e3, line_dash="dot", line_color="#94a3b8", row=i, col=1)
        fig.update_xaxes(range=[rng[0] / 1e3, rng[1] / 1e3])
        fig.update_xaxes(title_text="Frequency (kHz)", row=4, col=1)
        fig.update_yaxes(title_text="dB" if yscale == "dB" else "Amp.")
        if yscale == "dB":
            fig.update_yaxes(range=[-120, 5])
        style(fig, 760)
        show(fig)
        show_table(COMPARE_TABLE)

    # ---------------- Theory ----------------
    with tab_th:
        for sc in SCHEMES:
            with st.expander(sc, expanded=(sc == scheme)):
                for eq in THEORY[sc]["eq"]:
                    st.latex(eq)
                st.write(THEORY[sc]["txt"])

    st.caption(f"Simulation: fs = {FS / 1e3:.0f} kHz · {T_SIM * 1e3:.0f} ms record · "
               f"message BW W ≈ {W:.0f} Hz · demodulator LPF: 6th-order Butterworth @ {cutoff:.0f} Hz")


if __name__ == "__main__":
    main()
