# -*- coding: utf-8 -*-
"""
Swan 45 — Simulatore di regolazione vele (v2)
=============================================
Modello quasi-3D a "strip theory": ogni vela è divisa in fasce
d'altezza, ognuna con il proprio vento apparente (gradiente + twist
dell'apparente), incidenza, portanza e stato del flusso (filetti).
Include visualizzatore 3D della forma delle vele e flusso potenziale
2D (metodo dei vortici concentrati).

Avvio:
    pip install streamlit numpy matplotlib
    streamlit run swan45_trim_sim.py

NOTA: modello didattico calibrato su ordini di grandezza realistici
per uno Swan 45. Non sostituisce una CFD RANS né un VPP certificato.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Costanti e dati barca (Swan 45)
# ----------------------------------------------------------------------
RHO_AIR = 1.225
RHO_W = 1025.0
G = 9.81
KN = 0.5144

LWL = 12.15          # m
DISP = 9900.0        # kg
GM = 1.85            # m
RM_CREW = 38000.0    # Nm, equipaggio alla banda + stabilità di forma
Z_CE = 8.6           # m
SA_MAIN = 60.0       # m2
SA_JIB = 52.0        # m2
SA_TOT = SA_MAIN + SA_JIB
RIG_I = 19.2         # m
WSA = 34.0           # m2
AR_RIG = RIG_I ** 2 / SA_TOT

Z_BOOM = 2.0         # m, quota del boma
MAIN_SPAN = 17.0     # m, inferitura randa
Z_JIB_TACK = 1.3     # m, mura fiocco
JIB_SPAN = 16.2      # m
N_STRIPS = 9

ALPHA_LUFF = 1.5     # deg: sotto questa incidenza la vela rifiuta


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------
# Regolazioni -> forma delle vele
# ----------------------------------------------------------------------
@dataclass
class Trim:
    main_sheet: float
    traveller: float
    vang: float
    outhaul: float
    cunningham: float
    backstay: float
    jack_mm: float       # 0..30 mm  martinetto: spessori sotto il piede d'albero
    jib_sheet: float
    jib_car: float
    jib_halyard: float
    shroud_cap: float    # 0..1  sartie alte / esterne (V1, cap)
    shroud_d2: float     # 0..1  diagonali intermedie (D2)
    shroud_d1: float     # 0..1  sartie basse (D1)
    rake: float


@dataclass
class SailShape:
    main_camber: float
    main_draft: float
    main_angle: float
    main_twist: float
    jib_camber: float
    jib_draft: float
    jib_angle: float
    jib_twist: float
    forestay_sag: float
    rake_deg: float
    shroud_cap: float    # tensioni EFFICACI (regolazione × martinetto)
    shroud_d2: float
    shroud_d1: float
    backstay: float
    jack_mm: float


def shape_from_trim(t: Trim) -> SailShape:
    # Il paterazzo flette l'albero: quanto e dove dipende dalle sartie.
    # Il grasso "globale" qui è alla base; la distribuzione in altezza
    # (effetto D1/D2/cap) è calcolata striscia per striscia.
    main_camber = clamp(0.115 - 0.030 * t.backstay - 0.025 * t.outhaul
                        - 0.006 * t.cunningham, 0.045, 0.15)
    main_draft = clamp(0.45 + 0.06 * t.backstay - 0.10 * t.cunningham, 0.30, 0.55)
    main_angle = clamp(32.0 * (1.0 - t.main_sheet) + 10.0 * (0.5 - t.traveller), 0.0, 45.0)
    main_twist = clamp(6.0 + 20.0 * (1.0 - t.main_sheet) + 10.0 * t.backstay
                       - 14.0 * t.vang + 5.0 * (0.5 - t.traveller), 2.0, 35.0)

    # Martinetto: alzare l'albero pre-tensiona TUTTO il sartiame in modo
    # proporzionale. 0 mm = rig molle (~55% della tensione nominale),
    # 30 mm = rig molto carico (~130%).
    jack = clamp(t.jack_mm / 30.0, 0.0, 1.0)
    m_jack = 0.55 + 0.75 * jack
    cap_eff = clamp(t.shroud_cap * m_jack, 0.0, 1.2)
    d2_eff = clamp(t.shroud_d2 * m_jack, 0.0, 1.2)
    d1_eff = clamp(t.shroud_d1 * m_jack, 0.0, 1.2)

    # catenaria strallo: paterazzo, martinetto e sartie alte la riducono
    sag = clamp(0.055 * (1.0 - 0.60 * t.backstay) * (1.0 - 0.30 * jack)
                * (1.0 - 0.30 * min(cap_eff, 1.0)), 0.005, 0.06)
    jib_camber = clamp(0.090 + 1.2 * sag - 0.020 * t.jib_halyard, 0.05, 0.16)
    jib_draft = clamp(0.42 + 0.8 * sag - 0.08 * t.jib_halyard, 0.30, 0.52)
    jib_angle = clamp(9.0 + 18.0 * (1.0 - t.jib_sheet), 5.0, 35.0)
    jib_twist = clamp(4.0 + 16.0 * (1.0 - t.jib_sheet) + 16.0 * (t.jib_car - 0.35), 2.0, 35.0)

    rake_deg = 0.5 + 2.0 * t.rake
    return SailShape(main_camber, main_draft, main_angle, main_twist,
                     jib_camber, jib_draft, jib_angle, jib_twist, sag, rake_deg,
                     cap_eff, d2_eff, d1_eff, t.backstay, t.jack_mm)


# ----------------------------------------------------------------------
# Aerodinamica di sezione (profilo sottile + stallo)
# ----------------------------------------------------------------------
def sail_cl_cd(alpha_deg: float, camber: float):
    cl0 = 4.0 * camber
    slope = 0.070
    alpha_stall = 12.0 + 40.0 * camber
    a = alpha_deg
    if a < -5.0:
        cl = max(-0.3, cl0 + slope * a * 0.3)
        cd_sep = 0.05
    elif a <= alpha_stall:
        cl = cl0 + slope * a
        cd_sep = 0.008 + 0.010 * (a / alpha_stall) ** 2
    else:
        over = a - alpha_stall
        cl_max = cl0 + slope * alpha_stall
        cl = max(0.7, cl_max - 0.030 * over)
        cd_sep = 0.02 + 0.012 * over
    return cl, cd_sep, alpha_stall


# ----------------------------------------------------------------------
# Vento con gradiente verticale e apparente locale
# ----------------------------------------------------------------------
def wind_profile(tws10_ms: float, z):
    """Legge di potenza: il vento cresce con la quota."""
    return tws10_ms * (np.maximum(z, 1.0) / 10.0) ** 0.11


def apparent_wind(tws_ms, twa_deg, v_ms):
    twa = np.radians(twa_deg)
    ax = tws_ms * np.cos(twa) + v_ms
    ay = tws_ms * np.sin(twa)
    return np.hypot(ax, ay), np.degrees(np.arctan2(ay, ax))


# ----------------------------------------------------------------------
# Teoria delle strisce: analisi per fasce d'altezza
# ----------------------------------------------------------------------
SAIL_GEO = {
    "main": dict(z0=Z_BOOM, span=MAIN_SPAN, area=SA_MAIN, taper=0.85),
    "jib": dict(z0=Z_JIB_TACK, span=JIB_SPAN, area=SA_JIB, taper=0.92),
}


def strip_analysis(which: str, shape: SailShape, tws10_ms, twa, v_ms):
    g = SAIL_GEO[which]
    hs = (np.arange(N_STRIPS) + 0.5) / N_STRIPS
    z = g["z0"] + g["span"] * hs

    tws_z = wind_profile(tws10_ms, z)
    aws, awa = apparent_wind(tws_z, twa, v_ms)

    if which == "main":
        angle0, twist = shape.main_angle, shape.main_twist
        camber0, draft = shape.main_camber, shape.main_draft
    else:
        angle0, twist = shape.jib_angle, shape.jib_twist
        camber0, draft = shape.jib_camber, shape.jib_draft

    angle_local = angle0 + twist * hs ** 0.9
    camber_local = camber0 * (1.0 - 0.12 * hs)      # testa un filo più magra

    if which == "main":
        # Prebend locale: il paterazzo flette dove le sartie lo lasciano fare.
        # D1 lasche -> flessione bassa (smagrisce in basso);
        # D2 lasche -> flessione a metà (smagrisce la pancia);
        # sartie alte lasche -> la testa cade sottovento (twist extra in penna).
        bend_low = 0.022 * shape.main_camber / 0.115 * (1.3 - shape.shroud_d1) \
            * (1.0 - hs) ** 1.5
        bend_mid = 0.020 * shape.main_camber / 0.115 * (1.2 - shape.shroud_d2) \
            * np.exp(-((hs - 0.55) / 0.25) ** 2)
        camber_local = np.clip(camber_local - (bend_low + bend_mid)
                               * (0.25 + 0.75 * shape.backstay), 0.030, 0.16)
        angle_local = angle_local + max(0.0, 1.0 - shape.shroud_cap) * 7.0 * hs ** 2

    alpha = awa - angle_local

    cl = np.zeros(N_STRIPS)
    cd_sep = np.zeros(N_STRIPS)
    stall = np.zeros(N_STRIPS)
    for i in range(N_STRIPS):
        cl[i], cd_sep[i], stall[i] = sail_cl_cd(float(alpha[i]), float(camber_local[i]))

    # stato filetti: -1 rifiuto (inferitura), 0 ok, +1 stallo (balumina)
    state = np.where(alpha < ALPHA_LUFF, -1, np.where(alpha > stall, 1, 0))

    # aree per fascia (rastremazione verso la penna), normalizzate all'area vera
    w = 1.0 - g["taper"] * hs
    areas = w / w.sum() * g["area"]

    q = 0.5 * RHO_AIR * aws ** 2
    return dict(h=hs, z=z, aws=aws, awa=awa, alpha=alpha, cl=cl, cd_sep=cd_sep,
                stall=stall, state=state, areas=areas, q=q,
                angle_local=angle_local, camber=camber_local, draft=draft)


def rig_forces(tws10_ms, twa, v_ms, heel_deg, shape: SailShape):
    """Forze del piano velico integrate sulle strisce, riferimento barca."""
    sm = strip_analysis("main", shape, tws10_ms, twa, v_ms)
    sj = strip_analysis("jib", shape, tws10_ms, twa, v_ms)

    # effetto fessura: il fiocco lavora meglio se il canale è aperto
    slot = 1.0 + 0.10 * np.exp(-((shape.jib_angle - shape.main_angle - 8.0) / 8.0) ** 2)
    sj = dict(sj, cl=sj["cl"] * slot)

    eff = np.cos(np.radians(heel_deg)) ** 1.3
    beta = np.radians(0.5 * (sm["awa"].mean() + sj["awa"].mean()))

    # portanza e resistenza di separazione, striscia per striscia
    L = D = qa = cl_area = 0.0
    for s in (sm, sj):
        L += float(np.sum(s["q"] * s["areas"] * s["cl"]))
        D += float(np.sum(s["q"] * s["areas"] * s["cd_sep"]))
        qa += float(np.sum(s["q"] * s["areas"]))
        cl_area += float(np.sum(s["areas"] * s["cl"]))

    cl_mean = cl_area / SA_TOT
    # resistenza indotta globale + parassita del rig
    e_osw = 0.80
    D += qa * (0.015 + cl_mean ** 2 / (np.pi * AR_RIG * e_osw))

    # Rig in bando: se la tensione efficace delle alte è bassa e il vento
    # preme, il sartiame sottovento va in bando e l'intero piano velico
    # cade sottovento -> meno portanza utile, più resistenza.
    slack = max(0.0, 1.0 - shape.shroud_cap)
    wind_f = clamp((tws10_ms / KN - 8.0) / 10.0, 0.0, 1.0)
    L *= 1.0 - 0.12 * slack * wind_f
    D *= 1.0 + 0.18 * slack * wind_f

    L *= eff
    D *= eff ** 0.5
    fx = L * np.sin(beta) - D * np.cos(beta)
    fy = (L * np.cos(beta) + D * np.sin(beta)) * eff ** 0.5

    # twist ideale = twist del vento apparente lungo l'altezza
    ideal_twist = float(sm["awa"][-1] - sm["awa"][0])

    diag = dict(cl=cl_mean, cd=D / max(qa, 1e-6), q=qa,
                lift=L, drag=D,
                strips_main=sm, strips_jib=sj,
                ideal_twist=ideal_twist, e=e_osw,
                head_luff_main=bool(np.any(sm["state"][-3:] == -1)),
                head_stall_main=bool(np.any(sm["state"][-3:] == 1)),
                head_luff_jib=bool(np.any(sj["state"][-3:] == -1)),
                stalled_main=bool(np.mean(sm["state"] == 1) > 0.4),
                stalled_jib=bool(np.mean(sj["state"] == 1) > 0.4))
    return fx, fy, diag


# ----------------------------------------------------------------------
# Idrodinamica ed equilibrio
# ----------------------------------------------------------------------
def hull_resistance(v_ms, heel_deg, helm_deg):
    if v_ms <= 0.01:
        return 1.0
    fn = v_ms / np.sqrt(G * LWL)
    rv = 0.5 * RHO_W * v_ms ** 2 * WSA * 0.0045
    rr = DISP * G * 0.040 * (fn / 0.45) ** 8
    heel_pen = 1.0 + 0.6 * (heel_deg / 30.0) ** 2
    rudder = 0.5 * RHO_W * v_ms ** 2 * 1.1 * 0.02 * (abs(helm_deg) / 5.0) ** 2
    return (rv + rr) * heel_pen + rudder


def solve_equilibrium(tws_kn, twa_deg, trim: Trim):
    shape = shape_from_trim(trim)
    tws = tws_kn * KN

    def evaluate(v):
        heel = 8.0
        for _ in range(8):
            fx, fy, diag = rig_forces(tws, twa_deg, v, heel, shape)
            rm_max = DISP * G * GM + RM_CREW
            heel_new = np.degrees(np.arcsin(clamp(fy * Z_CE / rm_max, 0.0, 0.80)))
            heel += 0.6 * (heel_new - heel)
        rig_avg = min((shape.shroud_cap + shape.shroud_d2 + shape.shroud_d1) / 3.0, 1.0)
        helm = 0.5 + 5.0 * trim.rake + 2.5 * (trim.traveller - 0.5) + 0.18 * heel \
            - 1.0 * (1.0 - rig_avg)
        drag = hull_resistance(v, heel, helm)
        return fx - drag, heel, helm, diag

    lo, hi = 0.05, 13.0 * KN
    if evaluate(lo)[0] <= 0:
        v = lo
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if evaluate(mid)[0] > 0:
                lo = mid
            else:
                hi = mid
        v = 0.5 * (lo + hi)

    _, heel, helm, diag = evaluate(v)
    aws_mid, awa_mid = apparent_wind(wind_profile(tws, Z_CE), twa_deg, v)
    vmg = v / KN * np.cos(np.radians(twa_deg))
    return dict(v_kn=v / KN, heel=heel, helm=helm, aws_kn=aws_mid / KN,
                awa=float(awa_mid), vmg=vmg, shape=shape, diag=diag)


# ----------------------------------------------------------------------
# Flusso 2D: vortici concentrati su fiocco + randa
# ----------------------------------------------------------------------
def camber_line(camber, draft, chord, n):
    x = np.linspace(0.0, 1.0, n)
    p = clamp(draft, 0.2, 0.8)
    y = np.where(x < p,
                 camber / p ** 2 * (2 * p * x - x ** 2),
                 camber / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
    return x * chord, y * chord


def build_panels(shape: SailShape, awa_deg, n=26):
    sails = []
    geo = [(shape.jib_camber, shape.jib_draft, 1.15, shape.jib_angle, -1.35, 0.10),
           (shape.main_camber, shape.main_draft, 1.00, shape.main_angle, 0.0, 0.0)]
    for camber, draft, chord, ang, x0, y0 in geo:
        xs, ys = camber_line(camber, draft, chord, n)
        a = np.radians(awa_deg - ang)
        xr = x0 + xs * np.cos(a) + ys * np.sin(a)
        yr = y0 - (-xs * np.sin(a) + ys * np.cos(a))
        sails.append((xr, yr))
    return sails


def solve_vortices(sails):
    vx, vy, cx, cy, nx, ny, ds, sid = [], [], [], [], [], [], [], []
    for k, (xs, ys) in enumerate(sails):
        for i in range(len(xs) - 1):
            dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            length = np.hypot(dx, dy)
            vx.append(xs[i] + 0.25 * dx); vy.append(ys[i] + 0.25 * dy)
            cx.append(xs[i] + 0.75 * dx); cy.append(ys[i] + 0.75 * dy)
            nx.append(-dy / length); ny.append(dx / length)
            ds.append(length); sid.append(k)
    vx, vy = np.array(vx), np.array(vy)
    cx, cy = np.array(cx), np.array(cy)
    nx, ny = np.array(nx), np.array(ny)
    m = len(vx)
    A = np.zeros((m, m))
    for i in range(m):
        rx, ry = cx[i] - vx, cy[i] - vy
        r2 = rx ** 2 + ry ** 2
        A[i, :] = (ry / (2 * np.pi * r2)) * nx[i] + (-rx / (2 * np.pi * r2)) * ny[i]
    gam = np.linalg.solve(A, -nx)
    return vx, vy, gam, np.array(ds), np.array(sid)


def velocity_field(X, Y, vx, vy, gam):
    U = np.ones_like(X)
    V = np.zeros_like(X)
    for xg, yg, g in zip(vx, vy, gam):
        rx, ry = X - xg, Y - yg
        r2 = rx ** 2 + ry ** 2 + 1e-4
        U += g * ry / (2 * np.pi * r2)
        V += -g * rx / (2 * np.pi * r2)
    return U, V


# ----------------------------------------------------------------------
# Vela 3D con filetti
# ----------------------------------------------------------------------
STATE_COLORS = {-1: "#d62728", 0: "#2ca02c", 1: "#ff7f0e"}


def sail_surface(which, shape: SailShape, strips, n_h=22, n_c=13):
    """Superficie (X avanti, Y sottovento, Z alto) + filetti."""
    g = SAIL_GEO[which]
    hs = np.linspace(0.0, 1.0, n_h)
    if which == "main":
        angle0, twist = shape.main_angle, shape.main_twist
        camber0, draft = shape.main_camber, shape.main_draft
        chord0 = 6.0
        # inferitura sull'albero, inclinato del rake verso poppa
        x_luff = 0.0 - np.tan(np.radians(shape.rake_deg)) * (g["z0"] + g["span"] * hs)
    else:
        angle0, twist = shape.jib_angle, shape.jib_twist
        camber0, draft = shape.jib_camber, shape.jib_draft
        chord0 = 5.4
        # inferitura sullo strallo: dalla mura di prua verso la testa d'albero
        x_head = -np.tan(np.radians(shape.rake_deg)) * (Z_JIB_TACK + JIB_SPAN)
        x_luff = 6.2 + (x_head - 6.2) * hs
        # catenaria dello strallo (sag sottovento)
        x_luff = x_luff  # la catenaria la mettiamo su Y

    X = np.zeros((n_h, n_c))
    Y = np.zeros((n_h, n_c))
    Z = np.zeros((n_h, n_c))
    for j, h in enumerate(hs):
        chord = chord0 * (1.0 - g["taper"] * h)
        cam = camber0 * (1.0 - 0.12 * h)
        ang = np.radians(angle0 + twist * h ** 0.9)
        xc, yc = camber_line(cam, draft, chord, n_c)
        X[j] = x_luff[j] - (xc * np.cos(ang) - yc * np.sin(ang))
        Y[j] = xc * np.sin(ang) + yc * np.cos(ang)
        if which == "jib":
            Y[j] += shape.forestay_sag * chord0 * np.sin(np.pi * h) * 0.9
        else:
            Y[j] += mast_lateral(g["z0"] + g["span"] * h, shape,
                                 Z_BOOM + MAIN_SPAN + 0.4)
        Z[j] = g["z0"] + g["span"] * h

    # colore per fascia dallo stato dei filetti (interpolato sulle strisce)
    idx = np.clip((hs * N_STRIPS).astype(int), 0, N_STRIPS - 1)
    return X, Y, Z, strips["state"][idx]


def telltales_3d(ax, X, Y, Z, strips, which):
    """Filetti alla balumina (e all'inferitura per il fiocco)."""
    n_h = X.shape[0]
    for k in range(N_STRIPS):
        j = int((k + 0.5) / N_STRIPS * (n_h - 1))
        state = int(strips["state"][k])
        col = STATE_COLORS[state]
        # balumina: filetto in linea (verde) o che ricade (stallo)
        x0, y0, z0 = X[j, -1], Y[j, -1], Z[j, -1]
        if state == 1:
            dx, dy, dz = 0.25, 0.35, -0.35     # vortica sottovento
        else:
            dx, dy, dz = -0.55, 0.10, 0.0      # fila dritto a poppa
        ax.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color=col, lw=2)
        if which == "jib":
            # inferitura: rifiuto = filetto sopravvento che si alza
            xi, yi, zi = X[j, 2], Y[j, 2] - 0.05, Z[j, 2]
            if state == -1:
                ax.plot([xi, xi - 0.15], [yi, yi - 0.25], [zi, zi + 0.35],
                        color=STATE_COLORS[-1], lw=2)
            else:
                ax.plot([xi, xi - 0.35], [yi, yi - 0.05], [zi, zi],
                        color=STATE_COLORS[0], lw=1.5)


# ----------------------------------------------------------------------
# Digital twin: scafo parametrico Swan 45 + scena 3D Plotly
# ----------------------------------------------------------------------
# 🎨 PERSONALIZZAZIONE DALLE FOTO: modifica questi valori per adattare
# l'aspetto alla barca reale (colori, bordo libero, tuga...).
HULL_COLOR = "#f2f1ec"       # murata
ANTIFOUL_COLOR = "#27496d"   # opera viva
DECK_COLOR = "#e8dcc0"       # coperta (teak chiaro)
COACHROOF_COLOR = "#dedede"  # tuga
SAIL_OPACITY = 0.78
BOAT_NAME = "Swan 45"

LOA_HALF = 6.9               # metà lunghezza fuori tutto
BEAM_HALF = 1.95             # metà baglio massimo
FREEBOARD = 1.05             # bordo libero a mezza barca
CANOE_DEPTH = 0.62           # immersione del solo scafo


def _rot_heel(y, z, heel_deg):
    """Ruota attorno all'asse longitudinale (sbandamento sottovento = +Y)."""
    p = np.radians(heel_deg)
    return y * np.cos(p) + z * np.sin(p), -y * np.sin(p) + z * np.cos(p)


def hull_geometry(n_x=64, n_t=20):
    """Superficie dello scafo (mezza sezione specchiata) + coperta."""
    x = np.linspace(-LOA_HALF, LOA_HALF, n_x)      # poppa -> prua
    xm = -0.7                                       # stazione del baglio massimo
    lf, la = LOA_HALF - xm, xm + LOA_HALF
    fwd = x > xm
    bb = np.where(fwd, 1.0 - (np.abs(x - xm) / lf) ** 2.2,
                  1.0 - 0.45 * (np.abs(xm - x) / la) ** 1.8)
    halfb = BEAM_HALF * np.clip(bb, 0.0, 1.0) ** 0.7
    depth = CANOE_DEPTH * np.clip(
        np.where(fwd, 1.0 - (np.abs(x - xm) / lf) ** 1.6,
                 1.0 - 0.70 * (np.abs(xm - x) / la) ** 1.5), 0.0, 1.0)
    sheer = FREEBOARD + 0.28 * (x / LOA_HALF) ** 2 + 0.08 * (x / LOA_HALF)

    t = np.linspace(0.0, 1.0, n_t)
    Xg = np.repeat(x[:, None], n_t, axis=1)
    Zg = -depth[:, None] + t[None, :] * (sheer[:, None] + depth[:, None])
    Yg = halfb[:, None] * np.sin(0.5 * np.pi * t[None, :]) ** 0.85
    return Xg, Yg, Zg, x, halfb, sheer


def _surface(X, Y, Z, color, heel, opacity=1.0, twocolor=None):
    Yr, Zr = _rot_heel(Y, Z, heel)
    if twocolor:  # opera viva/morta separate alla linea di galleggiamento
        sc = (Z < -0.02).astype(float)
        colorscale = [[0.0, color], [0.5, color],
                      [0.5, twocolor], [1.0, twocolor]]
        return go.Surface(x=X, y=Yr, z=Zr, surfacecolor=sc, colorscale=colorscale,
                          cmin=0, cmax=1, showscale=False, opacity=opacity,
                          lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25),
                          hoverinfo="skip")
    return go.Surface(x=X, y=Yr, z=Zr,
                      surfacecolor=np.zeros_like(Z),
                      colorscale=[[0, color], [1, color]], showscale=False,
                      opacity=opacity,
                      lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25),
                      hoverinfo="skip")


def _line3d(xs, ys, zs, color, width, heel):
    ys, zs = _rot_heel(np.asarray(ys, float), np.asarray(zs, float), heel)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=width),
                        showlegend=False, hoverinfo="skip")


def boat_traces(heel):
    """Scafo, coperta, tuga, chiglia e timone, sbandati di `heel` gradi."""
    tr = []
    X, Y, Z, x, halfb, sheer = hull_geometry()
    tr.append(_surface(X, Y, Z, HULL_COLOR, heel, twocolor=ANTIFOUL_COLOR))
    tr.append(_surface(X, -Y, Z, HULL_COLOR, heel, twocolor=ANTIFOUL_COLOR))

    # coperta con bolzone
    tdk = np.linspace(-1.0, 1.0, 12)
    Xd = np.repeat(x[:, None], 12, axis=1)
    Yd = halfb[:, None] * tdk[None, :]
    Zd = sheer[:, None] + 0.06 * (1.0 - tdk[None, :] ** 2)
    tr.append(_surface(Xd, Yd, Zd, DECK_COLOR, heel))

    # tuga
    xt = np.linspace(-2.6, 1.6, 14)
    wt = 1.15 * (1.0 - 0.35 * ((xt - (-0.5)) / 2.1) ** 2)
    ht = np.interp(xt, x, sheer) + 0.42 * (1.0 - ((xt - (-0.5)) / 2.5) ** 2)
    Xt = np.repeat(xt[:, None], 10, axis=1)
    Yt = wt[:, None] * np.linspace(-1, 1, 10)[None, :]
    Zt = np.repeat(ht[:, None], 10, axis=1) - 0.05 * np.abs(np.linspace(-1, 1, 10))[None, :]
    tr.append(_surface(Xt, Yt, Zt, COACHROOF_COLOR, heel))

    # chiglia a T con bulbo e timone
    for x0, x1, z0, z1, thick in [(-0.9, 0.4, -0.55, -2.75, 0.09),
                                  (-5.6, -5.1, -0.35, -2.15, 0.05)]:
        xs = np.linspace(x0, x1, 6)
        zs = np.linspace(z0, z1, 8)
        Xk, Zk = np.meshgrid(xs, zs)
        taper = 1.0 - 0.5 * (Zk - z0) / (z1 - z0)
        tr.append(_surface(Xk, thick * taper, Zk, "#555b60", heel))
        tr.append(_surface(Xk, -thick * taper, Zk, "#555b60", heel))
    u = np.linspace(0, 2 * np.pi, 18)
    vb = np.linspace(0, np.pi, 10)
    Xb = -0.3 + 1.1 * np.outer(np.cos(u), np.sin(vb))
    Yb = 0.17 * np.outer(np.sin(u), np.sin(vb))
    Zb = -2.75 + 0.17 * np.outer(np.ones_like(u), np.cos(vb))
    tr.append(_surface(Xb, Yb, Zb, "#3a4046", heel))
    return tr


SPREADER_Z = (8.4, 13.8)          # quote delle due crocette
SPREADER_LEN = (1.15, 0.85)
CHAINPLATE = (0.0, 1.85, 1.15)    # x, y, z della landa


def mast_lateral(z, shape: SailShape, top):
    """Caduta laterale dell'albero (sottovento) con sartie alte lasche."""
    slack = max(0.0, 1.0 - shape.shroud_cap)
    return slack * 0.40 * (np.asarray(z) / top) ** 2


def rig_traces(shape: SailShape, heel):
    tr = []
    rake = np.tan(np.radians(shape.rake_deg))
    top = Z_BOOM + MAIN_SPAN + 0.4

    # albero: flesso lateralmente se le alte sono lasche
    zm = np.linspace(1.15, top, 24)
    xm = -rake * zm
    ym = mast_lateral(zm, shape, top)
    tr.append(_line3d(xm, ym, zm, "#2b2b2b", 7, heel))

    bo = np.radians(shape.main_angle)
    tr.append(_line3d([0, -6.0 * np.cos(bo)], [0, 6.0 * np.sin(bo)],
                      [Z_BOOM, Z_BOOM], "#2b2b2b", 6, heel))                        # boma
    x_head, y_head = -rake * top, float(mast_lateral(top, shape, top))
    tr.append(_line3d([6.4, x_head], [0, y_head], [1.25, top], "#777", 2.5, heel))  # strallo
    tr.append(_line3d([x_head, -6.6], [y_head, 0], [top, 1.35], "#777", 2.5, heel))  # paterazzo

    # crocette e sartiame per lato: V1 (alte/esterne), D2, D1
    def w(t):  # spessore proporzionale alla tensione
        return 1.2 + 3.5 * t

    z1, z2 = SPREADER_Z
    for side in (1, -1):
        s1 = (float(-rake * z1), side * SPREADER_LEN[0] + float(mast_lateral(z1, shape, top)), z1)
        s2 = (float(-rake * z2), side * SPREADER_LEN[1] + float(mast_lateral(z2, shape, top)), z2)
        r1 = (float(-rake * z1), float(mast_lateral(z1, shape, top)), z1)
        r2 = (float(-rake * z2), float(mast_lateral(z2, shape, top)), z2)
        head = (x_head, y_head, top)
        land = (CHAINPLATE[0], side * CHAINPLATE[1], CHAINPLATE[2])
        # crocette
        for root, tip in [(r1, s1), (r2, s2)]:
            tr.append(_line3d([root[0], tip[0]], [root[1], tip[1]],
                              [root[2], tip[2]], "#2b2b2b", 4, heel))
        # V1: testa -> crocetta alta -> crocetta bassa -> landa (esterne)
        tr.append(_line3d([head[0], s2[0], s1[0], land[0]],
                          [head[1], s2[1], s1[1], land[1]],
                          [head[2], s2[2], s1[2], land[2]],
                          "#8a8f94", w(shape.shroud_cap), heel))
        # D2: radice crocetta alta -> punta crocetta bassa
        tr.append(_line3d([r2[0], s1[0]], [r2[1], s1[1]], [r2[2], s1[2]],
                          "#a5742a", w(shape.shroud_d2), heel))
        # D1: radice crocetta bassa -> landa (basse)
        tr.append(_line3d([r1[0], land[0]], [r1[1], land[1]], [r1[2], land[2]],
                          "#2a7da5", w(shape.shroud_d1), heel))
    return tr


SAIL_COLORSCALE = [[0.0, STATE_COLORS[-1]], [1 / 3, STATE_COLORS[-1]],
                   [1 / 3, STATE_COLORS[0]], [2 / 3, STATE_COLORS[0]],
                   [2 / 3, STATE_COLORS[1]], [1.0, STATE_COLORS[1]]]


def sail_traces(shape: SailShape, sm, sj, heel):
    tr = []
    for which, strips in [("jib", sj), ("main", sm)]:
        X, Y, Z, st_h = sail_surface(which, shape, strips)
        sc = np.repeat(st_h[:, None].astype(float), X.shape[1], axis=1)
        Yr, Zr = _rot_heel(Y, Z, heel)
        tr.append(go.Surface(x=X, y=Yr, z=Zr, surfacecolor=sc,
                             colorscale=SAIL_COLORSCALE, cmin=-1, cmax=1,
                             showscale=False, opacity=SAIL_OPACITY,
                             lighting=dict(ambient=0.7, diffuse=0.5),
                             hoverinfo="skip"))
        # filetti
        n_h = X.shape[0]
        for k in range(N_STRIPS):
            j = int((k + 0.5) / N_STRIPS * (n_h - 1))
            state = int(strips["state"][k])
            col = STATE_COLORS[state]
            x0, y0, z0 = X[j, -1], Y[j, -1], Z[j, -1]
            d = (0.3, 0.4, -0.4) if state == 1 else (-0.65, 0.12, 0.0)
            tr.append(_line3d([x0, x0 + d[0]], [y0, y0 + d[1]], [z0, z0 + d[2]],
                              col, 5, heel))
            if which == "jib":
                xi, yi, zi = X[j, 2], Y[j, 2] - 0.05, Z[j, 2]
                d = (-0.18, -0.3, 0.4) if state == -1 else (-0.42, -0.06, 0.0)
                tr.append(_line3d([xi, xi + d[0]], [yi, yi + d[1]], [zi, zi + d[2]],
                                  STATE_COLORS[-1 if state == -1 else 0], 4, heel))
    return tr


def digital_twin_figure(shape: SailShape, sm, sj, heel):
    fig = go.Figure()
    for t in boat_traces(heel) + rig_traces(shape, heel) + sail_traces(shape, sm, sj, heel):
        fig.add_trace(t)
    # piano del mare
    xs = np.linspace(-11, 11, 2)
    Xw, Yw = np.meshgrid(xs, xs)
    fig.add_trace(go.Surface(x=Xw, y=Yw, z=np.zeros_like(Xw),
                             surfacecolor=np.zeros_like(Xw),
                             colorscale=[[0, "#7fb3d5"], [1, "#7fb3d5"]],
                             showscale=False, opacity=0.30, hoverinfo="skip"))
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data",
                   camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.55)),
                   bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=28, b=0), height=560, showlegend=False,
        title=dict(text=f"{BOAT_NAME} — sbandamento {heel:.0f}° · "
                        "🟢 flusso attaccato · 🔴 rifiuto · 🟠 stallo",
                   font=dict(size=13)))
    return fig


# ----------------------------------------------------------------------
# Interfaccia Streamlit
# ----------------------------------------------------------------------
st.set_page_config(page_title="Swan 45 — Regolazione vele", layout="wide")

st.title("⛵ Swan 45 — Simulatore di regolazione vele")
st.caption("Strip theory quasi-3D (vento apparente locale per ogni fascia d'altezza) "
           "+ flusso potenziale 2D. Didattico: mostra le *tendenze*, non è una CFD RANS.")

# Registro dei comandi: (chiave, min, max, default, step, unità, help)
CONTROLS = {
    "🌬️ Vento reale TWS": ("tws_kn", 4.0, 30.0, 14.0, 0.5, "kn", None),
    "🧭 Angolo al vento TWA": ("twa", 30.0, 175.0, 45.0, 1.0, "°", None),
    "— Scotta randa": ("main_sheet", 0.0, 1.0, 0.75, 0.05, "",
                       "1 = cazzata a ferro (chiude la balumina)"),
    "— Carrello randa": ("traveller", 0.0, 1.0, 0.50, 0.05, "",
                         "0 = tutto sottovento, 1 = sopravvento"),
    "— Vang": ("vang", 0.0, 1.0, 0.4, 0.05, "", None),
    "— Tesa base": ("outhaul", 0.0, 1.0, 0.6, 0.05, "", None),
    "— Cunningham": ("cunningham", 0.0, 1.0, 0.2, 0.05, "", None),
    "△ Scotta fiocco": ("jib_sheet", 0.0, 1.0, 0.75, 0.05, "", None),
    "△ Carrello fiocco": ("jib_car", 0.0, 1.0, 0.45, 0.05, "",
                          "0 = avanti (chiude la balumina), 1 = indietro (twist)"),
    "△ Drizza fiocco": ("jib_halyard", 0.0, 1.0, 0.6, 0.05, "", None),
    "⚙️ Martinetto (spessori)": ("jack_mm", 0.0, 30.0, 12.0, 2.0, "mm",
                                 "Pre-tensiona tutto il sartiame, colpi da 2 mm"),
    "⚙️ Paterazzo": ("backstay", 0.0, 1.0, 0.5, 0.05, "",
                     "Flette l'albero e tesa lo strallo"),
    "⚙️ Sartie alte/esterne (V1)": ("shroud_cap", 0.0, 1.0, 0.6, 0.05, "",
                                    "Lasche = la penna cade sottovento"),
    "⚙️ Diagonali (D2)": ("shroud_d2", 0.0, 1.0, 0.6, 0.05, "",
                          "Lasche = più prebend a metà albero"),
    "⚙️ Sartie basse (D1)": ("shroud_d1", 0.0, 1.0, 0.6, 0.05, "",
                             "Lasche = più prebend basso"),
    "⚙️ Rake albero": ("rake", 0.0, 1.0, 0.3, 0.05, "",
                       "Appoppamento: sposta il CE a poppa, aumenta l'orza"),
}

# valori persistenti tra un rerun e l'altro
for _, (key, _, _, default, _, _, _) in CONTROLS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Pannello di regolazione: una alla volta, slider grande in pagina ------
sel_col, sld_col = st.columns([1, 1.6])
with sel_col:
    chosen = st.selectbox("Regolazione", list(CONTROLS.keys()), index=2,
                          label_visibility="collapsed")
key, lo, hi, default, step, unit, hint = CONTROLS[chosen]
with sld_col:
    st.slider(chosen, lo, hi, step=step, key=key,
              help=hint, label_visibility="collapsed")
val = st.session_state[key]
st.caption(f"**{chosen.split(' ', 1)[-1]}: {val:g} {unit}**"
           + (f" — {hint}" if hint else ""))

with st.expander("📋 Tutte le regolazioni correnti"):
    riep = " · ".join(f"{n.split(' ', 1)[-1]} **{st.session_state[k]:g}{u}**"
                      for n, (k, _, _, _, _, u, _) in CONTROLS.items())
    st.markdown(riep)
    if st.button("↺ Ripristina regolazioni base"):
        for _, (k, _, _, d, _, _, _) in CONTROLS.items():
            st.session_state[k] = d
        st.rerun()

s = st.session_state
tws_kn, twa = s.tws_kn, s.twa
main_sheet, traveller, vang = s.main_sheet, s.traveller, s.vang
outhaul, cunningham, backstay = s.outhaul, s.cunningham, s.backstay
jack_mm = s.jack_mm
jib_sheet, jib_car, jib_halyard = s.jib_sheet, s.jib_car, s.jib_halyard
shroud_cap, shroud_d2, shroud_d1, rake = s.shroud_cap, s.shroud_d2, s.shroud_d1, s.rake

trim = Trim(main_sheet, traveller, vang, outhaul, cunningham, backstay,
            float(jack_mm), jib_sheet, jib_car, jib_halyard,
            shroud_cap, shroud_d2, shroud_d1, rake)

res = solve_equilibrium(tws_kn, twa, trim)
shape, diag = res["shape"], res["diag"]
sm, sj = diag["strips_main"], diag["strips_jib"]

# --- Metriche -------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Velocità barca", f"{res['v_kn']:.2f} kn")
c2.metric("VMG", f"{res['vmg']:.2f} kn")
c3.metric("Sbandamento", f"{res['heel']:.1f}°")
c4, c5, c6 = st.columns(3)
c4.metric("Timone (orza)", f"{res['helm']:.1f}°")
c5.metric("Vento apparente", f"{res['aws_kn']:.1f} kn")
c6.metric("AWA", f"{res['awa']:.0f}°")

warn = []
if diag["head_luff_main"]:
    warn.append("**Testa della randa in rifiuto** — in alto l'apparente è più largo: "
                "riduci il twist (vang/scotta) oppure poggia leggermente")
if diag["head_stall_main"]:
    warn.append("**Testa della randa stallata** — balumina troppo chiusa in alto: "
                "apri con twist (lasca vang o scotta, o più paterazzo)")
if diag["head_luff_jib"]:
    warn.append("**Testa del fiocco in rifiuto** — carrello troppo indietro o scotta troppo lasca in alto")
if diag["stalled_main"]:
    warn.append("**Randa in stallo diffuso** — lasca scotta/carrello o smagrisci col paterazzo")
if diag["stalled_jib"]:
    warn.append("**Fiocco in stallo diffuso** — lasca la scotta o arretra il carrello")
if res["heel"] > 25:
    warn.append(f"**Sbandamento eccessivo ({res['heel']:.0f}°)** — depotenzia: "
                "paterazzo, carrello sottovento, più twist")
if abs(res["helm"]) > 7:
    warn.append("**Troppa orza** — riduci il rake o scarica la randa")
if shape.shroud_cap < 0.40 and tws_kn > 14:
    warn.append("**Sartie alte lasche con questo vento** — la testa d'albero cade "
                "sottovento: perdi potenza in penna e lo strallo si allenta "
                "(fiocco più grasso proprio quando servirebbe magro)")
if trim.jack_mm <= 6 and tws_kn > 14:
    warn.append(f"**Rig molle ({trim.jack_mm:.0f} mm di martinetto) con "
                f"{tws_kn:.0f} kn** — le sartie sottovento vanno in bando, il rig "
                "si muove e lo strallo cade: aggiungi spessori (colpi da 2 mm)")
if trim.jack_mm >= 24 and tws_kn < 8:
    warn.append(f"**Rig molto carico ({trim.jack_mm:.0f} mm) in aria leggera** — "
                "strallo e vele troppo tesi/magri per questo vento: togli qualche "
                "spessore per dare potenza")
for w in warn:
    st.warning(w)

tab_3d, tab_flow, tab_shape, tab_forces, tab_polar = st.tabs(
    ["⛵ Vela 3D e filetti", "🌀 Flusso 2D", "📐 Forma vele",
     "⚖️ Forze", "🧭 Polare"])

# --- Vela 3D e filetti ------------------------------------------------------
with tab_3d:
    colL, colR = st.columns([1.4, 1])
    with colL:
        show_heel = st.toggle("Mostra lo sbandamento simulato", value=True)
        heel_view = res["heel"] if show_heel else 0.0
        st.plotly_chart(digital_twin_figure(shape, sm, sj, heel_view),
                        use_container_width=True,
                        config=dict(displayModeBar=False))
        st.caption("🖱️ Trascina per ruotare, pizzica per zoomare. "
                   "🎏 Filetti alla balumina e all'inferitura del fiocco: un filetto "
                   "che 'ricade' in penna = testa stallata; sopravvento che si alza "
                   "= rifiuto. Sartiame: **grigio = alte/esterne (V1)**, "
                   "**ocra = diagonali (D2)**, **azzurro = basse (D1)** — lo "
                   "spessore cresce con la tensione; con le alte lasche vedi la "
                   "testa d'albero cadere sottovento.")
    with colR:
        # incidenza locale vs altezza
        fig2, (axa, axw) = plt.subplots(2, 1, figsize=(5, 7), sharey=True,
                                        height_ratios=[1.3, 1])
        for s, name, color in [(sm, "Randa", "tab:blue"), (sj, "Fiocco", "tab:orange")]:
            axa.plot(s["alpha"], s["h"] * 100, "o-", color=color, label=name, ms=4)
            axa.plot(s["stall"], s["h"] * 100, "--", color=color, alpha=0.5, lw=1)
        axa.axvspan(-10, ALPHA_LUFF, color="red", alpha=0.08)
        axa.text(ALPHA_LUFF - 0.5, 55, "RIFIUTO", rotation=90, color="red",
                 fontsize=8, ha="right")
        axa.set_xlabel("incidenza locale [°]  (tratteggio = stallo)")
        axa.set_ylabel("altezza [%]")
        axa.set_title("Incidenza lungo l'altezza", fontsize=10)
        axa.legend(fontsize=8); axa.grid(alpha=0.3); axa.set_xlim(-6, 30)

        axw.plot(sm["awa"], sm["h"] * 100, "k-o", ms=4)
        axw.set_xlabel("AWA locale [°]")
        axw.set_title(f"Twist del vento apparente: "
                      f"+{diag['ideal_twist']:.1f}° dalla base alla penna", fontsize=10)
        axw.grid(alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2); plt.close(fig2)
        st.caption("In alto il vento reale è più forte → il vento apparente è più "
                   "**largo** in penna: qui l'apparente ruota di "
                   f"**+{diag['ideal_twist']:.1f}°** dalla base alla penna. La vela "
                   "deve avere *almeno* quel twist per non stallare in alto, più "
                   "qualche grado extra per scaricare la penna (meno sbandamento, "
                   "meno resistenza indotta). Guarda il grafico dell'incidenza: "
                   "l'ideale è una penna vicina allo stallo senza superarlo, e mai "
                   "sotto la banda del rifiuto.")

# --- Flusso 2D --------------------------------------------------------------
with tab_flow:
    st.caption("Sezione orizzontale a metà altezza, riferimento del vento apparente "
               "(flusso da sinistra). Flusso potenziale: lo stallo non è visualizzato, "
               "viene segnalato dai filetti e dai warning.")
    sails = build_panels(shape, res["awa"])
    vx, vy, gam, ds, sid = solve_vortices(sails)
    X, Y = np.meshgrid(np.linspace(-2.6, 2.2, 130), np.linspace(-1.8, 1.8, 100))
    U, V = velocity_field(X, Y, vx, vy, gam)
    speed = np.hypot(U, V)

    fig, (ax, axc) = plt.subplots(1, 2, figsize=(12, 4.6),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    pcm = ax.pcolormesh(X, Y, speed, cmap="coolwarm", shading="auto",
                        vmin=0.3, vmax=1.9)
    ax.streamplot(X, Y, U, V, density=1.5, color="k", linewidth=0.5, arrowsize=0.7)
    for xs, ys in sails:
        ax.plot(xs, ys, "k-", lw=3)
    ax.set_aspect("equal"); ax.set_title("Linee di flusso e |V|/V∞")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(pcm, ax=ax, shrink=0.8, label="|V| / V∞")

    for k, name, color in [(0, "Fiocco", "tab:orange"), (1, "Randa", "tab:blue")]:
        mask = sid == k
        s = np.cumsum(ds[mask]); s = s / s[-1]
        axc.plot(s, 2.0 * gam[mask] / ds[mask], color=color, label=name)
    axc.axhline(0, color="gray", lw=0.5)
    axc.set_xlabel("posizione lungo la corda")
    axc.set_ylabel("ΔCp (sottovento − sopravvento)")
    axc.set_title("Carico lungo la corda"); axc.legend()
    st.pyplot(fig); plt.close(fig)

# --- Forma vele -------------------------------------------------------------
with tab_shape:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
| Parametro | Randa | Fiocco |
|---|---|---|
| Grasso base → penna | {sm['camber'][0]*100:.1f} → {sm['camber'][-1]*100:.1f} % | {sj['camber'][0]*100:.1f} → {sj['camber'][-1]*100:.1f} % |
| Posizione grasso | {shape.main_draft*100:.0f} % | {shape.jib_draft*100:.0f} % |
| Angolo alla mezzeria | {shape.main_angle:.1f}° | {shape.jib_angle:.1f}° |
| Svergolamento (twist) | {shape.main_twist:.1f}° | {shape.jib_twist:.1f}° |
| Incidenza in penna | {sm['alpha'][-1]:.1f}° | {sj['alpha'][-1]:.1f}° |
| Incidenza alla base | {sm['alpha'][0]:.1f}° | {sj['alpha'][0]:.1f}° |

Catenaria strallo: **{shape.forestay_sag*100:.1f} %** della corda
&nbsp;·&nbsp; Rake: **{shape.rake_deg:.1f}°**
&nbsp;·&nbsp; Twist dell'apparente: **+{diag['ideal_twist']:.1f}°**

Martinetto: **{shape.jack_mm:.0f} mm** ({shape.jack_mm/2:.0f} spessori da 2 mm)
→ tensione base ×**{0.55+0.75*min(shape.jack_mm/30,1):.2f}**
&nbsp;·&nbsp; tensioni efficaci V1/D2/D1:
**{shape.shroud_cap:.2f} / {shape.shroud_d2:.2f} / {shape.shroud_d1:.2f}**
""")
    with col2:
        h = np.linspace(0, 1, 30)
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        ax2.plot(shape.main_angle + shape.main_twist * h ** 0.9, h * 100, label="Randa")
        ax2.plot(shape.jib_angle + shape.jib_twist * h ** 0.9, h * 100, label="Fiocco")
        ax2.plot(sm["awa"], sm["h"] * 100, "k--", lw=1, label="AWA locale")
        ax2.set_xlabel("angolo dalla mezzeria [°]")
        ax2.set_ylabel("altezza [% della vela]")
        ax2.set_title("Svergolamento vs twist dell'apparente")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        st.pyplot(fig2); plt.close(fig2)

# --- Forze ------------------------------------------------------------------
with tab_forces:
    beta = np.radians(res["awa"])
    lift, dragf = diag["lift"], diag["drag"]
    fx = lift * np.sin(beta) - dragf * np.cos(beta)
    fy = lift * np.cos(beta) + dragf * np.sin(beta)

    colA, colB = st.columns(2)
    with colA:
        fig3, ax3 = plt.subplots(figsize=(5, 3.6))
        ax3.bar(["Portanza", "Resistenza", "Spinta (Fx)", "Sbandante (Fy)"],
                [lift / 1000, dragf / 1000, fx / 1000, fy / 1000],
                color=["#2a9d8f", "#e76f51", "#264653", "#e9c46a"])
        ax3.set_ylabel("kN"); ax3.grid(axis="y", alpha=0.3)
        ax3.set_title("Forze aerodinamiche")
        st.pyplot(fig3); plt.close(fig3)

        # contributo di portanza per fascia d'altezza
        fig5, ax5 = plt.subplots(figsize=(5, 3.2))
        for s, name, color in [(sm, "Randa", "tab:blue"), (sj, "Fiocco", "tab:orange")]:
            ax5.plot(s["q"] * s["areas"] * s["cl"] / 1000, s["h"] * 100,
                     "o-", color=color, ms=4, label=name)
        ax5.set_xlabel("portanza per fascia [kN]"); ax5.set_ylabel("altezza [%]")
        ax5.set_title("Distribuzione verticale della portanza")
        ax5.legend(fontsize=8); ax5.grid(alpha=0.3)
        st.pyplot(fig5); plt.close(fig5)
    with colB:
        st.markdown(
            f"""
| Coefficiente | Valore |
|---|---|
| CL piano velico | **{diag['cl']:.2f}** |
| CD piano velico | **{diag['cd']:.3f}** |
| Efficienza L/D | **{diag['cl']/max(diag['cd'],1e-3):.1f}** |
| CL randa base → penna | {sm['cl'][0]:.2f} → {sm['cl'][-1]:.2f} |
| CL fiocco base → penna | {sj['cl'][0]:.2f} → {sj['cl'][-1]:.2f} |
| Pressione dinamica q·SA | {diag['q']/1000:.1f} kN |
""")
        st.caption("La distribuzione verticale mostra dove la vela lavora: una testa "
                   "in rifiuto o stallata 'buca' la curva in alto. L'obiettivo di un "
                   "buon twist è una distribuzione piena ma senza stallo in penna.")

# --- Polare -----------------------------------------------------------------
with tab_polar:
    st.caption("Velocità barca ai vari TWA **con la regolazione attuale** "
               "(vele non ottimizzate per ogni angolo).")

    @st.cache_data(show_spinner=False)
    def compute_polar(tws, trim_tuple):
        tr = Trim(*trim_tuple)
        angles = np.arange(35, 176, 5)
        return angles, [solve_equilibrium(tws, float(a), tr)["v_kn"] for a in angles]

    trim_tuple = (main_sheet, traveller, vang, outhaul, cunningham, backstay,
                  float(jack_mm), jib_sheet, jib_car, jib_halyard,
                  shroud_cap, shroud_d2, shroud_d1, rake)
    angles, speeds = compute_polar(tws_kn, trim_tuple)

    fig4 = plt.figure(figsize=(5.5, 5.5))
    ax4 = fig4.add_subplot(projection="polar")
    ax4.set_theta_zero_location("N"); ax4.set_theta_direction(-1)
    th = np.radians(angles)
    ax4.plot(th, speeds, "b-", lw=2)
    ax4.plot(-th, speeds, "b-", lw=2, alpha=0.4)
    ax4.plot(np.radians(twa), res["v_kn"], "ro", ms=9, label="condizione attuale")
    ax4.set_thetamin(-180); ax4.set_thetamax(180)
    ax4.set_title(f"Polare a TWS {tws_kn:.0f} kn"); ax4.legend(loc="lower left")
    st.pyplot(fig4); plt.close(fig4)

st.divider()
st.caption("⚠️ Modello semplificato a scopo didattico (strip theory + flusso "
           "potenziale + bilanci quasi-statici). Per analisi reali servono "
           "RANS CFD e un VPP calibrato.")
