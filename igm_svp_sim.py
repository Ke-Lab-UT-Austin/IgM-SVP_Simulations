#!/usr/bin/env python3
"""IgM-SVP Interaction Simulation — single-file PyQt5 GUI application."""

import sys
import os
import re
import math
import warnings
import threading
import multiprocessing as mp
import functools
import faulthandler
import traceback
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy import stats as scipy_stats
from scipy.interpolate import make_interp_spline
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QSplitter, QMessageBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QScrollArea, QRadioButton, QComboBox, QCheckBox,
    QDialog, QDialogButtonBox, QListWidget, QTextBrowser
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_OK = True
except ImportError:
    PYVISTA_OK = False
    warnings.warn(
        "pyvista / pyvistaqt not found; 3D viewport will use matplotlib fallback.",
        stacklevel=1,
    )

# ═══════════════════════════════════════════════════════════════════════════
# BIOLOGICAL CONSTANTS
#
# All measured from real physical dimensions (no fudge-factor scaling):
#   IgM max reach from center  = IGM_HUB_RADIUS + IGM_ARM_LENGTH + 2*IGM_FAB_RADIUS
#   SVP max reach from center  = svp_radius + SVP_SPIKE_DIAMETER
#   Binding (compute_binding)  = the two reaches, touching, from whole-particle centers
#
# These are session-wide pre-sets, editable at runtime via the "Geometry
# Parameters" dialog (GeometryParametersDialog) — Apply reassigns these
# module globals directly and clears the mesh caches so the next render/run
# picks them up.
# ═══════════════════════════════════════════════════════════════════════════
IGM_HUB_RADIUS    = 8.6    # nm — hub sphere radius
IGM_ARM_LENGTH    = 7.5    # nm — from hub SURFACE to arm-tip centerline
IGM_FAB_RADIUS    = 2.5    # nm — "blob" radius, sits fully beyond the arm tip
IGM_FAB_SPREAD    = 4.0    # nm — center-to-center spacing between an arm's two blobs
IGM_STERIC_RADIUS = IGM_HUB_RADIUS + IGM_ARM_LENGTH + 2 * IGM_FAB_RADIUS   # derived, 21.1 nm
N_ARMS            = 5
MAX_SVP_PER_IGM   = 10     # cap on distinct SVPs one whole IgM may bind (not per-arm)

SVP_SPIKE_DIAMETER = 2.5   # nm — fixed spike diameter, sits fully beyond the SVP surface

NN_INTERACTION_DISTANCE_NM = 33.1   # nm — fixed reference line on the NN plot; matches
                                    # the default contact-radius formula (IGM reach 21.1
                                    # + SVP radius 9.5 + spike 2.5) at default settings,
                                    # but does NOT recompute if those parameters change.

REAL_COLOR = 'red'      # shared "real data" color across the NN-plot and every
SIM_COLOR  = '#555555'  # real-vs-simulated bar chart (network/no-network, etc.)

_MAX_ATTEMPTS = 10_000

# Worker pools start from a clean forkserver process instead of forking the
# GUI process (which has Qt/X/OpenGL state and several threads — the pools
# are launched from a QThread) — forking a multithreaded GUI process is a
# documented source of intermittent hangs/crashes.
_MP_CTX = mp.get_context('forkserver')


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimParams:
    box_x:         float         = 200.0
    box_y:         float         = 200.0
    box_z:         float         = 200.0
    n_svp:         int           = 100
    n_igm:         int           = 20
    n_igm_type1:   int           = 7
    n_igm_type2:   int           = 7
    n_igm_type3:   int           = 6
    svp_diameter:  float         = 19.0
    svp_diameters: Optional[object] = None
    threshold:     float         = 0.0   # optional extra margin added on top of the physical contact distance
    border_conc:   float         = 1e-6
    border_thick:  float         = 50.0
    outer_box_enabled: bool      = False
    outer_box_x:   float         = 400.0
    outer_box_y:   float         = 400.0
    outer_box_z:   float         = 400.0
    outer_box_mode: str          = "center"   # "center" | "min_corner" | "max_corner" | "custom"
    outer_offset_x: float        = 0.0
    outer_offset_y: float        = 0.0
    outer_offset_z: float        = 0.0


def compute_outer_box_offset(params: 'SimParams'):
    """Return (off_x, off_y, off_z): the outer display box's minimum corner
    in the same world coordinates the inner simulation box occupies
    ([0,box_x] x [0,box_y] x [0,box_z]). Purely for rendering — never used
    by placement or binding."""
    Lx, Ly, Lz = params.box_x, params.box_y, params.box_z
    Ox, Oy, Oz = params.outer_box_x, params.outer_box_y, params.outer_box_z
    mode = params.outer_box_mode
    if mode == "center":
        return (-(Ox - Lx) / 2.0, -(Oy - Ly) / 2.0, -(Oz - Lz) / 2.0)
    elif mode == "min_corner":
        return (0.0, 0.0, 0.0)
    elif mode == "max_corner":
        return (Lx - Ox, Ly - Oy, Lz - Oz)
    elif mode == "custom":
        return (params.outer_offset_x, params.outer_offset_y, params.outer_offset_z)
    else:
        raise ValueError(f"Unknown outer_box_mode: {mode!r}")


def outer_box_contains_inner(params: 'SimParams') -> bool:
    """True iff the outer display box (given its computed offset) fully
    contains the inner simulation box on all 3 axes."""
    Lx, Ly, Lz = params.box_x, params.box_y, params.box_z
    Ox, Oy, Oz = params.outer_box_x, params.outer_box_y, params.outer_box_z
    off_x, off_y, off_z = compute_outer_box_offset(params)
    return (off_x <= 0 and off_x + Ox >= Lx and
            off_y <= 0 and off_y + Oy >= Ly and
            off_z <= 0 and off_z + Oz >= Lz)


def derive_dilution_params(orig: 'SimParams', k: int):
    """Derive the two dilution-comparison scenarios from an original
    (undiluted) SimParams and an integer dilution factor k (meaning 1/k):
      A — original volume, diluted concentration (particle counts divided
          by k, box unchanged, spread thin)
      B — same dilution, smaller volume (same ORIGINAL concentration
          recovered by shrinking the footprint) — only X/Y shrink, Z is
          held fixed (cryo-ET tomogram-thickness convention: the slab
          thickness doesn't change, only the lateral footprint does), shown
          inside a display box the size of the original box (reuses the
          outer-display-box feature)
    Both SVP and IgM counts are diluted proportionally (all 3 IgM subtypes).
    Every other parameter (SVP diameter, threshold, border conc/thickness,
    ...) is carried over from `orig` unchanged in both scenarios — dilution
    only ever changes particle counts and box X/Y."""
    n_svp_d = max(1, round(orig.n_svp / k))
    n1_d = max(0, round(orig.n_igm_type1 / k))
    n2_d = max(0, round(orig.n_igm_type2 / k))
    n_igm_d = max(1, round(orig.n_igm / k))
    n3_d = max(0, n_igm_d - n1_d - n2_d)

    params_a = dataclasses.replace(
        orig, n_svp=n_svp_d, n_igm=n1_d + n2_d + n3_d,
        n_igm_type1=n1_d, n_igm_type2=n2_d, n_igm_type3=n3_d,
        outer_box_enabled=False,   # left viewer: plain original box, no display-box overlay
    )

    # x*y*z_b == (x*y/k)*z_orig  =>  shrink only x/y, by k^-0.5 each, so their
    # product (and hence the volume, with z fixed) divides by exactly k.
    s_xy = k ** (-0.5)
    params_b = dataclasses.replace(
        params_a,
        box_x=orig.box_x * s_xy, box_y=orig.box_y * s_xy, box_z=orig.box_z,   # z unchanged
        outer_box_enabled=True,
        outer_box_x=orig.box_x, outer_box_y=orig.box_y, outer_box_z=orig.box_z,
        outer_box_mode="center",
    )
    return params_a, params_b


@dataclass
class SimResult:
    svp_positions:  object
    svp_radii:      object
    svp_is_border:  object
    igm_positions:  object
    igm_thetas:     object
    igm_bound_svps: list
    svp_bound_igms: list
    arm_svps:       list
    n_inner:        int = 0
    igm_subtypes:   object = None   # np.ndarray shape (n_igm,), values 0/1/2


# ═══════════════════════════════════════════════════════════════════════════
# IgM GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

# 5 of the 6 hexamer-spoke positions (60° spacing); one 120° gap where the
# missing 6th spike would be (gap between 240° and 360°/0°) — this gives the
# real pseudo-pentamer asymmetry rather than a symmetric pentagon.
_IGM_ARM_ANGLE_OFFSETS = np.array([0.0, 60.0, 120.0, 180.0, 240.0]) * np.pi / 180.0


def _igm_arm_angles(theta):
    return _IGM_ARM_ANGLE_OFFSETS + theta


def igm_arm_tips(center, theta):
    """Blob (Fab-lobe) centers. Each arm reaches from the hub center out to
    the tip centerline (hub radius + arm length), then the blob sits fully
    beyond that point — its near edge touching the tip, per the "blob
    protrudes fully past the reach" convention shared with SVP spikes."""
    arm_angles = _igm_arm_angles(theta)
    tips = []
    for i in range(N_ARMS):
        angle    = arm_angles[i]
        z_wb     = np.sin(i * 2.4) * 1.0
        raw      = np.array([np.cos(angle), np.sin(angle), z_wb])
        arm_dir  = raw / np.linalg.norm(raw)
        arm_tip  = np.asarray(center, dtype=float) + arm_dir * (IGM_HUB_RADIUS + IGM_ARM_LENGTH)
        blob_ctr = arm_tip + arm_dir * IGM_FAB_RADIUS
        perp_raw = np.array([-np.sin(angle), np.cos(angle), 0.0])
        perp_n   = np.linalg.norm(perp_raw)
        perp     = perp_raw / perp_n if perp_n > 1e-8 else perp_raw
        tips.append(blob_ctr + perp * (IGM_FAB_SPREAD / 2.0))
        tips.append(blob_ctr - perp * (IGM_FAB_SPREAD / 2.0))
    return np.array(tips)   # shape (10, 3)


# ═══════════════════════════════════════════════════════════════════════════
# CSV LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_svp_csv(path: str) -> np.ndarray:
    try:
        df = pd.read_csv(path)
        col_lower = {c.lower(): c for c in df.columns}
        for candidate in ('diameter', 'diameter_nm', 'size', 'size_nm'):
            if candidate in col_lower:
                return df[col_lower[candidate]].values.astype(float)
        for col in df.columns:
            try:
                return df[col].astype(float).values
            except (ValueError, TypeError):
                continue
        raise ValueError(f"No numeric diameter column found in '{path}'.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to load CSV '{path}': {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH (lowcc_IgM.ods) — curated per-IgM SVP contact counts for the
# 3 real tomograms (011/014/015). When the user loads one of these exact
# real tomogram STAR pairs, the "SVPs per IgM" histogram uses these curated
# counts instead of compute_binding()'s geometric estimate. Everything else
# (3D render, "IgMs per SVP" histogram) keeps using compute_binding() as
# before — there is no per-SVP ground truth in the .ods to substitute there.
# ═══════════════════════════════════════════════════════════════════════════

GROUND_TRUTH_ODS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'lowcc_IgM.ods')

_GROUND_TRUTH_CACHE: Optional[dict] = None


def _load_ground_truth_ods(path: str = GROUND_TRUTH_ODS_PATH) -> dict:
    """Parses lowcc_IgM.ods once and caches {'011': array, '014': array,
    '015': array} of contacted_SVPs counts, in IgM row order. Returns {} on
    any failure (missing file, missing odfpy, malformed sheet) so callers
    degrade gracefully to the geometric computation instead of crashing."""
    global _GROUND_TRUTH_CACHE
    if _GROUND_TRUTH_CACHE is not None:
        return _GROUND_TRUTH_CACHE
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine='odf')
        cache = {}
        for sheet_name, df in sheets.items():
            m = re.search(r'(\d+)', sheet_name)
            if not m:
                continue
            col = next((c for c in df.columns if 'contacted' in str(c).lower()), None)
            if col is None:
                continue
            cache[m.group(1)] = df[col].astype(int).values
        _GROUND_TRUTH_CACHE = cache
    except Exception:
        _GROUND_TRUTH_CACHE = {}
    return _GROUND_TRUTH_CACHE


def _tomogram_id_for_stars(igm_path: Optional[str], hbv_path: Optional[str]) -> Optional[str]:
    """Returns e.g. '011' when the IgM and SVP STAR filenames agree on the
    same tomogram id (per the data/ naming convention), else None."""
    if not igm_path or not hbv_path:
        return None
    m_igm = re.search(r'IgM_(\d+)_corrected', os.path.basename(igm_path))
    m_svp = re.search(r'SVP_picking_(\d+)', os.path.basename(hbv_path))
    if m_igm and m_svp and m_igm.group(1) == m_svp.group(1):
        return m_igm.group(1)
    return None


def _ground_truth_for_current_stars(igm_path: Optional[str], hbv_path: Optional[str],
                                     n_igm: int) -> Optional[np.ndarray]:
    """Ground-truth contacted_SVPs array for the currently loaded real STAR
    pair, or None if it isn't one of the 3 known tomograms, the .ods
    couldn't be read, or the row count doesn't match the loaded particles."""
    tomo_id = _tomogram_id_for_stars(igm_path, hbv_path)
    if tomo_id is None:
        return None
    ground_truth = _load_ground_truth_ods().get(tomo_id)
    if ground_truth is None or len(ground_truth) != n_igm:
        return None
    return ground_truth


# ═══════════════════════════════════════════════════════════════════════════
# STAR FILE PARSING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_star(path: str) -> dict:
    """
    Parse a RELION STAR file into {block_name: (columns, rows)} dict.
    Handles multi-block files (data_optics + data_particles).
    """
    result: dict = {}
    with open(path, 'r') as fh:
        lines = fh.read().splitlines()

    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        if stripped.startswith('data_'):
            block_name = stripped
            i += 1
            # advance to loop_
            while i < n and lines[i].strip() != 'loop_':
                i += 1
            i += 1  # skip loop_
            # read column definitions
            cols: List[str] = []
            while i < n and lines[i].strip().startswith('_'):
                cols.append(lines[i].strip().split()[0])
                i += 1
            # read data rows
            rows: List[List[str]] = []
            while i < n:
                s = lines[i].strip()
                if not s or s.startswith('data_') or s.startswith('#'):
                    break
                vals = s.split()
                if vals:
                    rows.append(vals)
                i += 1
            result[block_name] = (cols, rows)
        else:
            i += 1
    return result


def _star_df(parsed: dict, block: str) -> pd.DataFrame:
    """Convert a parsed STAR block to a DataFrame, filtering malformed rows."""
    if block not in parsed:
        return pd.DataFrame()
    cols, rows = parsed[block]
    good = [r for r in rows if len(r) == len(cols)]
    return pd.DataFrame(good, columns=cols)


def _find_particles_df(parsed: dict) -> pd.DataFrame:
    """
    Find the main particle table in a parsed STAR file.
    Prefers 'data_particles'; falls back to any non-optics block with rows.
    """
    for name in ('data_particles', 'data_'):
        df = _star_df(parsed, name)
        if not df.empty:
            return df
    # last resort: any block that isn't optics and has rows
    for name, (cols, rows) in parsed.items():
        if name == 'data_optics':
            continue
        df = _star_df(parsed, name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _find_optics_df(parsed: dict) -> pd.DataFrame:
    """Find the optics/metadata block (usually data_optics)."""
    for name in ('data_optics', 'data_general'):
        df = _star_df(parsed, name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _pixel_size_angst(optics: pd.DataFrame) -> Optional[float]:
    """
    Extract the effective (binned) image pixel size in Ångströms from an
    optics DataFrame.  Tries columns from multiple RELION versions.
    """
    for col in ('_rlnImagePixelSize',          # RELION 5 STA (binned)
                '_rlnTomoTiltSeriesPixelSize',  # raw tilt-series pixel size
                '_rlnPixelSize',               # RELION 3/4 generic
                '_rlnMicrographOriginalPixelSize'):
        if col in optics.columns:
            try:
                return float(optics[col].iloc[0])
            except (ValueError, IndexError):
                continue
    return None


def _image_size_px(optics: pd.DataFrame) -> Optional[float]:
    """Extract subtomogram box size in pixels from an optics DataFrame."""
    for col in ('_rlnImageSize',):
        if col in optics.columns:
            try:
                return float(optics[col].iloc[0])
            except (ValueError, IndexError):
                continue
    return None


def _extract_coords_nm(df: pd.DataFrame,
                        optics: pd.DataFrame) -> Tuple[Optional[pd.Series],
                                                        Optional[pd.Series],
                                                        Optional[pd.Series]]:
    """
    Return (xs, ys, zs) in nm from a particles DataFrame.

    Tries coordinate columns in this order:
      1. _rlnCenteredCoordinateX/Y/ZAngst  (RELION 5 STA, already in Å)
      2. _rlnCoordinateX/Y/ZAngst          (non-RELION / generic tools, already in Å,
                                             no origin-centering implied by the name)
      3. _rlnCoordinateX/Y/Z               (RELION 3/4, in pixels → needs px size)

    Returns (None, None, None) if no recognised columns are found.
    """
    # --- RELION 5 STA: coordinates in Ångströms, origin-centered ---
    angst_cols = ('_rlnCenteredCoordinateXAngst',
                  '_rlnCenteredCoordinateYAngst',
                  '_rlnCenteredCoordinateZAngst')
    if all(c in df.columns for c in angst_cols):
        xs = df[angst_cols[0]].astype(float) / 10.0
        ys = df[angst_cols[1]].astype(float) / 10.0
        zs = df[angst_cols[2]].astype(float) / 10.0
        return xs, ys, zs

    # --- Generic Å coordinates (e.g. written by non-RELION tools): same units
    # as the RELION 5 case above, just without "Centered" in the column name ---
    angst_plain_cols = ('_rlnCoordinateXAngst',
                        '_rlnCoordinateYAngst',
                        '_rlnCoordinateZAngst')
    if all(c in df.columns for c in angst_plain_cols):
        xs = df[angst_plain_cols[0]].astype(float) / 10.0
        ys = df[angst_plain_cols[1]].astype(float) / 10.0
        zs = df[angst_plain_cols[2]].astype(float) / 10.0
        return xs, ys, zs

    # --- RELION 3/4: coordinates in pixels ---
    px_cols = ('_rlnCoordinateX', '_rlnCoordinateY', '_rlnCoordinateZ')
    if all(c in df.columns for c in px_cols):
        px_angst = _pixel_size_angst(optics) if not optics.empty else None
        scale = (px_angst / 10.0) if px_angst else 1.0   # Å → nm; 1.0 if unknown
        xs = df[px_cols[0]].astype(float) * scale
        ys = df[px_cols[1]].astype(float) * scale
        zs = df[px_cols[2]].astype(float) * scale
        return xs, ys, zs

    return None, None, None


def extract_params_from_stars(hbv_path: str, igm_path: str) -> dict:
    """
    Derive simulation parameters from two RELION STAR files (SVP and IgM).

    Compatible with RELION 3, 4, and 5 subtomogram-averaging STAR files.
    Any parameter that cannot be determined from the files is returned as
    None so the caller can decide whether to skip filling that field.

    Returned keys
    -------------
    n_svp        : int   — particle count from SVP file
    n_igm        : int   — particle count from IgM file
    box_x/y/z    : float — bounding-box extent in nm (or None)
    svp_diameter : float — estimated SVP diameter in nm (or None)
    threshold    : float — suggested interaction threshold in nm (or None)
    warnings     : list  — human-readable strings for unresolved parameters
    """
    hbv_p = _parse_star(hbv_path)
    igm_p = _parse_star(igm_path)

    hbv_opt  = _find_optics_df(hbv_p)
    hbv_part = _find_particles_df(hbv_p)
    igm_opt  = _find_optics_df(igm_p)
    igm_part = _find_particles_df(igm_p)

    warns: List[str] = []

    if hbv_part.empty:
        raise ValueError(
            f"No particle table found in SVP file:\n{hbv_path}\n\n"
            "Expected a block named 'data_particles' with a loop_ section."
        )
    if igm_part.empty:
        raise ValueError(
            f"No particle table found in IgM file:\n{igm_path}\n\n"
            "Expected a block named 'data_particles' with a loop_ section."
        )

    n_svp = len(hbv_part)
    n_igm = len(igm_part)

    # --- Coordinates → box dimensions ---
    hbv_xs, hbv_ys, hbv_zs = _extract_coords_nm(hbv_part, hbv_opt)
    igm_xs, igm_ys, igm_zs = _extract_coords_nm(igm_part, igm_opt)

    box_x = box_y = box_z = None
    if hbv_xs is not None and igm_xs is not None:
        all_x = pd.concat([hbv_xs, igm_xs])
        all_y = pd.concat([hbv_ys, igm_ys])
        all_z = pd.concat([hbv_zs, igm_zs])
        box_x = round(float(all_x.max() - all_x.min()), 1)
        box_y = round(float(all_y.max() - all_y.min()), 1)
        box_z = round(float(all_z.max() - all_z.min()), 1)
    elif hbv_xs is not None:
        all_x, all_y, all_z = hbv_xs, hbv_ys, hbv_zs
        box_x = round(float(all_x.max() - all_x.min()), 1)
        box_y = round(float(all_y.max() - all_y.min()), 1)
        box_z = round(float(all_z.max() - all_z.min()), 1)
        warns.append("Box size estimated from SVP file only (no usable coordinates in IgM file).")
    elif igm_xs is not None:
        all_x, all_y, all_z = igm_xs, igm_ys, igm_zs
        box_x = round(float(all_x.max() - all_x.min()), 1)
        box_y = round(float(all_y.max() - all_y.min()), 1)
        box_z = round(float(all_z.max() - all_z.min()), 1)
        warns.append("Box size estimated from IgM file only (no usable coordinates in SVP file).")
    else:
        warns.append(
            "Could not determine box size: no recognised coordinate columns "
            "(_rlnCenteredCoordinateX/Y/ZAngst, _rlnCoordinateX/Y/ZAngst, "
            "or _rlnCoordinateX/Y/Z) found."
        )

    # --- SVP diameter from HBV optics block ---
    # Convention: subtomogram box ≈ 2 × particle diameter
    svp_diameter: Optional[float] = None
    px  = _pixel_size_angst(hbv_opt)
    sz  = _image_size_px(hbv_opt)
    if px is not None and sz is not None:
        svp_diameter = round(px * sz / 2.0 / 10.0, 1)   # Å → nm
    else:
        warns.append(
            "Could not estimate SVP diameter from optics block "
            "(_rlnImagePixelSize / _rlnImageSize not found). "
            "Please set it manually."
        )

    # Interaction threshold is now just an optional extra margin on top of the
    # physical contact distance (see compute_binding) — default it to 0 rather
    # than deriving it from SVP size.
    threshold: Optional[float] = 0.0

    # Border concentration = same density as inner volume (Option 1)
    border_conc: Optional[float] = None
    if box_x is not None and box_y is not None and box_z is not None:
        border_conc = n_svp / (box_x * box_y * box_z)   # SVPs/nm³
    else:
        warns.append("Border concentration not set (box dimensions unknown). Please set manually.")

    return {
        'n_svp':        n_svp,
        'n_igm':        n_igm,
        'box_x':        box_x,
        'box_y':        box_y,
        'box_z':        box_z,
        'svp_diameter': svp_diameter,
        'threshold':    threshold,
        'border_conc':  border_conc,
        'warnings':     warns,
    }


def extract_real_coords_from_stars(hbv_path: str, igm_path: str, svp_diameter: float) -> dict:
    """Extract the REAL per-particle (x,y,z) positions from two STAR files
    (not just derived summary statistics), origin-shifted so the combined
    bounding box starts at (0,0,0) — matching the simulator's [0,L]^3
    convention used everywhere else (place_particles_interleaved, the
    inner/outer display box, etc). Raises ValueError if coordinates can't
    be found in either file."""
    hbv_p = _parse_star(hbv_path)
    igm_p = _parse_star(igm_path)

    hbv_opt  = _find_optics_df(hbv_p)
    hbv_part = _find_particles_df(hbv_p)
    igm_opt  = _find_optics_df(igm_p)
    igm_part = _find_particles_df(igm_p)

    hbv_xs, hbv_ys, hbv_zs = _extract_coords_nm(hbv_part, hbv_opt)
    igm_xs, igm_ys, igm_zs = _extract_coords_nm(igm_part, igm_opt)
    if hbv_xs is None or igm_xs is None:
        raise ValueError(
            "Could not extract per-particle coordinates from one or both "
            "STAR files (no recognised coordinate columns found)."
        )

    all_x = pd.concat([hbv_xs, igm_xs])
    all_y = pd.concat([hbv_ys, igm_ys])
    all_z = pd.concat([hbv_zs, igm_zs])
    min_x, min_y, min_z = float(all_x.min()), float(all_y.min()), float(all_z.min())

    svp_pos = np.column_stack([
        (hbv_xs - min_x).to_numpy(), (hbv_ys - min_y).to_numpy(), (hbv_zs - min_z).to_numpy(),
    ])
    igm_pos = np.column_stack([
        (igm_xs - min_x).to_numpy(), (igm_ys - min_y).to_numpy(), (igm_zs - min_z).to_numpy(),
    ])

    box_x = float(all_x.max() - min_x)
    box_y = float(all_y.max() - min_y)
    box_z = float(all_z.max() - min_z)
    n_svp, n_igm = len(svp_pos), len(igm_pos)

    return dict(
        svp_positions = svp_pos,
        svp_radii     = np.full(n_svp, svp_diameter / 2.0),
        svp_is_border = np.zeros(n_svp, dtype=bool),   # real data: no border-shell concept
        igm_positions = igm_pos,
        igm_thetas    = np.random.default_rng().uniform(0.0, 2.0 * np.pi, n_igm),
        igm_subtypes  = np.zeros(n_igm, dtype=int),    # real STAR data: no subtype distinction
        box_x=box_x, box_y=box_y, box_z=box_z,
        n_svp=n_svp, n_igm=n_igm,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PLACEMENT — fixed coordinate convention:
#   Inner box : [0, L]^3
#   Total box : [-B, L+B]^3
# ═══════════════════════════════════════════════════════════════════════════

def _in_border_shell(pos: np.ndarray, B: float, Lx: float, Ly: float, Lz: float) -> bool:
    inside_total = bool(
        (pos[0] >= -B) and (pos[0] <= Lx + B) and
        (pos[1] >= -B) and (pos[1] <= Ly + B) and
        (pos[2] >= -B) and (pos[2] <= Lz + B)
    )
    inside_inner = bool(
        (pos[0] >= 0.0) and (pos[0] <= Lx) and
        (pos[1] >= 0.0) and (pos[1] <= Ly) and
        (pos[2] >= 0.0) and (pos[2] <= Lz)
    )
    return inside_total and not inside_inner


def place_svps_with_steric(params: SimParams):
    Lx, Ly, Lz = params.box_x, params.box_y, params.box_z
    B   = params.border_thick
    rng = np.random.default_rng()

    n_inner = params.n_svp
    if params.svp_diameters is not None and len(params.svp_diameters) > 0:
        pool        = np.asarray(params.svp_diameters, dtype=float)
        inner_diams = rng.choice(pool, size=n_inner, replace=True)
    else:
        inner_diams = np.full(n_inner, params.svp_diameter)
    inner_radii = inner_diams / 2.0

    V_border = (Lx + 2.0*B)*(Ly + 2.0*B)*(Lz + 2.0*B) - Lx*Ly*Lz
    n_border = round(params.border_conc * V_border)
    if n_border > 0:
        if params.svp_diameters is not None and len(params.svp_diameters) > 0:
            pool         = np.asarray(params.svp_diameters, dtype=float)
            border_diams = rng.choice(pool, size=n_border, replace=True)
        else:
            border_diams = np.full(n_border, params.svp_diameter)
        border_radii = border_diams / 2.0
    else:
        border_radii = np.array([], dtype=float)

    positions: List[np.ndarray] = []
    radii:     List[float]      = []
    is_border: List[bool]       = []

    # inner SVPs — placed first so indices 0…n_inner-1 are inner
    for k in range(n_inner):
        r_k    = float(inner_radii[k])
        placed = False
        for _ in range(_MAX_ATTEMPTS):
            pos = rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz])
            ok  = all(np.linalg.norm(pos - positions[j]) >=
                      (r_k + SVP_SPIKE_DIAMETER) + (radii[j] + SVP_SPIKE_DIAMETER)
                      for j in range(len(positions)))
            if ok:
                positions.append(pos); radii.append(r_k); is_border.append(False)
                placed = True
                break
        if not placed:
            pos = rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz])
            positions.append(pos); radii.append(r_k); is_border.append(False)

    # border SVPs — placed in [-B, Lx+B] x [-B, Ly+B] x [-B, Lz+B] \ inner box
    for k in range(n_border):
        r_k    = float(border_radii[k])
        placed = False
        for _ in range(_MAX_ATTEMPTS):
            pos = rng.uniform([-B, -B, -B], [Lx+B, Ly+B, Lz+B])
            if not _in_border_shell(pos, B, Lx, Ly, Lz):
                continue
            ok = all(np.linalg.norm(pos - positions[j]) >=
                     (r_k + SVP_SPIKE_DIAMETER) + (radii[j] + SVP_SPIKE_DIAMETER)
                     for j in range(len(positions)))
            if ok:
                positions.append(pos); radii.append(r_k); is_border.append(True)
                placed = True
                break
        if not placed:
            for _ in range(_MAX_ATTEMPTS):
                pos = rng.uniform([-B, -B, -B], [Lx+B, Ly+B, Lz+B])
                if _in_border_shell(pos, B, Lx, Ly, Lz):
                    positions.append(pos); radii.append(r_k); is_border.append(True)
                    break

    if not positions:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=bool)
    return (np.array(positions),
            np.array(radii, dtype=float),
            np.array(is_border, dtype=bool))


def place_igms_with_steric(params: SimParams,
                            svp_positions: np.ndarray,
                            svp_radii: np.ndarray):
    Lx, Ly, Lz = params.box_x, params.box_y, params.box_z
    n_igm = params.n_igm
    rng   = np.random.default_rng()

    igm_positions: List[np.ndarray] = []
    igm_thetas:    List[float]      = []
    have_svps = len(svp_positions) > 0

    for _ in range(n_igm):
        placed = False
        for _attempt in range(_MAX_ATTEMPTS):
            pos   = rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz])
            theta = rng.uniform(0.0, 2.0 * np.pi)

            ok = all(np.linalg.norm(pos - prev) >= 2.0 * IGM_STERIC_RADIUS
                     for prev in igm_positions)
            if not ok:
                continue

            if have_svps:
                ok = all(np.linalg.norm(pos - svp_positions[j]) >= IGM_HUB_RADIUS + svp_radii[j]
                         for j in range(len(svp_positions)))
            if ok:
                igm_positions.append(pos); igm_thetas.append(theta)
                placed = True
                break

        if not placed:
            igm_positions.append(rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz]))
            igm_thetas.append(rng.uniform(0.0, 2.0 * np.pi))

    if not igm_positions:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(igm_positions), np.array(igm_thetas)


def place_particles_interleaved(params: SimParams,
                                progress_cb=None, stop_event=None):
    """Place inner SVPs and IgMs in strict GCD ratio groups (r_svp SVPs then r_igm IgMs)."""
    Lx, Ly, Lz = params.box_x, params.box_y, params.box_z
    B           = params.border_thick
    rng         = np.random.default_rng()

    n_inner_svp = params.n_svp
    n_igm       = params.n_igm
    n1, n2, n3  = params.n_igm_type1, params.n_igm_type2, params.n_igm_type3

    if params.svp_diameters is not None and len(params.svp_diameters) > 0:
        pool        = np.asarray(params.svp_diameters, dtype=float)
        inner_diams = rng.choice(pool, size=n_inner_svp, replace=True)
    else:
        inner_diams = np.full(n_inner_svp, params.svp_diameter)
    inner_radii = inner_diams / 2.0

    svp_pos_list:   List[np.ndarray] = []
    svp_rad_list:   List[float]      = []
    igm_pos_list:   List[np.ndarray] = []
    igm_theta_list: List[float]      = []

    svp_done = 0
    igm_done = 0

    total_inner = n_inner_svp + n_igm
    emit_every  = max(1, total_inner // 40)

    def _stopped():
        return stop_event is not None and stop_event.is_set()

    # Fractional interleaving: at each step place whichever type is furthest
    # behind its target fraction.  Works correctly for every ratio including
    # co-prime inputs.  Guarantees the second particle is always from the
    # minority type (true simultaneous placement from step 2 onward).
    for _step in range(total_inner):
        if _stopped():
            return None
        if progress_cb is not None and _step % emit_every == 0:
            progress_cb(svp_done, n_inner_svp, igm_done, n_igm)

        svp_frac  = svp_done / n_inner_svp if n_inner_svp > 0 else 1.0
        igm_frac  = igm_done / n_igm       if n_igm       > 0 else 1.0
        place_svp = svp_done < n_inner_svp and (igm_done >= n_igm or svp_frac <= igm_frac)

        if place_svp:
            r_k    = float(inner_radii[svp_done])
            placed = False
            # Snapshot already-placed particles ONCE per particle (not once per
            # attempt — nothing here changes across attempts) so each attempt's
            # collision check is a single vectorized numpy call instead of a
            # Python-level per-pair loop.
            svp_pos_arr = np.asarray(svp_pos_list) if svp_pos_list else np.empty((0, 3))
            svp_rad_arr = np.asarray(svp_rad_list) if svp_rad_list else np.empty((0,))
            igm_pos_arr = np.asarray(igm_pos_list) if igm_pos_list else np.empty((0, 3))
            for _att in range(_MAX_ATTEMPTS):
                if _att % 500 == 499 and _stopped():
                    return None
                pos = rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz])
                if len(svp_pos_arr) and np.any(
                        np.linalg.norm(svp_pos_arr - pos, axis=1) <
                        (r_k + SVP_SPIKE_DIAMETER) + (svp_rad_arr + SVP_SPIKE_DIAMETER)):
                    continue
                if len(igm_pos_arr) and np.any(
                        np.linalg.norm(igm_pos_arr - pos, axis=1) < IGM_HUB_RADIUS + r_k):
                    continue
                svp_pos_list.append(pos); svp_rad_list.append(r_k)
                placed = True; break
            if not placed:
                svp_pos_list.append(rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz]))
                svp_rad_list.append(r_k)
            svp_done += 1

        else:
            placed = False
            svp_pos_arr = np.asarray(svp_pos_list) if svp_pos_list else np.empty((0, 3))
            svp_rad_arr = np.asarray(svp_rad_list) if svp_rad_list else np.empty((0,))
            igm_pos_arr = np.asarray(igm_pos_list) if igm_pos_list else np.empty((0, 3))
            for _att in range(_MAX_ATTEMPTS):
                if _att % 500 == 499 and _stopped():
                    return None
                pos   = rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz])
                theta = rng.uniform(0.0, 2.0 * np.pi)
                if len(igm_pos_arr) and np.any(
                        np.linalg.norm(igm_pos_arr - pos, axis=1) < 2.0 * IGM_STERIC_RADIUS):
                    continue
                if len(svp_pos_arr) and np.any(
                        np.linalg.norm(svp_pos_arr - pos, axis=1) < IGM_HUB_RADIUS + svp_rad_arr):
                    continue
                igm_pos_list.append(pos); igm_theta_list.append(theta)
                placed = True; break
            if not placed:
                igm_pos_list.append(rng.uniform([0.0, 0.0, 0.0], [Lx, Ly, Lz]))
                igm_theta_list.append(rng.uniform(0.0, 2.0 * np.pi))
            igm_done += 1

    # ── Border SVPs ───────────────────────────────────────────────────────
    V_border = (Lx + 2*B)*(Ly + 2*B)*(Lz + 2*B) - Lx*Ly*Lz
    n_border  = round(params.border_conc * V_border)
    if n_border > 0:
        if params.svp_diameters is not None and len(params.svp_diameters) > 0:
            pool         = np.asarray(params.svp_diameters, dtype=float)
            border_diams = rng.choice(pool, size=n_border, replace=True)
        else:
            border_diams = np.full(n_border, params.svp_diameter)
        border_radii = border_diams / 2.0

        b_positions: List[np.ndarray] = list(svp_pos_list)
        b_radii:     List[float]      = list(svp_rad_list)
        b_is_border: List[bool]       = [False] * len(svp_pos_list)

        for k in range(n_border):
            if _stopped():
                return None
            r_k    = float(border_radii[k])
            placed = False
            # Snapshot ONCE per particle — b_positions/b_radii grow across the
            # k loop (each new border particle must see everyone placed so
            # far), but don't change across attempts for THIS particle.
            b_pos_arr = np.asarray(b_positions) if b_positions else np.empty((0, 3))
            b_rad_arr = np.asarray(b_radii) if b_radii else np.empty((0,))
            for _att in range(_MAX_ATTEMPTS):
                if _att % 500 == 499 and _stopped():
                    return None
                pos = rng.uniform([-B, -B, -B], [Lx+B, Ly+B, Lz+B])
                if not _in_border_shell(pos, B, Lx, Ly, Lz):
                    continue
                if len(b_pos_arr) and np.any(
                        np.linalg.norm(b_pos_arr - pos, axis=1) <
                        (r_k + SVP_SPIKE_DIAMETER) + (b_rad_arr + SVP_SPIKE_DIAMETER)):
                    continue
                b_positions.append(pos); b_radii.append(r_k); b_is_border.append(True)
                placed = True; break
            if not placed:
                for _ in range(_MAX_ATTEMPTS):
                    pos = rng.uniform([-B, -B, -B], [Lx+B, Ly+B, Lz+B])
                    if _in_border_shell(pos, B, Lx, Ly, Lz):
                        b_positions.append(pos); b_radii.append(r_k); b_is_border.append(True)
                        break

        all_svp_pos    = np.array(b_positions)   if b_positions else np.zeros((0, 3))
        all_svp_rad    = np.array(b_radii, dtype=float) if b_radii else np.zeros(0)
        all_svp_border = np.array(b_is_border, dtype=bool) if b_is_border else np.zeros(0, dtype=bool)
    else:
        if svp_pos_list:
            all_svp_pos    = np.array(svp_pos_list)
            all_svp_rad    = np.array(svp_rad_list, dtype=float)
            all_svp_border = np.zeros(len(svp_pos_list), dtype=bool)
        else:
            all_svp_pos    = np.zeros((0, 3))
            all_svp_rad    = np.zeros(0)
            all_svp_border = np.zeros(0, dtype=bool)

    if igm_pos_list:
        igm_positions = np.array(igm_pos_list)
        igm_thetas    = np.array(igm_theta_list)
    else:
        igm_positions = np.zeros((0, 3))
        igm_thetas    = np.zeros(0)

    # Subtype assignment: first n1 → 0, next n2 → 1, last n3 → 2
    igm_subtypes = np.array([0]*n1 + [1]*n2 + [2]*n3, dtype=int)
    placed_count = len(igm_pos_list)
    if len(igm_subtypes) > placed_count:
        igm_subtypes = igm_subtypes[:placed_count]
    elif len(igm_subtypes) < placed_count:
        igm_subtypes = np.pad(igm_subtypes, (0, placed_count - len(igm_subtypes)),
                              constant_values=2)

    return (all_svp_pos, all_svp_rad, all_svp_border,
            igm_positions, igm_thetas, igm_subtypes)


# ═══════════════════════════════════════════════════════════════════════════
# BINDING — whole-molecule contact test: two "maximal reach" spheres (one
# centered on the IgM, one on the SVP) touching. MAX_SVP_PER_IGM enforced
# per whole IgM (not per arm). arm_svps is populated afterward, purely for
# visualization (nearest-arm attribution), and never gates or caps binding.
# ═══════════════════════════════════════════════════════════════════════════

def compute_binding(svp_positions, svp_radii, svp_is_border,
                    igm_positions, igm_thetas, igm_subtypes,
                    params: SimParams) -> SimResult:
    n_igm   = len(igm_positions)
    n_svp   = len(svp_positions)
    n_inner = int(np.sum(~svp_is_border))

    if n_svp == 0 or n_igm == 0:
        return SimResult(
            svp_positions  = svp_positions,
            svp_radii      = svp_radii,
            svp_is_border  = svp_is_border,
            igm_positions  = igm_positions,
            igm_thetas     = igm_thetas,
            igm_bound_svps = [set() for _ in range(n_igm)],
            svp_bound_igms = [set() for _ in range(n_svp)],
            arm_svps       = [[set() for _ in range(N_ARMS)] for _ in range(n_igm)],
            n_inner        = n_inner,
            igm_subtypes   = igm_subtypes,
        )

    tree           = KDTree(svp_positions)
    arm_svps       = [[set() for _ in range(N_ARMS)] for _ in range(n_igm)]
    igm_bound_svps = [set() for _ in range(n_igm)]
    svp_bound_igms = [set() for _ in range(n_svp)]

    # contact_r(j) = igm_reach + svp_reach(j) + threshold, where:
    #   igm_reach    = IGM_HUB_RADIUS + IGM_ARM_LENGTH + 2*IGM_FAB_RADIUS  (center -> blob outer tip)
    #   svp_reach(j) = svp_radii[j] + SVP_SPIKE_DIAMETER                  (center -> spike outer tip)
    # Over-query with max svp_radius, then filter exactly per candidate.
    igm_reach = IGM_HUB_RADIUS + IGM_ARM_LENGTH + 2 * IGM_FAB_RADIUS
    max_svp_r = float(np.max(svp_radii)) if n_svp > 0 else 0.0
    query_r   = igm_reach + max_svp_r + SVP_SPIKE_DIAMETER + params.threshold

    for i, igm_pos in enumerate(igm_positions):
        candidates = tree.query_ball_point(igm_pos, query_r)
        if not candidates:
            continue
        hits = []
        for svp_idx in candidates:
            contact_r = igm_reach + svp_radii[svp_idx] + SVP_SPIKE_DIAMETER + params.threshold
            d = float(np.linalg.norm(igm_pos - svp_positions[svp_idx]))
            if d <= contact_r:
                hits.append((d, svp_idx))
        if not hits:
            continue
        hits.sort(key=lambda t: t[0])
        kept = hits[:MAX_SVP_PER_IGM]
        for _, svp_idx in kept:
            igm_bound_svps[i].add(svp_idx)
            svp_bound_igms[svp_idx].add(i)

        # Visualization only: attribute each bound SVP to its geometrically
        # nearest arm (by Fab-lobe-midpoint distance) so existing per-arm
        # coloring in the 3D view keeps working. Not part of the binding test.
        tips    = igm_arm_tips(igm_pos, igm_thetas[i])   # (10, 3)
        arm_mid = [(tips[a * 2] + tips[a * 2 + 1]) / 2.0 for a in range(N_ARMS)]
        for _, svp_idx in kept:
            svp_c = svp_positions[svp_idx]
            nearest_arm = min(range(N_ARMS), key=lambda a: np.linalg.norm(arm_mid[a] - svp_c))
            arm_svps[i][nearest_arm].add(svp_idx)

    # Cosmetic-only reorientation for the 3D view: point arm 0 (whose
    # direction is exactly atan2(theta) — zero Z-wobble, see igm_arm_tips)
    # at each bound IgM's nearest bound SVP, so the render visibly shows an
    # arm reaching out and touching it. Does NOT affect which SVPs count as
    # bound — that was already decided above from whole-particle centers
    # alone. Copy (not mutate) igm_thetas: the input array is the same
    # object shared with the caller's saved pre-binding placement tuple.
    igm_thetas_render = np.array(igm_thetas, dtype=float, copy=True)
    for i in range(n_igm):
        if not igm_bound_svps[i]:
            continue
        nearest_j = min(igm_bound_svps[i],
                         key=lambda j: np.linalg.norm(svp_positions[j] - igm_positions[i]))
        d = svp_positions[nearest_j] - igm_positions[i]
        igm_thetas_render[i] = np.arctan2(d[1], d[0])

    return SimResult(
        svp_positions  = svp_positions,
        svp_radii      = svp_radii,
        svp_is_border  = svp_is_border,
        igm_positions  = igm_positions,
        igm_thetas     = igm_thetas_render,
        igm_bound_svps = igm_bound_svps,
        svp_bound_igms = svp_bound_igms,
        arm_svps       = arm_svps,
        n_inner        = n_inner,
        igm_subtypes   = igm_subtypes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# NEAREST-NEIGHBOR DISTANCE — for every inner SVP, distance to its nearest
# IgM. Used by the "Nearest Neighbour Analysis" mode (a distance-distribution
# comparison, real vs. simulated), independent of the binding/contact test.
# ═══════════════════════════════════════════════════════════════════════════

def nearest_neighbor_distances_and_indices(svp_positions_inner, igm_positions):
    """Like nearest_neighbor_distances, but also returns the index (into
    igm_positions) of each SVP's nearest IgM — used by NNViewerDialog to draw
    SVP -> nearest-IgM lines."""
    if len(svp_positions_inner) == 0 or len(igm_positions) == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    dists, idx = KDTree(igm_positions).query(svp_positions_inner, k=1)
    return np.asarray(dists, dtype=float), np.asarray(idx, dtype=int)


NN_DEFAULT_RADIUS_NM = 40.0   # default centre-centre interaction radius for the NN plot


def interaction_pairs(svp_positions, igm_positions, radius: float) -> dict:
    """SVP-IgM pairs counted as interactions in the Nearest Neighbour plot:
    a pair counts if ANY of (1) the IgM is that SVP's nearest IgM, (2) the
    SVP is that IgM's nearest SVP, (3) their centres are <= radius apart.
    Built as one boolean (n_svp, n_igm) mask, so a pair meeting several
    rules still appears exactly once. Returns a dict of equal-length arrays:
    svp_idx, igm_idx, dist, plus is_svp_nn / is_igm_nn / in_radius flags
    recording which rule(s) each pair met."""
    n_svp, n_igm = len(svp_positions), len(igm_positions)
    if n_svp == 0 or n_igm == 0:
        empty_i, empty_b = np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
        return dict(svp_idx=empty_i, igm_idx=empty_i, dist=np.zeros(0),
                    is_svp_nn=empty_b, is_igm_nn=empty_b, in_radius=empty_b)
    D = cdist(np.asarray(svp_positions, dtype=float), np.asarray(igm_positions, dtype=float))
    in_radius = D <= radius
    svp_nn = np.zeros_like(in_radius)
    svp_nn[np.arange(n_svp), D.argmin(axis=1)] = True
    igm_nn = np.zeros_like(in_radius)
    igm_nn[D.argmin(axis=0), np.arange(n_igm)] = True
    si, ii = np.nonzero(in_radius | svp_nn | igm_nn)
    return dict(svp_idx=si, igm_idx=ii, dist=D[si, ii],
                is_svp_nn=svp_nn[si, ii], is_igm_nn=igm_nn[si, ii], in_radius=in_radius[si, ii])


def run_nn_trial(params: SimParams, edges: np.ndarray, radius: float = NN_DEFAULT_RADIUS_NM):
    """One fresh random placement, histogrammed into raw counts of the
    interaction-pair distances (see interaction_pairs) over `edges`. Binned
    here, in the worker process, so only a small counts row travels back
    through the pool. Bare module-level function (mirrors
    run_bootstrap_trial_full above) so it can be dispatched via
    multiprocessing.Pool (with functools.partial binding edges/radius)."""
    svp_pos, svp_radii, svp_border, igm_pos, igm_thetas, igm_subtypes = \
        place_particles_interleaved(params)
    dists = interaction_pairs(svp_pos[~svp_border], igm_pos, radius)['dist']
    counts, _ = np.histogram(dists, bins=edges)
    result = compute_binding(svp_pos, svp_radii, svp_border, igm_pos,
                             igm_thetas, igm_subtypes, params)
    placement = (svp_pos, svp_radii, svp_border, igm_pos, igm_thetas)
    return counts, float(dists.sum()), placement, result


# ═══════════════════════════════════════════════════════════════════════════
# IgM MESH
# ═══════════════════════════════════════════════════════════════════════════

def _build_igm_mesh_raw(center, theta):
    """Actual mesh construction (expensive: 16 primitive shapes tessellated
    from scratch). Only ever called once, at the origin with theta=0, to
    build the cached template — see build_igm_mesh below."""
    center     = np.asarray(center, dtype=float)
    arm_angles = _igm_arm_angles(theta)
    tips       = igm_arm_tips(center, theta)
    parts      = []

    parts.append(pv.Sphere(radius=IGM_HUB_RADIUS, center=center,
                            theta_resolution=10, phi_resolution=10))
    for i in range(N_ARMS):
        angle   = arm_angles[i]
        z_wb    = np.sin(i * 2.4) * 1.0
        raw_dir = np.array([np.cos(angle), np.sin(angle), z_wb])
        arm_dir = raw_dir / np.linalg.norm(raw_dir)
        arm_tip   = center + arm_dir * (IGM_HUB_RADIUS + IGM_ARM_LENGTH)
        cyl_start = center + arm_dir * IGM_HUB_RADIUS
        cyl_mid   = (cyl_start + arm_tip) / 2.0
        parts.append(pv.Cylinder(center=cyl_mid, direction=arm_dir,
                                  radius=1.2, height=IGM_ARM_LENGTH,
                                  resolution=8, capping=True))
        parts.append(pv.Sphere(radius=IGM_FAB_RADIUS, center=tips[i * 2],
                                theta_resolution=8, phi_resolution=8))
        parts.append(pv.Sphere(radius=IGM_FAB_RADIUS, center=tips[i * 2 + 1],
                                theta_resolution=8, phi_resolution=8))

    return pv.MultiBlock(parts).combine()


_IGM_TEMPLATE_MESH = None


def _igm_template():
    """Lazily-built, cached IgM mesh at the origin with theta=0. Every IgM
    is geometrically identical up to a Z-rotation (by theta) + translation
    (by center) — see build_igm_mesh — so this expensive 16-primitive
    tessellation only ever needs to happen once, total, for the whole app
    session, instead of once per IgM particle."""
    global _IGM_TEMPLATE_MESH
    if _IGM_TEMPLATE_MESH is None:
        _IGM_TEMPLATE_MESH = _build_igm_mesh_raw(np.zeros(3), 0.0)
    return _IGM_TEMPLATE_MESH


def build_igm_mesh(center, theta):
    """Same rendered geometry as rebuilding from scratch (verified exactly,
    not approximately: the arm-direction vectors at angle+theta are exactly
    the Z-rotation-by-theta of the vectors at angle+0, since the per-arm
    Z-wobble depends only on arm index, never theta) — but built by
    transforming the cached template instead of re-tessellating 16
    primitive shapes every call."""
    mesh = _igm_template().rotate_z(np.degrees(theta), point=(0, 0, 0), inplace=False)
    return mesh.translate(np.asarray(center, dtype=float), inplace=False)


def _build_igm_hub_arms_raw(center, theta):
    """Same geometry as _build_igm_mesh_raw, but keeps the hub sphere and the
    arm+Fab-tip parts as two separate meshes instead of combining everything
    into one — needed to color the hub and arms differently for a bound IgM.
    Only ever called once (at the origin, theta=0) to build the cached
    templates — see build_igm_mesh_split below."""
    center     = np.asarray(center, dtype=float)
    arm_angles = _igm_arm_angles(theta)
    tips       = igm_arm_tips(center, theta)
    arm_parts  = []

    hub_mesh = pv.Sphere(radius=IGM_HUB_RADIUS, center=center,
                          theta_resolution=10, phi_resolution=10)
    for i in range(N_ARMS):
        angle   = arm_angles[i]
        z_wb    = np.sin(i * 2.4) * 1.0
        raw_dir = np.array([np.cos(angle), np.sin(angle), z_wb])
        arm_dir = raw_dir / np.linalg.norm(raw_dir)
        arm_tip   = center + arm_dir * (IGM_HUB_RADIUS + IGM_ARM_LENGTH)
        cyl_start = center + arm_dir * IGM_HUB_RADIUS
        cyl_mid   = (cyl_start + arm_tip) / 2.0
        arm_parts.append(pv.Cylinder(center=cyl_mid, direction=arm_dir,
                                     radius=1.2, height=IGM_ARM_LENGTH,
                                     resolution=8, capping=True))
        arm_parts.append(pv.Sphere(radius=IGM_FAB_RADIUS, center=tips[i * 2],
                                   theta_resolution=8, phi_resolution=8))
        arm_parts.append(pv.Sphere(radius=IGM_FAB_RADIUS, center=tips[i * 2 + 1],
                                   theta_resolution=8, phi_resolution=8))

    return hub_mesh, pv.MultiBlock(arm_parts).combine()


_IGM_HUB_TEMPLATE_MESH  = None
_IGM_ARMS_TEMPLATE_MESH = None


def _igm_hub_arms_template():
    """Lazily-built, cached (hub, arms) template pair at the origin with
    theta=0 — the split-mesh analogue of _igm_template()."""
    global _IGM_HUB_TEMPLATE_MESH, _IGM_ARMS_TEMPLATE_MESH
    if _IGM_HUB_TEMPLATE_MESH is None:
        _IGM_HUB_TEMPLATE_MESH, _IGM_ARMS_TEMPLATE_MESH = _build_igm_hub_arms_raw(np.zeros(3), 0.0)
    return _IGM_HUB_TEMPLATE_MESH, _IGM_ARMS_TEMPLATE_MESH


def build_igm_mesh_split(center, theta):
    """Split-mesh analogue of build_igm_mesh: returns (hub_mesh, arms_mesh)
    transformed from the cached templates, so a bound IgM can be rendered as
    two independently-colored actors (pale hub, darker arms)."""
    hub_t, arms_t = _igm_hub_arms_template()
    center = np.asarray(center, dtype=float)
    deg = np.degrees(theta)
    hub_mesh  = hub_t.rotate_z(deg, point=(0, 0, 0), inplace=False).translate(center, inplace=False)
    arms_mesh = arms_t.rotate_z(deg, point=(0, 0, 0), inplace=False).translate(center, inplace=False)
    return hub_mesh, arms_mesh


# ═══════════════════════════════════════════════════════════════════════════
# SVP MESH
# ═══════════════════════════════════════════════════════════════════════════

def _fibonacci_sphere(n: int) -> np.ndarray:
    golden = (1 + 5**0.5) / 2
    pts = []
    for i in range(n):
        theta = np.arccos(1 - 2*(i + 0.5) / n)
        phi   = 2 * np.pi * i / golden
        pts.append([np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(theta)])
    return np.array(pts)


_SVP_BLOB_DIRS = _fibonacci_sphere(24)


_SVP_TEMPLATE_CACHE: dict = {}


def _svp_template(radius: float):
    """Lazily-built, cached SVP mesh (at the origin) for a given radius.
    Every SVP of the same radius is geometrically identical up to
    translation (SVPs have no orientation parameter) — so this expensive
    25-primitive tessellation only needs to happen once per distinct radius
    for the whole app session, instead of once per SVP particle."""
    key = round(radius, 6)
    if key not in _SVP_TEMPLATE_CACHE:
        spike_r = SVP_SPIKE_DIAMETER / 2.0
        parts   = [pv.Sphere(radius=radius, center=(0.0, 0.0, 0.0),
                              theta_resolution=14, phi_resolution=14)]
        for d in _SVP_BLOB_DIRS:
            # Spike sits fully beyond the true surface, its near edge
            # touching it: center = radius + spike_r, outer tip = radius + SVP_SPIKE_DIAMETER.
            bc = d * (radius + spike_r)
            parts.append(pv.Sphere(radius=spike_r, center=bc,
                                   theta_resolution=8, phi_resolution=8))
        _SVP_TEMPLATE_CACHE[key] = pv.MultiBlock(parts).combine()
    return _SVP_TEMPLATE_CACHE[key]


def build_svp_mesh(center, radius: float):
    """Same rendered geometry as rebuilding from scratch, but built by
    translating the cached per-radius template instead of re-tessellating
    25 primitive shapes every call."""
    return _svp_template(radius).translate(np.asarray(center, dtype=float), inplace=False)


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY PARAMETERS DIALOG — interactive, live-diagram editor for the
# module-level biological constants above. Reachable from every mode (it's
# wired to a button on the always-visible button bar, not any one mode's
# panel). Apply reassigns those globals and clears the mesh caches; it never
# auto-triggers a simulation run — the user still clicks the mode's own Run
# button afterward, matching the rest of the app's strictly button-driven UX.
# ═══════════════════════════════════════════════════════════════════════════

_GEOM_DEFAULTS = dict(
    hub_radius=8.6, arm_length=7.5, blob_radius=2.5,
    svp_diameter=19.0, spike_diameter=2.5,
    max_per_igm=10, threshold=0.0,
)


class GeometryParametersDialog(QDialog):
    """Live cartoon-diagram editor for the IgM/SVP geometry pre-sets. Every
    spin box redraws both panels immediately; nothing is written back to the
    running simulation until Apply is clicked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Geometry Parameters")
        self.setMinimumSize(1050, 800)

        root = QVBoxLayout(self)

        # ── live cartoon diagrams ──────────────────────────────────────
        self._fig = Figure(figsize=(11, 5.5))
        self._ax_igm = self._fig.add_subplot(1, 2, 1)
        self._ax_svp = self._fig.add_subplot(1, 2, 2)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(440)
        root.addWidget(self._canvas)

        self._lbl_derived = QLabel()
        self._lbl_derived.setWordWrap(True)
        self._lbl_derived.setStyleSheet("font-weight:bold; font-size:13px; padding:4px;")
        root.addWidget(self._lbl_derived)

        # ── controls ────────────────────────────────────────────────────
        form = QFormLayout()

        def _mk_spin(lo, hi, step, val):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setValue(val)
            sb.valueChanged.connect(self._redraw)
            return sb

        self.sb_hub    = _mk_spin(0.5, 100.0, 0.1, IGM_HUB_RADIUS)
        self.sb_arm    = _mk_spin(0.5, 200.0, 0.1, IGM_ARM_LENGTH)
        self.sb_blob   = _mk_spin(0.1, 50.0,  0.1, IGM_FAB_RADIUS)
        self.sb_svp_d  = _mk_spin(1.0, 500.0, 0.5, _GEOM_DEFAULTS['svp_diameter'])
        self.sb_spike  = _mk_spin(0.1, 50.0,  0.1, SVP_SPIKE_DIAMETER)
        self.sb_cap    = QSpinBox()
        self.sb_cap.setRange(1, 100)
        self.sb_cap.setValue(MAX_SVP_PER_IGM)
        self.sb_cap.valueChanged.connect(self._redraw)
        self.sb_thresh = _mk_spin(0.0, 200.0, 0.5, _GEOM_DEFAULTS['threshold'])

        form.addRow("IgM hub radius (nm)", self.sb_hub)
        form.addRow("IgM arm length (nm) — hub surface → tip", self.sb_arm)
        form.addRow("IgM blob (Fab) radius (nm)", self.sb_blob)
        form.addRow("SVP diameter (nm)", self.sb_svp_d)
        form.addRow("SVP spike diameter (nm)", self.sb_spike)
        form.addRow("Max SVPs per IgM", self.sb_cap)
        form.addRow("Threshold — extra margin (nm)", self.sb_thresh)
        root.addLayout(form)

        # ── buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        root.addLayout(btn_row)

        self._redraw()

    # -- live diagram ----------------------------------------------------
    def _redraw(self):
        hub    = self.sb_hub.value()
        arm    = self.sb_arm.value()
        blob_r = self.sb_blob.value()
        svp_r  = self.sb_svp_d.value() / 2.0
        spike  = self.sb_spike.value()
        thresh = self.sb_thresh.value()

        igm_reach = hub + arm + 2 * blob_r
        svp_reach = svp_r + spike
        contact   = igm_reach + svp_reach + thresh

        # Shared scale between both panels so a viewer can trust relative
        # sizes at a glance — otherwise each panel would auto-fit to its own
        # content independently, making very different real sizes (e.g. a
        # small hub vs. a large SVP) look similarly sized side by side.
        lim = max(igm_reach, svp_reach) * 1.4

        ax = self._ax_igm
        ax.cla()
        ax.set_title("IgM (one arm shown)", fontsize=13)
        ax.add_patch(mpatches.Circle((0, 0), hub, facecolor='#dbe3ea',
                                      edgecolor='#2b3a4a', linewidth=2))
        # Hub radius tick + label.
        hub_ang = np.deg2rad(125)
        hub_tip = (hub * np.cos(hub_ang), hub * np.sin(hub_ang))
        ax.plot([0, hub_tip[0]], [0, hub_tip[1]], color='#2b3a4a', linewidth=1.5)
        ax.text(hub_tip[0] * 0.55, hub_tip[1] * 0.55 + lim * 0.03, f"hub r={hub:.1f} nm",
                ha='center', va='bottom', fontsize=11, color='#2b3a4a')

        ax.plot([hub, hub + arm], [0, 0], color='#2b3a4a', linewidth=3)
        ax.text((hub + hub + arm) / 2, lim * 0.04, f"arm={arm:.1f} nm",
                ha='center', va='bottom', fontsize=11, color='#2b3a4a')

        blob_cx = hub + arm + blob_r
        ax.add_patch(mpatches.Circle((blob_cx, 0), blob_r, facecolor='#dbe3ea',
                                      edgecolor='#2b3a4a', linewidth=2))
        ax.text(blob_cx, blob_r + lim * 0.05, f"Fab r={blob_r:.1f} nm",
                ha='center', va='bottom', fontsize=11, color='#2b3a4a')

        y0 = -lim * 0.55
        ax.annotate("", xy=(igm_reach, y0), xytext=(0, y0),
                    arrowprops=dict(arrowstyle='<->', color='#7c8894'))
        ax.text(igm_reach / 2, y0 - lim * 0.08, f"reach = {igm_reach:.2f} nm",
                ha='center', fontsize=11, color='#45525e')
        ax.set_xlim(-lim * 0.3, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.axis('off')

        ax = self._ax_svp
        ax.cla()
        ax.set_title("SVP", fontsize=13)
        ax.add_patch(mpatches.Circle((0, 0), svp_r, facecolor='#e3f1fd',
                                      edgecolor='#2196f3', linewidth=2))
        svp_ang = np.deg2rad(125)
        svp_tip = (svp_r * np.cos(svp_ang), svp_r * np.sin(svp_ang))
        ax.plot([0, svp_tip[0]], [0, svp_tip[1]], color='#1565C0', linewidth=1.5)
        ax.text(svp_tip[0] * 0.55, svp_tip[1] * 0.55 + lim * 0.03, f"SVP ⌀={2 * svp_r:.1f} nm",
                ha='center', va='bottom', fontsize=11, color='#1565C0')

        spike_r = spike / 2.0
        for deg in (0, 60, 120, 180, 240, 300):
            rad = np.deg2rad(deg)
            sc = ((svp_r + spike_r) * np.cos(rad), (svp_r + spike_r) * np.sin(rad))
            ax.add_patch(mpatches.Circle(sc, spike_r, facecolor='#2196f3',
                                          edgecolor='#2196f3', alpha=0.55))
        spike_label_pos = ((svp_r + spike_r) * np.cos(0), (svp_r + spike_r) * np.sin(0))
        ax.text(spike_label_pos[0], spike_label_pos[1] + spike_r + lim * 0.05,
                f"spike ⌀={spike:.1f} nm", ha='center', va='bottom', fontsize=11, color='#0D47A1')

        y1 = -lim * 0.7
        ax.annotate("", xy=(svp_reach, y1), xytext=(0, y1),
                    arrowprops=dict(arrowstyle='<->', color='#7c8894'))
        ax.text(svp_reach / 2, y1 - lim * 0.08, f"reach = {svp_reach:.2f} nm",
                ha='center', fontsize=11, color='#45525e')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.axis('off')

        self._fig.tight_layout()
        self._canvas.draw()

        self._lbl_derived.setText(
            f"IgM max reach: {igm_reach:.2f} nm   |   SVP max reach: {svp_reach:.2f} nm   |   "
            f"Bound if center-to-center ≤ {contact:.2f} nm "
            f"(= reaches + threshold {thresh:.2f} nm)"
        )

    def _reset_defaults(self):
        self.sb_hub.setValue(_GEOM_DEFAULTS['hub_radius'])
        self.sb_arm.setValue(_GEOM_DEFAULTS['arm_length'])
        self.sb_blob.setValue(_GEOM_DEFAULTS['blob_radius'])
        self.sb_svp_d.setValue(_GEOM_DEFAULTS['svp_diameter'])
        self.sb_spike.setValue(_GEOM_DEFAULTS['spike_diameter'])
        self.sb_cap.setValue(_GEOM_DEFAULTS['max_per_igm'])
        self.sb_thresh.setValue(_GEOM_DEFAULTS['threshold'])
        self._redraw()

    def _apply(self):
        global IGM_HUB_RADIUS, IGM_ARM_LENGTH, IGM_FAB_RADIUS, IGM_STERIC_RADIUS
        global SVP_SPIKE_DIAMETER, MAX_SVP_PER_IGM, _IGM_TEMPLATE_MESH

        IGM_HUB_RADIUS    = self.sb_hub.value()
        IGM_ARM_LENGTH    = self.sb_arm.value()
        IGM_FAB_RADIUS    = self.sb_blob.value()
        IGM_STERIC_RADIUS = IGM_HUB_RADIUS + IGM_ARM_LENGTH + 2 * IGM_FAB_RADIUS
        SVP_SPIKE_DIAMETER = self.sb_spike.value()
        MAX_SVP_PER_IGM     = self.sb_cap.value()

        # invalidate the lazy mesh caches so the next render/run picks up
        # the new geometry instead of stale cached shapes
        _IGM_TEMPLATE_MESH = None
        _SVP_TEMPLATE_CACHE.clear()

        # keep the main Parameters panel in sync (single source of truth
        # for svp_diameter / threshold stays the main panel's widgets)
        mw = self.parent()
        if mw is not None and hasattr(mw, 'sb_svp_diam'):
            mw.sb_svp_diam.setValue(self.sb_svp_d.value())
        if mw is not None and hasattr(mw, 'sb_threshold'):
            mw.sb_threshold.setValue(self.sb_thresh.value())

        self.accept()


# ═══════════════════════════════════════════════════════════════════════════
# NETWORK SUB-ANALYSIS POPUP — click-to-expand "(i)" detail for the
# SVPs-per-IgM comparison stats: "did a network form" (>=2 SVPs bound on an
# IgM), A vs B as two bars with a paper-style significance bracket.
# ═══════════════════════════════════════════════════════════════════════════

class NetworkInfoDialog(QDialog):
    """Two bars (% of IgMs in a network, A vs B) styled like
    MainWindow._plot_binned_overlap, with a significance bracket whose
    p-value the caller computes (the right test differs between
    real-vs-simulated and simulated-vs-simulated — see _open_h1_info_dialog)."""

    def __init__(self, parent, label_a: str, label_b: str,
                 avg_a: float, std_a: float, avg_b: float, std_b: float,
                 p: float, p_label: str, p_method: str, real_vs_sim: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Network Sub-Analysis (SVPs per IgM)")
        self.resize(480, 560)
        layout = QVBoxLayout(self)

        header = QLabel(
            "Network = IgM with ≥2 SVPs bound. Bars show the % of IgMs in a "
            "network; error bars are the trial-to-trial std.\n"
            f"Test: {p_method}")
        header.setWordWrap(True)
        layout.addWidget(header)

        fig = Figure(figsize=(4.5, 4.5), tight_layout=True)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        color_a, color_b = (REAL_COLOR, SIM_COLOR) if real_vs_sim else ('steelblue', 'darkorange')
        ax = fig.add_subplot(111)
        for x, avg, std, color in ((0, avg_a, std_a, color_a), (1, avg_b, std_b, color_b)):
            ax.bar(x, avg, 0.6, yerr=std, capsize=4,
                   facecolor=mcolors.to_rgba(color, 0.35), edgecolor=color, linewidth=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_a, label_b])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("% of IgMs in a network (≥2 SVPs bound)")
        ax.set_title("Network analysis", fontsize=10)
        top = max(avg_a + std_a, avg_b + std_b)
        draw_sig_bracket(ax, 0, 1, top * 1.08 if top > 0 else 1.0,
                         f"{p_to_stars(p)}  ({p_label})")

        canvas.draw()
        self._fig = fig

        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save Plot")
        btn_save.clicked.connect(self._save_plot)
        btn_row.addWidget(btn_save)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

    def _save_plot(self):
        _save_figure_dialog(self, self._fig, "network.svg", "Save Network Plot")


# ═══════════════════════════════════════════════════════════════════════════
# MEAN COMPARISON POPUP — Observed Data Comparison mode: mean, interval,
# effect size and p-value for observed vs simulated (SVPs per IgM and IgMs
# per SVP), plus a methods write-up of how both sides were derived.
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v: float, digits: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{digits}f}"


def _fmt_p(p: float) -> str:
    return "n/a" if p != p else f"{p:.3g}"


class MeanComparisonDialog(QDialog):
    """Read-only HTML report built from a RealDataCompareResult — see
    mean_comparison_stats for the statistics."""

    def __init__(self, parent, result: 'RealDataCompareResult'):
        super().__init__(parent)
        self.setWindowTitle("Mean Comparison — Observed vs Simulated")
        self.resize(820, 760)
        layout = QVBoxLayout(self)

        self._html = self._build_html(result)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._html)
        layout.addWidget(browser, stretch=1)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save as HTML")
        btn_save.clicked.connect(self._save_html)
        btn_row.addWidget(btn_save)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

    def _save_html(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mean Comparison", "mean_comparison.html", "HTML Files (*.html)")
        if path:
            if not path.lower().endswith(('.html', '.htm')):
                path += '.html'
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(self._html)

    @staticmethod
    def _metric_rows(title: str, unit: str, st: dict) -> str:
        return f"""
<h3>{title}</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th></th><th>Observed</th><th>Simulated</th></tr>
<tr><td>Mean ({unit})</td><td>{_fmt(st['obs_mean'])}</td><td>{_fmt(st['sim_mean'])}</td></tr>
<tr><td>95% interval</td>
    <td>{_fmt(st['obs_ci'][0])} – {_fmt(st['obs_ci'][1])}<br><small>bootstrap CI of the mean</small></td>
    <td>{_fmt(st['sim_range'][0])} – {_fmt(st['sim_range'][1])}<br><small>mean ± 1.96 SD of trial means</small></td></tr>
<tr><td>SD of trial means</td><td>—</td><td>{_fmt(st['sim_sd_means'])}</td></tr>
<tr><td>Per-particle SD</td><td>{_fmt(st['obs_sd'])}</td><td>{_fmt(st['sim_sd'])}</td></tr>
<tr><td>n</td><td>{st['obs_n']} particles</td>
    <td>{st['sim_n_trials']} trials ({st['sim_n_particles']} particles)</td></tr>
</table>
<p>Difference (observed − simulated): <b>{_fmt(st['diff'])}</b>
&nbsp;|&nbsp; Ratio (observed / simulated): <b>{_fmt(st['ratio'], 2)}×</b><br>
Effect size (Cohen's d): <b>{_fmt(st['d'], 2)}</b> ({st['d_label']})<br>
z = <b>{_fmt(st['z'], 2)}</b>, two-sided p = <b>{_fmt_p(st['p'])}</b>
&nbsp;<b>{p_to_stars(st['p'])}</b></p>
"""

    def _build_html(self, r: 'RealDataCompareResult') -> str:
        obs1 = r.obs_values1 if r.obs_values1 is not None else np.array([r.real_mean1])
        obs2 = r.obs_values2 if r.obs_values2 is not None else np.array([r.real_mean2])
        st1 = mean_comparison_stats(obs1, r.means1_a, r.pooled_counts1_a, r.z1, r.zpvalue1)
        st2 = mean_comparison_stats(obs2, r.means2_a, r.pooled_counts2_a, r.z2, r.zpvalue2)

        tomos = r.per_tomogram or []
        tomo_rows = "".join(
            f"<tr><td>{t['name']}</td>"
            f"<td>{t['real_params'].box_x:.0f} × {t['real_params'].box_y:.0f} × "
            f"{t['real_params'].box_z:.0f}</td>"
            f"<td>{t['real_params'].n_svp}</td><td>{t['real_params'].n_igm}</td>"
            f"<td>{t.get('data1_source', 'n/a')}</td>"
            f"<td>{_fmt(t.get('obs_mean1', float('nan')))} / {_fmt(t.get('sim_mean1', float('nan')))}</td>"
            f"<td>{_fmt(t.get('obs_mean2', float('nan')))} / {_fmt(t.get('sim_mean2', float('nan')))}</td></tr>"
            for t in tomos)
        p0 = tomos[0]['real_params'] if tomos else r.real_params
        n_tomo = len(tomos)
        border_note = ("forced off (\"No border shell on simulated side\" checked), so every "
                       "simulated SVP is counted, as in the observed data"
                       if r.force_no_border or not p0.border_conc else
                       f"on (border conc. {p0.border_conc:g}, thickness {p0.border_thick:g} nm): "
                       "border-shell SVPs are excluded from IgMs per SVP on the simulated side, "
                       "while every observed SVP is counted")

        return f"""
<html><body style="font-family: sans-serif; font-size: 10pt;">
<h2>Observed vs simulated means</h2>
<p>{n_tomo} observed tomogram(s): {r.n_svp_real} SVPs, {r.n_igm_real} IgMs in total.
{r.n_trials_a} simulated trials per tomogram.</p>
{self._metric_rows("SVPs per IgM", "SVPs bound per IgM", st1)}
{self._metric_rows("IgMs per SVP", "IgMs bound per SVP", st2)}

<h3>Per tomogram</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Tomogram</th><th>Box (nm)</th><th>SVPs</th><th>IgMs</th>
<th>SVPs-per-IgM source</th><th>SVPs/IgM obs / sim</th><th>IgMs/SVP obs / sim</th></tr>
{tomo_rows}
</table>

<h2>Methods</h2>
<h3>Observed tomograms</h3>
<ul>
<li>Each SVP and IgM STAR file is read for per-particle coordinates, converted to nm (from
Å coordinates, or pixel coordinates × the file's pixel size), and shifted so the combined particle cloud starts at (0, 0, 0). The
tomogram's box is the bounding box of all its particles. Observed data has no border shell:
every SVP is counted.</li>
<li><b>SVPs per IgM</b>: for the known tomograms (011/014/015), the curated per-IgM contact
counts from <i>lowcc_IgM.ods</i> are used. Otherwise it is the number of SVPs in geometric
contact with each IgM (SVP diameter {p0.svp_diameter:g} nm, interaction threshold
{p0.threshold:g} nm). The source for each tomogram is listed above.</li>
<li><b>IgMs per SVP</b>: number of IgMs in geometric contact with each SVP (same geometry).</li>
<li>Per-particle values are pooled across all tomograms. The observed mean is the mean over
every IgM (SVPs per IgM) or every SVP (IgMs per SVP).</li>
</ul>
<h3>Simulated data</h3>
<ul>
<li>For each observed tomogram, {r.n_trials_a} trials randomly place the same number of SVPs
and IgMs in a box of the same size, with the same SVP diameter and threshold. Binding is then
computed with the same geometric contact rule.</li>
<li>Border shell: {border_note}.</li>
<li>Tomograms are combined <b>by trial index</b>: trial <i>t</i> of every tomogram is summed into
one "combined experiment", so each of the {r.n_trials_a} combined trials matches the size of
the whole observed data set. Each combined trial gives one mean. The simulated mean is the
average of those {r.n_trials_a} trial means.</li>
</ul>
<h3>Statistics</h3>
<ul>
<li><b>Observed 95% CI</b>: percentile bootstrap of the mean, with 10,000 resamples of the pooled
particles (fixed seed). It treats particles as independent. Particles within one tomogram may
be correlated, so this interval may be too narrow.</li>
<li><b>Simulated 95% interval</b>: mean of the {r.n_trials_a} trial means ± 1.96 × their SD
(SD computed with n − 1). This is the range of means that random placement produces for an
experiment of this size, assuming the trial means are roughly normal. The z-test below uses the
same mean and SD, so the observed mean lies outside this interval exactly when p &lt; 0.05.</li>
<li><b>Per-particle SD</b>: observed, from the pooled per-particle values. Simulated, from all
simulated particles across all trials.</li>
<li><b>Effect size</b>: Cohen's d = (observed mean − simulated mean) /
√((SD<sub>obs</sub>² + SD<sub>sim</sub>²) / 2). |d| &lt; 0.2 negligible, &lt; 0.5 small,
&lt; 0.8 medium, otherwise large.</li>
<li><b>p-value</b>: z-test of the observed mean against the simulated trial means,
z = (observed mean − mean of trial means) / SD of trial means. The two-sided p comes from the
normal distribution. * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001, ns otherwise.</li>
</ul>
</body></html>
"""


class NNViewerDialog(QDialog):
    """Interactive 3D viewer for the real tomogram(s) loaded into Nearest
    Neighbour Analysis: every IgM/SVP labeled, a line for every counted
    interaction pair (see interaction_pairs), and each SVP's nearest-IgM
    distance — toggle between tomograms when more
    than one is loaded. Reuses MainWindow._update_3d_placement/_update_3d
    (via `mw`, the MainWindow instance) for the particle meshes/coloring, so
    this viewer always looks identical to the main viewports; lines/labels
    are added on top with pyvista's add_lines/add_point_labels. pyvista-only
    (the caller guards on PYVISTA_OK before opening this dialog)."""

    def __init__(self, mw, tomograms: list, radius: float = NN_DEFAULT_RADIUS_NM):
        super().__init__(mw)
        self._mw = mw
        self.tomograms = tomograms
        self.radius = radius   # interaction radius (nm) the lines are built with
        self.setWindowTitle("Nearest Neighbour 3D Viewer")
        self.resize(900, 750)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Tomogram:"))
        self.cmb_nn_viewer_tomo = QComboBox()
        self.cmb_nn_viewer_tomo.addItems([t['name'] for t in tomograms])
        self.cmb_nn_viewer_tomo.currentIndexChanged.connect(self._nn_viewer_render)
        controls.addWidget(self.cmb_nn_viewer_tomo, stretch=1)

        self.cb_nn_show_lines = QCheckBox("Lines")
        self.cb_nn_show_lines.setChecked(True)
        self.cb_nn_show_igm_labels = QCheckBox("IgM labels")
        self.cb_nn_show_igm_labels.setChecked(True)
        self.cb_nn_show_svp_labels = QCheckBox("SVP labels")
        self.cb_nn_show_svp_labels.setChecked(False)
        self.cb_nn_show_dist_labels = QCheckBox("Nearest-distance labels")
        self.cb_nn_show_dist_labels.setChecked(False)
        for cb in (self.cb_nn_show_lines, self.cb_nn_show_igm_labels,
                  self.cb_nn_show_svp_labels, self.cb_nn_show_dist_labels):
            cb.stateChanged.connect(self._nn_viewer_render)
            controls.addWidget(cb)
        controls.addWidget(QLabel(f"Interaction radius: {radius:g} nm"))
        layout.addLayout(controls)

        self.plotterN = QtInteractor(self)
        self.plotterN.set_background('white')
        layout.addWidget(self.plotterN, stretch=1)

        # Debounced rescale of the distance labels on zoom (camera changes
        # fire many times per scroll/drag).
        self._observed_camera = None
        self._cam_obs_tag = None
        self._closed = False
        self._dist_ref_height = 0.0
        self._dist_font_size = self._DIST_FONT_BASE
        self._dist_zoom_timer = QTimer(self)
        self._dist_zoom_timer.setSingleShot(True)
        self._dist_zoom_timer.setInterval(120)
        self._dist_zoom_timer.timeout.connect(self._nn_rescale_dist_labels)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save Screenshot")
        btn_save.clicked.connect(self._save_nn_viewer)
        btn_row.addWidget(btn_save)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        self._nn_viewer_render()

    def _remove_cam_observer(self):
        if self._observed_camera is not None and self._cam_obs_tag is not None:
            self._observed_camera.RemoveObserver(self._cam_obs_tag)
        self._observed_camera = self._cam_obs_tag = None

    def _shutdown(self):
        """Release the VTK render window. Without this, a closed viewer
        (kept alive by its Qt parent) leaves a live OpenGL window and camera
        observer behind — each reopen piles up another, a known source of
        XCB BadWindow errors and segfaults with pyvistaqt."""
        if self._closed:
            return
        self._closed = True
        self._dist_zoom_timer.stop()
        self._remove_cam_observer()
        self.plotterN.close()

    def done(self, r):
        self._shutdown()
        super().done(r)

    def closeEvent(self, event):
        self._shutdown()
        super().closeEvent(event)

    def _save_nn_viewer(self):
        if self._closed:
            return
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save NN 3D Viewer", "nn_viewer.svg",
            "SVG Files (*.svg);;PNG Files (*.png)")
        if not path:
            return
        ext = _pick_save_ext(path, chosen_filter)
        if not path.lower().endswith(f'.{ext}'):
            path = f"{path}.{ext}"
        if ext == 'svg':
            lines = None
            if self.cb_nn_show_lines.isChecked():
                points, segs = self._nn_pair_segments()
                if len(segs):
                    lines = (points, segs, '#555555', self._NN_LINE_SVG_ALPHA,
                             self._NN_LINE_SVG_WIDTH, 'nn_lines')
            _save_plotter_hires_svg(self.plotterN, path,
                                    labels=list(self._nn_label_sets().values()),
                                    lines=lines)
        else:
            self.plotterN.screenshot(path)

    def _nn_viewer_render(self):
        if self._closed:
            return
        tomo = self.tomograms[self.cmb_nn_viewer_tomo.currentIndex()]

        # Particle meshes + binding coloring: reuse the exact same rendering
        # path as the main viewports (dedicated actor-list attribute names
        # on the MainWindow instance so this doesn't collide with the
        # primary/secondary viewports' own actor lists). _update_3d_placement
        # already clears the plotter and rebuilds the box + every particle.
        self._mw._update_3d_placement(
            tomo['placement'], plotter=self.plotterN,
            svp_actors_attr='_nn_viewer_svp_actors', igm_actors_attr='_nn_viewer_igm_actors',
            params=tomo['params'])
        self._mw._update_3d(
            tomo['result'], plotter=self.plotterN,
            svp_actors_attr='_nn_viewer_svp_actors', igm_actors_attr='_nn_viewer_igm_actors')

        if self.cb_nn_show_lines.isChecked():
            points, segs = self._nn_pair_segments()
            if len(segs):
                actor = self.plotterN.add_lines(points[segs.ravel()], color='#666666',
                                                width=self._NN_LINE_WIDTH, name='nn_lines')
                actor.GetProperty().SetOpacity(self._NN_LINE_OPACITY)

        for name, (points, texts, color, font_px) in self._nn_label_sets().items():
            if name != 'nn_dist_labels':
                self._nn_add_labels(name, points, texts, color, font_px)

        # Home view (just reset by _update_3d_placement) is the 1x zoom
        # reference for scaling the distance labels.
        self._dist_ref_height = self._nn_visible_height()
        self._dist_font_size = self._DIST_FONT_BASE
        cam = self.plotterN.camera
        if cam is not self._observed_camera:
            self._remove_cam_observer()
            self._cam_obs_tag = cam.AddObserver(
                'ModifiedEvent', lambda *_: None if self._closed else self._dist_zoom_timer.start())
            self._observed_camera = cam
        self._nn_add_dist_labels()

    _DIST_FONT_BASE = 9
    _DIST_FONT_MAX_ZOOM = 5.0
    _NN_LINE_OPACITY = 0.45      # on-screen interaction lines
    _NN_LINE_WIDTH = 2
    _NN_LINE_SVG_ALPHA = 0.55    # SVG export (vector, drawn behind particles)
    _NN_LINE_SVG_WIDTH = 1.0     # screen px, scaled with the export

    def _nn_pair_segments(self):
        """(points, segments) for the counted interaction pairs of the
        current tomogram — the same pairs the plot counted (tomo['pairs'],
        from the last run; computed with this viewer's radius if there is no
        run yet): points = SVPs then IgMs stacked, segments = (K, 2) index
        pairs into points."""
        tomo = self.tomograms[self.cmb_nn_viewer_tomo.currentIndex()]
        pairs = tomo.get('pairs')
        if pairs is None:
            pairs = tomogram_interaction_pairs(tomo, self.radius)
        svp_positions = np.asarray(tomo['placement'][0], dtype=float).reshape(-1, 3)
        igm_positions = np.asarray(tomo['placement'][3], dtype=float).reshape(-1, 3)
        points = np.vstack([svp_positions, igm_positions])
        segs = np.column_stack([pairs['svp_idx'], len(svp_positions) + pairs['igm_idx']])
        return points, segs.reshape(-1, 2).astype(int)

    def _nn_visible_height(self) -> float:
        """World-space height of the view — shrinks as the user zooms in."""
        cam = self.plotterN.camera
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale()
        return 2.0 * cam.GetDistance() * math.tan(math.radians(cam.GetViewAngle()) / 2.0)

    def _nn_label_sets(self) -> dict:
        """Every label set currently switched on, as
        {actor name: (points, texts, color, font_px)} — shared by the
        on-screen render and the SVG export (which must redraw the labels
        itself; see _save_plotter_hires_svg)."""
        tomo = self.tomograms[self.cmb_nn_viewer_tomo.currentIndex()]
        svp_positions = tomo['placement'][0]
        igm_positions = tomo['placement'][3]
        dists, nn_idx = tomo['dists'], tomo['nn_idx']
        sets = {}
        if self.cb_nn_show_igm_labels.isChecked() and len(igm_positions):
            sets['nn_igm_labels'] = (igm_positions,
                                     [f"IgM {i}" for i in range(len(igm_positions))],
                                     'black', 12)
        if self.cb_nn_show_svp_labels.isChecked() and len(svp_positions):
            sets['nn_svp_labels'] = (svp_positions,
                                     [f"SVP {i}" for i in range(len(svp_positions))],
                                     '#1565C0', 10)
        if self.cb_nn_show_dist_labels.isChecked() and len(dists):
            midpoints = (svp_positions + igm_positions[nn_idx]) / 2.0
            sets['nn_dist_labels'] = (midpoints, [f"{d:.1f} nm" for d in dists],
                                      '#B71C1C', self._dist_font_size)
        return sets

    def _nn_add_labels(self, name, points, texts, color, font_px):
        self.plotterN.add_point_labels(
            points, texts, name=name, font_size=font_px, point_size=1,
            shape_opacity=0.6, text_color=color)

    def _nn_add_dist_labels(self):
        dist = self._nn_label_sets().get('nn_dist_labels')
        if dist is not None:
            self._nn_add_labels('nn_dist_labels', *dist)

    def _nn_rescale_dist_labels(self):
        """Point labels are fixed-pixel-size, so grow the distance labels
        in proportion to the zoom (up to _DIST_FONT_MAX_ZOOM x) — otherwise
        they stay tiny when zoomed in on a single particle."""
        if self._closed or not self.cb_nn_show_dist_labels.isChecked():
            return
        height = self._nn_visible_height()
        if height <= 0 or not self._dist_ref_height:
            return
        zoom = min(max(self._dist_ref_height / height, 1.0), self._DIST_FONT_MAX_ZOOM)
        font = int(round(self._DIST_FONT_BASE * zoom))
        if font != self._dist_font_size:
            self._dist_font_size = font
            self._nn_add_dist_labels()
            self.plotterN.render()


# ═══════════════════════════════════════════════════════════════════════════
# COLORING
# ═══════════════════════════════════════════════════════════════════════════

_SVP_COLOR_BOUND     = '#A4BBDE'   # light blue
_SVP_COLOR_UNBOUND   = '#B2B2B2'   # grey
_IGM_COLOR_UNBOUND   = '#B2B2B2'   # grey (whole IgM, no hub/arm split when unbound)
_IGM_HUB_COLOR_BOUND = '#DEB8C8'   # pale pink (hub/center)
_IGM_ARM_COLOR_BOUND = '#E88A9A'   # salmon-red (arms)


def svp_color(n_igms_bound: int) -> str:
    return _SVP_COLOR_BOUND if n_igms_bound > 0 else _SVP_COLOR_UNBOUND


def _hex_rgb(h: str):
    h = h.lstrip('#')
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255


# ═══════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════

class SimulationWorker(QThread):
    finished  = pyqtSignal(object)
    placed    = pyqtSignal(object)   # emitted after placement, before binding
    error     = pyqtSignal(str)
    progress  = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, params: SimParams):
        super().__init__()
        self.params      = params
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            def _progress_cb(svp_done, n_svp, igm_done, n_igm):
                ratio = f"{n_svp}:{n_igm}" if n_igm else str(n_svp)
                self.progress.emit(
                    f"Placing (ratio {ratio}) — "
                    f"SVPs: {svp_done}/{n_svp}  IgMs: {igm_done}/{n_igm}"
                )

            out = place_particles_interleaved(
                self.params,
                progress_cb=_progress_cb,
                stop_event=self._stop_event,
            )
            if out is None:
                self.cancelled.emit()
                return
            svp_pos, svp_radii, svp_border, igm_pos, igm_thetas, igm_subtypes = out
            # Show all particles placed (unbound colors) before computing binding
            self.placed.emit((svp_pos, svp_radii, svp_border, igm_pos, igm_thetas))
            self.progress.emit("Computing binding ...")
            result = compute_binding(svp_pos, svp_radii, svp_border,
                                     igm_pos, igm_thetas, igm_subtypes, self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP — repeated-random-placement null-model test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BootstrapResult:
    trial_means1: object   # np.ndarray (n_trials,) — mean SVPs bound per IgM, per trial (kept for grand-mean/CI title)
    trial_means2: object   # np.ndarray (n_trials,) — mean IgMs bound per SVP, per trial (kept for grand-mean/CI title)
    bin_avg1: object       # np.ndarray (11,) — avg # IgMs with exactly k SVPs bound, k=0..10, averaged across trials
    bin_std1: object       # np.ndarray (11,) — std of that count across trials (error bars)
    bin_avg2: object       # np.ndarray (max_k2+1,) — avg # SVPs with exactly k IgMs bound, averaged across trials
    bin_std2: object       # np.ndarray (max_k2+1,) — std of that count across trials (error bars)
    n_trials:     int = 0
    sample_trial_index: Optional[int] = None   # 1-based index of the trial shown below
    sample_placement:   object        = None   # (svp_pos, svp_radii, svp_border, igm_pos, igm_thetas) of that trial
    sample_result:      object        = None   # full SimResult of that trial (for bound/unbound coloring)


def run_bootstrap_trial_full(params: SimParams):
    """Run one fresh random placement + binding computation. Returns
    (mean1, std1, mean2, std2, placement, result, data1, data2) where
    placement is the (svp_pos, svp_radii, svp_border, igm_pos, igm_thetas)
    tuple, result is the full SimResult, and data1/data2 are the raw
    per-particle whole-number counts (data1[i] = SVPs bound to IgM i,
    data2[j] = IgMs bound to inner SVP j) — everything needed to both
    summarize the trial (scalar means) AND bin it into whole-number counts
    AND render it in the 3D view. Single source of truth for
    run_bootstrap_trial and _run_n_trials_collect."""
    svp_pos, svp_radii, svp_border, igm_pos, igm_thetas, igm_subtypes = \
        place_particles_interleaved(params)
    result = compute_binding(svp_pos, svp_radii, svp_border,
                             igm_pos, igm_thetas, igm_subtypes, params)
    data1 = [len(s) for s in result.igm_bound_svps]
    data2 = [len(result.svp_bound_igms[j]) for j in range(result.n_inner)]
    mean1 = float(np.mean(data1)) if data1 else 0.0
    std1  = float(np.std(data1))  if data1 else 0.0
    mean2 = float(np.mean(data2)) if data2 else 0.0
    std2  = float(np.std(data2))  if data2 else 0.0
    placement = (svp_pos, svp_radii, svp_border, igm_pos, igm_thetas)
    return mean1, std1, mean2, std2, placement, result, data1, data2


def run_bootstrap_trial(params: SimParams):
    """Run one fresh random placement + binding computation and return only
    the per-trial summary stats (mean1, std1, mean2, std2). A bare
    module-level function (no Qt dependency) so it can be exercised
    headless/standalone. Each call re-randomizes placement via
    place_particles_interleaved's own unseeded np.random.default_rng() — no
    extra seeding needed here."""
    mean1, std1, mean2, std2, _placement, _result, _data1, _data2 = run_bootstrap_trial_full(params)
    return mean1, std1, mean2, std2


def _run_n_trials_collect(params: SimParams, n_trials: int, stop_event: threading.Event,
                          progress_cb=None):
    """Run n_trials fresh placement+binding trials for one SimParams, using
    a multiprocessing.Pool since trials are fully independent (fresh
    unseeded RNG each call, no shared state) — this parallelizes across CPU
    cores instead of running trials one at a time.
    Returns a dict with: means1/means2 (np.ndarray (n_trials,), the scalar
    per-trial means — still needed for Welch's t-test / grand-mean-CI text),
    bin_avg1/bin_std1 (np.ndarray (11,) — whole-number-binned SVPs-per-IgM
    counts, averaged/stddev'd across trials), bin_avg2/bin_std2 (same for
    IgMs-per-SVP, length depends on the data's observed max), counts1/counts2
    (the raw (n_trials, width) per-trial matrices behind bin_avg1/bin_avg2 —
    RealDataCompareWorker stacks these raw matrices across tomograms before
    aggregating, rather than averaging each tomogram's bin_avg separately),
    and sample_placement/sample_result (from the last-COMPLETED trial — not
    necessarily the last-submitted, since pool.imap_unordered returns
    results in completion order; statistically inconsequential since every
    trial is an independent, identically-distributed draw). Returns None if
    stop_event fires mid-run. Shared loop used by BootstrapWorker (single
    scenario), DilutionWorker (two scenarios), and RealDataCompareWorker (one
    scenario per real tomogram)."""
    means1, means2 = [], []
    all_data1, all_data2 = [], []
    sample_placement = sample_result = None
    completed = 0
    n_workers = max(1, (os.cpu_count() or 1) - 1)   # leave a core free for the GUI thread
    chunksize = max(1, n_trials // (4 * n_workers))  # amortize IPC overhead across several trials/task

    with _MP_CTX.Pool(processes=n_workers) as pool:
        for mean1, _std1, mean2, _std2, placement, result, data1, data2 in pool.imap_unordered(
                run_bootstrap_trial_full, [params] * n_trials, chunksize=chunksize):
            if stop_event.is_set():
                pool.terminate()
                return None
            means1.append(mean1); means2.append(mean2)
            all_data1.append(data1); all_data2.append(data2)
            sample_placement, sample_result = placement, result   # last COMPLETED trial
            completed += 1
            if progress_cb is not None and (completed % 25 == 0 or completed == n_trials):
                progress_cb(completed, n_trials)

    max_k1 = MAX_SVP_PER_IGM   # fixed cap — always the same regardless of params
    counts1 = np.zeros((n_trials, max_k1 + 1))
    for i, d1 in enumerate(all_data1):
        if d1:
            counts1[i] = np.bincount(d1, minlength=max_k1 + 1)[:max_k1 + 1]

    max_k2 = max((max(d2) for d2 in all_data2 if d2), default=0)   # unbounded — derive from the data
    counts2 = np.zeros((n_trials, max_k2 + 1))
    for i, d2 in enumerate(all_data2):
        if d2:
            counts2[i] = np.bincount(d2, minlength=max_k2 + 1)

    return dict(
        means1=np.array(means1), means2=np.array(means2),
        bin_avg1=counts1.mean(axis=0), bin_std1=counts1.std(axis=0),
        bin_avg2=counts2.mean(axis=0), bin_std2=counts2.std(axis=0),
        pooled_counts1=counts1.sum(axis=0), pooled_counts2=counts2.sum(axis=0),
        counts1=counts1, counts2=counts2,
        sample_placement=sample_placement, sample_result=sample_result,
    )


def _run_n_trials_collect_nn(params: SimParams, n_trials: int, edges: np.ndarray,
                             stop_event: threading.Event, progress_cb=None,
                             radius: float = NN_DEFAULT_RADIUS_NM):
    """Sibling of _run_n_trials_collect for the Nearest Neighbour Analysis
    mode: each trial's payload is its raw interaction-pair distance
    histogram over the shared `edges` (see run_nn_trial). Returns
    counts (n_trials, n_bins) and dist_sum (total of all pair distances
    over all trials, for an exact simulated mean), or None if stop_event
    fires mid-run."""
    rows = []
    dist_sum = 0.0
    sample_placement = sample_result = None
    completed = 0
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    chunksize = max(1, n_trials // (4 * n_workers))

    with _MP_CTX.Pool(processes=n_workers) as pool:
        for counts, trial_sum, placement, result in pool.imap_unordered(
                functools.partial(run_nn_trial, edges=edges, radius=radius),
                [params] * n_trials,
                chunksize=chunksize):
            if stop_event.is_set():
                pool.terminate()
                return None
            rows.append(counts)
            dist_sum += trial_sum
            sample_placement, sample_result = placement, result
            completed += 1
            if progress_cb is not None and (completed % 25 == 0 or completed == n_trials):
                progress_cb(completed, n_trials)

    return dict(
        counts=np.array(rows, dtype=float).reshape(len(rows), len(edges) - 1),
        dist_sum=dist_sum,
        sample_placement=sample_placement, sample_result=sample_result,
    )


def zscore_vs_distribution(means_a, means_b) -> float:
    """z-score of mean(means_b) against the distribution of means_a. Works
    whether means_b has 1 trial or many — np.mean of a single-element array
    is just that element, so 'the mean of B' degrades naturally to 'the
    single real value' when n_trials_b == 1. Returns nan if means_a has
    fewer than 2 trials (no defined spread to measure against) or is
    degenerate (zero spread)."""
    b_stat = float(np.mean(means_b))
    mean_a = float(np.mean(means_a))
    std_a  = float(np.std(means_a, ddof=1)) if len(means_a) > 1 else float('nan')
    return (b_stat - mean_a) / std_a if std_a > 0 else float('nan')


def zscore_pvalue(z: float) -> float:
    """Two-sided p-value for a z-score under the standard normal
    distribution. nan in, nan out (scipy's norm.sf(nan) is already nan, no
    special-casing needed)."""
    return float(2 * scipy_stats.norm.sf(abs(z)))


def mean_comparison_stats(obs_values, sim_trial_means, sim_pooled_counts,
                          z: float, p: float, n_boot: int = 10000) -> dict:
    """Observed-vs-simulated mean comparison for one per-particle metric.
    Observed: mean of the pooled per-particle values, 95% percentile
    bootstrap CI over particles (fixed seed), per-particle SD. Simulated:
    mean of the per-trial means, 95% interval = mean +/- 1.96 SD of those
    trial means (the same mean/SD the z-test uses, so the observed mean lies
    outside it exactly when p < 0.05), per-particle SD from the pooled histogram (value k counted
    sim_pooled_counts[k] times). Effect size: Cohen's d with the per-particle
    SDs averaged, sqrt((SD_obs^2 + SD_sim^2) / 2). z/p are passed through
    (the worker's z-test of the observed mean against the trial means)."""
    obs = np.asarray(obs_values, dtype=float)
    sim_means = np.asarray(sim_trial_means, dtype=float)
    counts = np.asarray(sim_pooled_counts, dtype=float)
    ks = np.arange(len(counts))

    obs_mean = float(obs.mean()) if len(obs) else float('nan')
    obs_sd = float(obs.std(ddof=1)) if len(obs) > 1 else float('nan')
    if len(obs):
        rng = np.random.default_rng(0)
        chunk = 1000   # bounds memory to chunk x n_particles indices at a time
        boot = np.concatenate([
            obs[rng.integers(0, len(obs), size=(min(chunk, n_boot - i), len(obs)))].mean(axis=1)
            for i in range(0, n_boot, chunk)])
        obs_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    else:
        obs_ci = (float('nan'), float('nan'))

    sim_mean = float(sim_means.mean()) if len(sim_means) else float('nan')
    sim_sd_means = float(sim_means.std(ddof=1)) if len(sim_means) > 1 else float('nan')
    sim_range = (sim_mean - 1.96 * sim_sd_means, sim_mean + 1.96 * sim_sd_means)
    n_sim_particles = float(counts.sum())
    if n_sim_particles > 1:
        pooled_mean = float((ks * counts).sum() / n_sim_particles)
        sim_sd = float(np.sqrt(((ks - pooled_mean) ** 2 * counts).sum() / (n_sim_particles - 1)))
    else:
        sim_sd = float('nan')

    sd_avg = math.sqrt((obs_sd ** 2 + sim_sd ** 2) / 2.0)
    d = (obs_mean - sim_mean) / sd_avg if sd_avg > 0 else float('nan')
    abs_d = abs(d)
    d_label = ("n/a" if d != d else "negligible" if abs_d < 0.2 else
               "small" if abs_d < 0.5 else "medium" if abs_d < 0.8 else "large")
    return dict(
        obs_mean=obs_mean, obs_ci=obs_ci, obs_sd=obs_sd, obs_n=len(obs),
        sim_mean=sim_mean, sim_range=sim_range, sim_sd=sim_sd, sim_sd_means=sim_sd_means,
        sim_n_trials=len(sim_means), sim_n_particles=int(n_sim_particles),
        diff=obs_mean - sim_mean,
        ratio=obs_mean / sim_mean if sim_mean else float('nan'),
        d=d, d_label=d_label, z=z, p=p,
    )


def p_to_stars(p: float) -> str:
    """Conventional significance stars for a p-value."""
    if p != p:   # nan
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def draw_sig_bracket(ax, x0: float, x1: float, y: float, label: str):
    """Paper-style significance bracket between bars at x0 and x1, starting
    at height y, with `label` centred above it. Raises the y-limit so the
    bracket and label aren't clipped."""
    h = 0.03 * y if y > 0 else 1.0
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color='black', linewidth=1.2)
    ax.text((x0 + x1) / 2, y + h * 1.4, label, ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, max(ax.get_ylim()[1], (y + h) * 1.18))


def z_tag(z: float) -> str:
    """Short qualitative label for a z-score's magnitude, for display next
    to the number (mirrors the app's existing 'significant, p<0.05' style
    tagging)."""
    if z != z:   # nan
        return ""
    if abs(z) > 3:
        return " (rare)"
    if abs(z) > 2:
        return " (unusual)"
    return ""


def pooled_binding_counts(counts1: np.ndarray):
    """From a (n_trials, MAX_SVP_PER_IGM+1) counts1 array (counts1[t, k] =
    number of IgMs with exactly k SVPs bound, in trial t), derive two
    per-trial scalar counts: how many IgMs had >=1 SVP bound ('binding') and
    how many had >=2 SVPs bound ('network'). Works uniformly whether
    counts1 has many rows (a batch of simulated trials) or a single row (one
    real/scenario-B observation) — same degrade-to-scalar behavior as
    zscore_vs_distribution."""
    pooled_ge1 = counts1[:, 1:].sum(axis=1)
    pooled_ge2 = counts1[:, 2:].sum(axis=1)
    return pooled_ge1, pooled_ge2


def pct_from_pooled_ge(pooled_ge, total: float):
    """Convert a per-trial pooled >=k IgM count array (see
    pooled_binding_counts) into (Non-X, X) percentage avg/std, using the
    TRUE per-trial complement (total - pooled_ge). Deriving Non-X's std
    independently (e.g. by summing the variances of several underlying
    bins as if they were independent) overestimates it — those bins are
    negatively correlated (mutually exclusive categories of the same
    fixed-size trial: more IgMs in one bin mechanically means fewer
    elsewhere) — which is what let the Binding/Network popup's displayed
    bar+error-bar exceed 100% even after bin_avg1_a/bin_std1_a themselves
    were fixed. Degrades correctly to std=0 for a single/pooled real
    observation (an array of length 1)."""
    pooled_ge = np.asarray(pooled_ge, dtype=float)
    if total <= 0:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    pct_x = pooled_ge / total * 100.0
    avg_x, std_x = float(pct_x.mean()), float(pct_x.std())
    return np.array([100.0 - avg_x, avg_x]), np.array([std_x, std_x])


def _pick_save_ext(path: str, chosen_filter: str) -> str:
    """PNG/SVG extension implied by a QFileDialog result — trusts a typed
    '.svg'/'.png' suffix first, falls back to whichever filter was active."""
    return 'svg' if (path.lower().endswith('.svg') or 'svg' in chosen_filter.lower()) else 'png'


def _save_figure_dialog(parent, fig, default_name: str, title: str = "Save Plot") -> Optional[str]:
    """Prompt for a PNG/SVG save path and write `fig` to it. Returns the
    saved path, or None if the user cancelled. Shared by any single-figure
    export (e.g. NetworkInfoDialog's plot) — MainWindow._save_screenshots
    has its own multi-file flow (one dialog, several suffixed outputs) and
    reuses only _pick_save_ext, not this."""
    from PyQt5.QtWidgets import QFileDialog
    path, chosen_filter = QFileDialog.getSaveFileName(
        parent, title, default_name, "PNG Files (*.png);;SVG Files (*.svg)")
    if not path:
        return None
    ext = _pick_save_ext(path, chosen_filter)
    if not path.lower().endswith(f'.{ext}'):
        path = f"{path}.{ext}"
    kwargs = dict(bbox_inches='tight', facecolor=fig.get_facecolor())
    if ext == 'png':
        kwargs['dpi'] = 150
    fig.savefig(path, format=ext, **kwargs)
    return path


def _project_labels_to_display(plotter, labels):
    """World -> display-pixel (x from left, y from bottom, at 1x) for each
    label point, keeping only points inside the window and in front of the
    camera. Returns [(x, y, text, color, font_px), ...]."""
    ren = plotter.renderer
    w, h = plotter.ren_win.GetSize()
    out = []
    for points, texts, color, font_px in labels:
        for pt, text in zip(np.asarray(points, dtype=float), texts):
            ren.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
            ren.WorldToDisplay()
            x, y, z = ren.GetDisplayPoint()
            if 0 <= x <= w and 0 <= y <= h and 0 <= z <= 1:
                out.append((x, y, text, color, font_px))
    return out


def _project_to_display(plotter, points) -> np.ndarray:
    """World -> display coords (x from left, y from bottom, at 1x; z = depth
    in [0, 1] when in front of the camera) for each point, WITHOUT dropping
    off-screen ones (lines crossing the edge are clipped by matplotlib)."""
    ren = plotter.renderer
    out = np.empty((len(points), 3))
    for i, pt in enumerate(np.asarray(points, dtype=float)):
        ren.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
        ren.WorldToDisplay()
        out[i] = ren.GetDisplayPoint()
    return out


_SVG_BOX_ACTORS = ('inner_box', 'outer_box')


def _save_plotter_hires_svg(plotter, path: str, scale: int = 4, labels=None, lines=None) -> str:
    """Export a pyvista 3D view as an .svg that stays sharp. VTK's own
    save_graphic (GL2PS) embeds a flat, aliased, unshaded screen grab, so
    instead render the same view at `scale`x resolution with the normal
    lighting/anti-aliasing and embed that PNG via matplotlib (true vector
    output isn't practical for shaded meshes). A same-named .png is written
    alongside. Returns the .svg path.

    The 3D box wireframe (perspective, double-framed) is hidden for the
    render and replaced by one flat black rectangle — the 2D outline of the
    inner box's projection — drawn as an editable vector element. The image
    is cropped to that frame, so every mode exports the same framing.

    `labels` ([(points (N,3), texts, color, font_px), ...]): VTK point labels
    vanish from magnified renders, so they are redrawn on top as editable
    text. `lines` ((points (M,3), segments (K,2) index pairs, color, alpha,
    width_px, actor_name)): drawn as editable vector paths BEHIND the
    particles — the named on-screen line actor is hidden and the particles
    are rendered on a transparent background over them."""
    from matplotlib.patches import Rectangle
    base = path[:-4] if path.lower().endswith(('.svg', '.png')) else path
    dpi = 100

    box_actor = plotter.actors.get('inner_box')
    box_bounds = box_actor.GetBounds() if box_actor is not None else None
    hide_names = _SVG_BOX_ACTORS + ((lines[5],) if lines else ())
    hidden = []
    for name in hide_names:
        actor = plotter.actors.get(name)
        if actor is not None and actor.GetVisibility():
            actor.SetVisibility(False)
            hidden.append(actor)
    try:
        plotter.render()
        win_w, win_h = plotter.ren_win.GetSize()
        projected = _project_labels_to_display(plotter, labels) if labels else []
        if box_bounds is not None:
            x0b, x1b, y0b, y1b, z0b, z1b = box_bounds
            corners = np.array([[x, y, z] for x in (x0b, x1b) for y in (y0b, y1b)
                                for z in (z0b, z1b)])
            cd = _project_to_display(plotter, corners)
            fx0, fx1 = max(cd[:, 0].min(), 0.0), min(cd[:, 0].max(), float(win_w))
            fy0, fy1 = max(cd[:, 1].min(), 0.0), min(cd[:, 1].max(), float(win_h))
        else:
            fx0, fx1, fy0, fy1 = 0.0, float(win_w), 0.0, float(win_h)
        seg_disp = None
        if lines:
            pts, segs = lines[0], np.asarray(lines[1], dtype=int)
            pd_ = _project_to_display(plotter, pts)
            if len(segs):
                in_front = (pd_[:, 2] >= 0) & (pd_[:, 2] <= 1)
                segs = segs[in_front[segs[:, 0]] & in_front[segs[:, 1]]]
                seg_disp = pd_[segs][:, :, :2]   # (K, 2, 2) display xy
        img = plotter.screenshot(return_img=True, scale=scale,
                                 transparent_background=bool(lines))
    finally:
        for actor in hidden:
            actor.SetVisibility(True)
        plotter.render()

    full_h, full_w = img.shape[:2]
    sx, sy = full_w / win_w, full_h / win_h

    def _to_img(x, y):
        return x * sx, full_h - y * sy

    # Crop to the frame plus a small pad (particles straddling the box edge
    # stay whole).
    pad = int(0.02 * max(full_w, full_h))
    left, top = _to_img(fx0, fy1)
    right, bottom = _to_img(fx1, fy0)
    c0, r0 = max(int(left) - pad, 0), max(int(top) - pad, 0)
    c1, r1 = min(int(np.ceil(right)) + pad, full_w), min(int(np.ceil(bottom)) + pad, full_h)
    img = img[r0:r1, c0:c1]
    h, w = img.shape[:2]
    px_to_pt = 72 / dpi

    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    if seg_disp is not None and len(seg_disp):
        seg_img = np.empty_like(seg_disp)
        seg_img[..., 0] = seg_disp[..., 0] * sx - c0
        seg_img[..., 1] = full_h - seg_disp[..., 1] * sy - r0
        ax.add_collection(LineCollection(
            seg_img, colors=lines[2], alpha=lines[3],
            linewidths=lines[4] * scale * px_to_pt, zorder=1))
    ax.imshow(img, interpolation='none', zorder=2)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis('off')
    for x, y, text, color, font_px in projected:
        ix, iy = _to_img(x, y)
        ax.text(ix - c0, iy - r0, text,
                color=color, fontsize=font_px * scale * px_to_pt,
                ha='left', va='bottom', clip_on=False, zorder=3,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
    ax.add_patch(Rectangle((left - c0, top - r0), right - left, bottom - top,
                           fill=False, edgecolor='black',
                           linewidth=1.5 * scale * px_to_pt, zorder=4, clip_on=False))
    with matplotlib.rc_context({'svg.fonttype': 'none'}):
        fig.savefig(f"{base}.svg", format='svg', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.05, facecolor='white')
    fig.savefig(f"{base}.png", format='png', dpi=dpi,
                bbox_inches='tight', pad_inches=0.05, facecolor='white')
    return f"{base}.svg"


def chi2_homogeneity(pooled_a, pooled_b):
    """Chi-square test of homogeneity between two pooled whole-number-bin
    count profiles — tests whether the SHAPE of the distribution differs,
    independent of the z-score's test of whether the MEAN differs. Pads to
    equal length, drops bins that are zero in BOTH (avoids zero-expected-
    frequency errors), and requires >=2 remaining bins. Returns
    (chi2_stat, pvalue, dof), with (nan, nan, 0) if the test isn't
    well-defined (too little data)."""
    n = max(len(pooled_a), len(pooled_b))
    a = np.pad(pooled_a, (0, n - len(pooled_a)))
    b = np.pad(pooled_b, (0, n - len(pooled_b)))
    mask = (a + b) > 0
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float('nan'), float('nan'), 0
    chi2, p, dof, _expected = scipy_stats.chi2_contingency(np.array([a, b]))
    return float(chi2), float(p), int(dof)


def bootstrap_result_to_dataframe(result: BootstrapResult) -> pd.DataFrame:
    return pd.DataFrame({
        'trial':            np.arange(1, result.n_trials + 1),
        'mean_svp_per_igm': result.trial_means1,
        'mean_igm_per_svp': result.trial_means2,
    })


class BootstrapWorker(QThread):
    finished  = pyqtSignal(object)   # BootstrapResult
    progress  = pyqtSignal(str)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, params: SimParams, n_trials: int):
        super().__init__()
        self.params      = params
        self.n_trials    = n_trials
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            def _cb(i, n):
                self.progress.emit(f"Bootstrap trial {i}/{n}")
            out = _run_n_trials_collect(self.params, self.n_trials, self._stop_event, _cb)
            if out is None:
                self.cancelled.emit()
                return
            self.finished.emit(BootstrapResult(
                trial_means1=out['means1'], trial_means2=out['means2'],
                bin_avg1=out['bin_avg1'], bin_std1=out['bin_std1'],
                bin_avg2=out['bin_avg2'], bin_std2=out['bin_std2'],
                n_trials=self.n_trials,
                sample_trial_index=self.n_trials,
                sample_placement=out['sample_placement'],
                sample_result=out['sample_result'],
            ))
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# DILUTION COMPARISON — bootstraps two dilution scenarios and Welch's t-tests
# their trial-mean distributions against each other
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DilutionResult:
    params_a: object; params_b: object
    means1_a: object; means1_b: object   # SVPs/IgM trial means, scenario A/B (feed zscore_vs_distribution)
    means2_a: object; means2_b: object   # IgMs/SVP trial means, scenario A/B (feed zscore_vs_distribution)
    bin_avg1_a: object; bin_std1_a: object; bin_avg1_b: object; bin_std1_b: object
    bin_avg2_a: object; bin_std2_a: object; bin_avg2_b: object; bin_std2_b: object
    sample_placement_a: object; sample_result_a: object
    sample_placement_b: object; sample_result_b: object
    z1: float; z2: float                 # z-score of B's mean vs A's distribution, metric 1 and 2
    zpvalue1: float; zpvalue2: float     # two-sided normal p-value for z1/z2
    chi2_stat1: float; chi2_pvalue1: float; chi2_dof1: int   # chi-square homogeneity test, metric 1
    chi2_stat2: float; chi2_pvalue2: float; chi2_dof2: int   # chi-square homogeneity test, metric 2
    # Binding/network sub-analysis (SVPs-per-IgM metric only) shown in the "(i)" popup:
    # pooled per-trial count of IgMs with >=1 / >=2 SVPs bound, scenario A/B, plus
    # the z-score of B's pooled value against A's distribution of that pooled value.
    pooled_ge1_a: object = None; pooled_ge1_b: object = None
    pooled_ge2_a: object = None; pooled_ge2_b: object = None
    z_ge1: float = float('nan'); zp_ge1: float = float('nan')
    z_ge2: float = float('nan'); zp_ge2: float = float('nan')
    # Conditional chi-square shape test among qualifying bins only (>=1 / >=2) —
    # see chi2p_ge1/chi2p_ge2 computation in DilutionWorker.run().
    chi2p_ge1: float = float('nan'); chi2p_ge2: float = float('nan')
    n_trials_a: int = 0
    n_trials_b: int = 0


def dilution_result_to_dataframe(result: DilutionResult) -> pd.DataFrame:
    df_a = pd.DataFrame({
        'trial': np.arange(1, result.n_trials_a + 1), 'scenario': 'A',
        'mean_svp_per_igm': result.means1_a, 'mean_igm_per_svp': result.means2_a,
    })
    df_b = pd.DataFrame({
        'trial': np.arange(1, result.n_trials_b + 1), 'scenario': 'B',
        'mean_svp_per_igm': result.means1_b, 'mean_igm_per_svp': result.means2_b,
    })
    return pd.concat([df_a, df_b], ignore_index=True)


def scenario_summary_text(letter: str, title: str, p: SimParams) -> str:
    """One-line-per-field summary of a dilution scenario's SimParams, shown
    above its 3D viewport so it's visible that only particle counts/box X-Y
    changed — everything else matches what was set in the Parameters group."""
    return (
        f"{letter} — {title}\n"
        f"Box: {p.box_x:.1f} × {p.box_y:.1f} × {p.box_z:.1f} nm  |  "
        f"SVPs: {p.n_svp}  |  IgMs: {p.n_igm} ({p.n_igm_type1}/{p.n_igm_type2}/{p.n_igm_type3})  |  "
        f"diam: {p.svp_diameter:.1f} nm  |  threshold: {p.threshold:.1f} nm  |  "
        f"border conc: {p.border_conc:.3g} /nm³"
    )


class DilutionWorker(QThread):
    finished  = pyqtSignal(object)   # DilutionResult
    progress  = pyqtSignal(str)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, params_a: SimParams, params_b: SimParams, n_trials_a: int, n_trials_b: int):
        super().__init__()
        self.params_a    = params_a
        self.params_b    = params_b
        self.n_trials_a  = n_trials_a
        self.n_trials_b  = n_trials_b
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            def _cb_a(i, n):
                self.progress.emit(f"Scenario A (same volume, diluted concentration): trial {i}/{n}")
            def _cb_b(i, n):
                self.progress.emit(f"Scenario B (same dilution, smaller volume): trial {i}/{n}")

            out_a = _run_n_trials_collect(self.params_a, self.n_trials_a, self._stop_event, _cb_a)
            if out_a is None:
                self.cancelled.emit(); return

            out_b = _run_n_trials_collect(self.params_b, self.n_trials_b, self._stop_event, _cb_b)
            if out_b is None:
                self.cancelled.emit(); return

            z1 = zscore_vs_distribution(out_a['means1'], out_b['means1'])
            z2 = zscore_vs_distribution(out_a['means2'], out_b['means2'])
            zp1 = zscore_pvalue(z1)
            zp2 = zscore_pvalue(z2)
            chi2_stat1, chi2_p1, chi2_dof1 = chi2_homogeneity(out_a['pooled_counts1'], out_b['pooled_counts1'])
            chi2_stat2, chi2_p2, chi2_dof2 = chi2_homogeneity(out_a['pooled_counts2'], out_b['pooled_counts2'])

            ge1_a, ge2_a = pooled_binding_counts(out_a['counts1'])
            ge1_b, ge2_b = pooled_binding_counts(out_b['counts1'])
            z_ge1 = zscore_vs_distribution(ge1_a, ge1_b); zp_ge1 = zscore_pvalue(z_ge1)
            z_ge2 = zscore_vs_distribution(ge2_a, ge2_b); zp_ge2 = zscore_pvalue(z_ge2)
            # 2x2 chi-square: {Non-binding, Binding} x {A, B}, and
            # {No-network, Network} x {A, B} — a true 2-category contingency
            # table (scipy applies Yates' continuity correction automatically
            # for 2x2 tables), distinct from z_ge1/z_ge2 (which test whether
            # the Binding/Network COUNT itself is unusual, not its proportion).
            chi2p_ge1 = chi2_homogeneity(
                np.array([out_a['pooled_counts1'][0], out_a['pooled_counts1'][1:].sum()]),
                np.array([out_b['pooled_counts1'][0], out_b['pooled_counts1'][1:].sum()]))[1]
            chi2p_ge2 = chi2_homogeneity(
                np.array([out_a['pooled_counts1'][:2].sum(), out_a['pooled_counts1'][2:].sum()]),
                np.array([out_b['pooled_counts1'][:2].sum(), out_b['pooled_counts1'][2:].sum()]))[1]

            self.finished.emit(DilutionResult(
                params_a=self.params_a, params_b=self.params_b,
                means1_a=out_a['means1'], means1_b=out_b['means1'],
                means2_a=out_a['means2'], means2_b=out_b['means2'],
                bin_avg1_a=out_a['bin_avg1'], bin_std1_a=out_a['bin_std1'],
                bin_avg1_b=out_b['bin_avg1'], bin_std1_b=out_b['bin_std1'],
                bin_avg2_a=out_a['bin_avg2'], bin_std2_a=out_a['bin_std2'],
                bin_avg2_b=out_b['bin_avg2'], bin_std2_b=out_b['bin_std2'],
                sample_placement_a=out_a['sample_placement'], sample_result_a=out_a['sample_result'],
                sample_placement_b=out_b['sample_placement'], sample_result_b=out_b['sample_result'],
                z1=z1, z2=z2, zpvalue1=zp1, zpvalue2=zp2,
                chi2_stat1=chi2_stat1, chi2_pvalue1=chi2_p1, chi2_dof1=chi2_dof1,
                chi2_stat2=chi2_stat2, chi2_pvalue2=chi2_p2, chi2_dof2=chi2_dof2,
                pooled_ge1_a=ge1_a, pooled_ge1_b=ge1_b, pooled_ge2_a=ge2_a, pooled_ge2_b=ge2_b,
                z_ge1=z_ge1, zp_ge1=zp_ge1, z_ge2=z_ge2, zp_ge2=zp_ge2,
                chi2p_ge1=chi2p_ge1, chi2p_ge2=chi2p_ge2,
                n_trials_a=self.n_trials_a, n_trials_b=self.n_trials_b,
            ))
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# REAL DATA COMPARISON — one simulated distribution (N trials) vs. one real
# tomogram (n=1, loaded from STAR files). Statistically, this is Dilution
# Comparison with side B's "distribution" hard-capped at a single real
# observation instead of a second batch of random placements — the same
# zscore_vs_distribution/chi2_homogeneity functions already degrade
# correctly to a length-1 B side.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RealDataCompareResult:
    params_a: object; real_params: object
    means1_a: object; means2_a: object
    bin_avg1_a: object; bin_std1_a: object
    bin_avg2_a: object; bin_std2_a: object
    pooled_counts1_a: object; pooled_counts2_a: object
    sample_placement_a: object; sample_result_a: object
    real_mean1: float; real_mean2: float
    real_bin1: object; real_bin2: object
    real_placement: object; real_result: object
    z1: float; z2: float
    zpvalue1: float; zpvalue2: float
    chi2_stat1: float; chi2_pvalue1: float; chi2_dof1: int
    chi2_stat2: float; chi2_pvalue2: float; chi2_dof2: int
    # Binding/network sub-analysis (SVPs-per-IgM metric only) shown in the "(i)" popup —
    # see DilutionResult for field meaning; "b" side here is the single real observation.
    pooled_ge1_a: object = None; ge1_real: float = float('nan')
    pooled_ge2_a: object = None; ge2_real: float = float('nan')
    z_ge1: float = float('nan'); zp_ge1: float = float('nan')
    z_ge2: float = float('nan'); zp_ge2: float = float('nan')
    chi2p_ge1: float = float('nan'); chi2p_ge2: float = float('nan')
    n_trials_a: int = 0
    n_svp_real: int = 0
    n_igm_real: int = 0
    # One entry per loaded tomogram (load order), each {name, real_placement,
    # real_result, real_params, sim_placement, sim_result, sim_params} — lets
    # the Viewer A/B toggle show any tomogram's real data + its OWN matched
    # simulated sample, not just whichever ran last (sample_*/real_* above).
    per_tomogram: list = None
    # Flat per-particle observed values pooled over tomograms (SVPs per IgM,
    # IgMs per SVP) — the mean-comparison popup bootstraps these.
    obs_values1: object = None; obs_values2: object = None
    force_no_border: bool = False


def real_data_compare_result_to_dataframe(result: 'RealDataCompareResult') -> pd.DataFrame:
    # means1_a/means2_a are concatenated across every (tomogram, trial) pair
    # (see RealDataCompareWorker.run()), so their length is n_trials_a *
    # n_tomograms, not n_trials_a alone — 'trial' here is a pooled running
    # index across all tomograms, not a per-tomogram trial number.
    df_a = pd.DataFrame({
        'trial': np.arange(1, len(result.means1_a) + 1), 'scenario': 'Simulated',
        'mean_svp_per_igm': result.means1_a, 'mean_igm_per_svp': result.means2_a,
    })
    df_b = pd.DataFrame({
        'trial': [1], 'scenario': 'Observed (pooled)',
        'mean_svp_per_igm': [result.real_mean1], 'mean_igm_per_svp': [result.real_mean2],
    })
    return pd.concat([df_a, df_b], ignore_index=True)


def compare_scenario_summary_text(letter: str, title: str, p: SimParams, n_extra: str = "") -> str:
    """Same one-liner style as scenario_summary_text, used for viewer A/B
    labels in Real Data Comparison mode (no subtype breakdown or border conc,
    since the real side has neither)."""
    return (
        f"{letter} — {title}\n"
        f"Box: {p.box_x:.1f} × {p.box_y:.1f} × {p.box_z:.1f} nm  |  "
        f"SVPs: {p.n_svp}  |  IgMs: {p.n_igm}  |  "
        f"diam: {p.svp_diameter:.1f} nm  |  threshold: {p.threshold:.1f} nm{n_extra}")


def _pad_to_common_width(matrices: list) -> list:
    """Zero-pad a list of 2D (n_rows, width) arrays — that may have
    different widths (e.g. per-tomogram counts2 matrices, whose column
    count depends on that tomogram's own observed max) — to the widest
    one, so they can be summed/combined elementwise by matching row index
    (unlike vstacking, which would treat every row as an independent
    sample instead of combining same-trial-index rows across tomograms —
    see RealDataCompareWorker.run())."""
    max_w = max(m.shape[1] for m in matrices)
    return [np.pad(m, ((0, 0), (0, max_w - m.shape[1]))) for m in matrices]


def _pad_and_sum(arrays: list) -> np.ndarray:
    """Sum 1D count arrays that may have different lengths, zero-padding
    every array to the longest one first."""
    max_w = max(len(a) for a in arrays)
    padded = [np.pad(np.asarray(a, dtype=float), (0, max_w - len(a))) for a in arrays]
    return np.sum(padded, axis=0)


class RealDataCompareWorker(QThread):
    """Runs one simulated batch PER loaded real tomogram (each box/particle-
    count matched to that tomogram, mirroring NearestNeighbourWorker). Real
    bin counts are pooled by summing across tomograms (raw counts genuinely
    just add up). Simulated bin counts are combined by MATCHING TRIAL INDEX —
    trial t of every tomogram is summed into one "combined experiment"
    sample — rather than pooling every (tomogram, trial) row independently;
    the latter would conflate different-sized tomograms' raw-count scale
    differences with genuine trial-to-trial noise and inflate error bars
    (see the comment above the combined_counts1/2 computation in run())."""
    finished  = pyqtSignal(object)   # RealDataCompareResult
    progress  = pyqtSignal(str)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, n_trials_a: int, tomograms: list, force_no_border: bool):
        super().__init__()
        self.n_trials_a = n_trials_a
        self.tomograms = tomograms
        self.force_no_border = force_no_border
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            all_counts1, all_counts2 = [], []
            all_means1, all_means2 = [], []
            all_real_data1, all_real_data2 = [], []
            real_bin1_list, real_bin2_list = [], []
            per_tomogram = []
            sample_placement_a = sample_result_a = sample_params_a = None
            sample_real_placement = sample_real_result = sample_real_params = None
            n_tomo = len(self.tomograms)

            for t_i, tomo in enumerate(self.tomograms):
                params_i = tomo['params']
                if self.force_no_border:
                    params_i = dataclasses.replace(params_i, border_conc=0.0)

                def _cb(i, n, t_i=t_i):
                    self.progress.emit(f"Tomogram {t_i + 1}/{n_tomo} ({tomo['name']}), trial {i}/{n}")

                out = _run_n_trials_collect(params_i, self.n_trials_a, self._stop_event, _cb)
                if out is None:
                    self.cancelled.emit(); return

                all_counts1.append(out['counts1']); all_counts2.append(out['counts2'])
                all_means1.append(out['means1']); all_means2.append(out['means2'])
                real_bin1_list.append(tomo['bin1']); real_bin2_list.append(tomo['bin2'])
                all_real_data1.append(tomo['data1']); all_real_data2.append(tomo['data2'])

                sample_placement_a, sample_result_a, sample_params_a = (
                    out['sample_placement'], out['sample_result'], params_i)
                sample_real_placement, sample_real_result, sample_real_params = (
                    tomo['placement'], tomo['result'], params_i)
                per_tomogram.append(dict(
                    name=tomo['name'],
                    real_placement=tomo['placement'], real_result=tomo['result'], real_params=params_i,
                    sim_placement=out['sample_placement'], sim_result=out['sample_result'],
                    sim_params=params_i,
                    # For the mean-comparison popup's per-tomogram table/methods.
                    data1_source=tomo.get('data1_source', 'geometric contact (threshold)'),
                    n_inner_svp=len(tomo['data2']),
                    obs_mean1=float(np.mean(tomo['data1'])) if len(tomo['data1']) else float('nan'),
                    obs_mean2=float(np.mean(tomo['data2'])) if len(tomo['data2']) else float('nan'),
                    sim_mean1=float(np.mean(out['means1'])), sim_mean2=float(np.mean(out['means2'])),
                ))

            # Combine tomograms by MATCHING TRIAL INDEX, not by pooling every
            # (tomogram, trial) row together: trial t of every tomogram is
            # summed into one "combined experiment" sample (all tomograms'
            # own random placements at once), giving n_trials_a independent
            # combined samples. Taking mean/std across THOSE (rather than
            # across all n_trials_a*n_tomograms individual rows) keeps
            # different-sized tomograms from inflating the apparent spread —
            # pooling raw rows directly conflates "these tomograms are
            # different sizes" with "this experiment is noisy", which is what
            # previously pushed displayed bar+error-bar past 100%. counts1 is
            # always the same fixed width (MAX_SVP_PER_IGM+1) across
            # tomograms, so it can be summed directly; counts2's width
            # depends on each tomogram's own observed max, so it needs
            # padding to a common width first.
            combined_counts1 = sum(all_counts1)
            combined_counts2 = sum(_pad_to_common_width(all_counts2))
            bin_avg1_a, bin_std1_a = combined_counts1.mean(axis=0), combined_counts1.std(axis=0)
            bin_avg2_a, bin_std2_a = combined_counts2.mean(axis=0), combined_counts2.std(axis=0)
            pooled_counts1_a = combined_counts1.sum(axis=0)
            pooled_counts2_a = combined_counts2.sum(axis=0)

            def _weighted_mean_per_trial(combined_counts):
                row_totals = combined_counts.sum(axis=1)
                weighted = combined_counts @ np.arange(combined_counts.shape[1])
                return np.divide(weighted, row_totals, out=np.zeros_like(weighted), where=row_totals > 0)

            means1_a = _weighted_mean_per_trial(combined_counts1)
            means2_a = _weighted_mean_per_trial(combined_counts2)

            real_bin1 = _pad_and_sum(real_bin1_list)
            real_bin2 = _pad_and_sum(real_bin2_list)
            flat_real_data1 = [v for d in all_real_data1 for v in d]
            flat_real_data2 = [v for d in all_real_data2 for v in d]
            real_mean1 = float(np.mean(flat_real_data1)) if flat_real_data1 else 0.0
            real_mean2 = float(np.mean(flat_real_data2)) if flat_real_data2 else 0.0

            z1 = zscore_vs_distribution(means1_a, np.array([real_mean1]))
            z2 = zscore_vs_distribution(means2_a, np.array([real_mean2]))
            zp1, zp2 = zscore_pvalue(z1), zscore_pvalue(z2)
            c1, cp1, dof1 = chi2_homogeneity(pooled_counts1_a, real_bin1)
            c2, cp2, dof2 = chi2_homogeneity(pooled_counts2_a, real_bin2)

            ge1_a, ge2_a = pooled_binding_counts(combined_counts1)
            ge1_real_arr, ge2_real_arr = pooled_binding_counts(real_bin1[None, :])
            z_ge1 = zscore_vs_distribution(ge1_a, ge1_real_arr); zp_ge1 = zscore_pvalue(z_ge1)
            z_ge2 = zscore_vs_distribution(ge2_a, ge2_real_arr); zp_ge2 = zscore_pvalue(z_ge2)
            # 2x2 chi-square on true {Non-binding, Binding} / {No-network,
            # Network} categories — see DilutionWorker.run() for rationale.
            chi2p_ge1 = chi2_homogeneity(
                np.array([pooled_counts1_a[0], pooled_counts1_a[1:].sum()]),
                np.array([real_bin1[0], real_bin1[1:].sum()]))[1]
            chi2p_ge2 = chi2_homogeneity(
                np.array([pooled_counts1_a[:2].sum(), pooled_counts1_a[2:].sum()]),
                np.array([real_bin1[:2].sum(), real_bin1[2:].sum()]))[1]

            self.finished.emit(RealDataCompareResult(
                params_a=sample_params_a, real_params=sample_real_params,
                means1_a=means1_a, means2_a=means2_a,
                bin_avg1_a=bin_avg1_a, bin_std1_a=bin_std1_a,
                bin_avg2_a=bin_avg2_a, bin_std2_a=bin_std2_a,
                pooled_counts1_a=pooled_counts1_a, pooled_counts2_a=pooled_counts2_a,
                sample_placement_a=sample_placement_a, sample_result_a=sample_result_a,
                real_mean1=real_mean1, real_mean2=real_mean2,
                real_bin1=real_bin1, real_bin2=real_bin2,
                real_placement=sample_real_placement, real_result=sample_real_result,
                z1=z1, z2=z2, zpvalue1=zp1, zpvalue2=zp2,
                chi2_stat1=c1, chi2_pvalue1=cp1, chi2_dof1=dof1,
                chi2_stat2=c2, chi2_pvalue2=cp2, chi2_dof2=dof2,
                pooled_ge1_a=ge1_a, ge1_real=float(ge1_real_arr[0]),
                pooled_ge2_a=ge2_a, ge2_real=float(ge2_real_arr[0]),
                z_ge1=z_ge1, zp_ge1=zp_ge1, z_ge2=z_ge2, zp_ge2=zp_ge2,
                chi2p_ge1=chi2p_ge1, chi2p_ge2=chi2p_ge2,
                n_trials_a=self.n_trials_a,
                n_svp_real=sum(t['n_svp'] for t in self.tomograms),
                n_igm_real=sum(t['n_igm'] for t in self.tomograms),
                per_tomogram=per_tomogram,
                obs_values1=np.asarray(flat_real_data1, dtype=float),
                obs_values2=np.asarray(flat_real_data2, dtype=float),
                force_no_border=self.force_no_border,
            ))
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# NEAREST NEIGHBOUR ANALYSIS — distance distribution of the SVP-IgM
# interaction pairs (each SVP's nearest IgM, each IgM's nearest SVP, and every
# pair within the interaction radius; each pair counted once — see
# interaction_pairs). Observed: raw counts per bin pooled over tomograms.
# Simulated: raw counts summed over every tomogram and trial / n_trials.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NearestNeighbourResult:
    params_a: object
    bin_edges: object       # (n_bins+1,) nm, shared between real and simulated
    sim_counts_avg: object  # (n_bins,) pair counts summed over tomograms and trials / n_trials
    sim_counts_std: object  # (n_bins,) std across the n_trials combined trials (trial t summed over tomograms)
    real_counts: object     # (n_bins,) observed pair counts, pooled over tomograms
    real_dists: object      # observed interaction-pair distances (nm), pooled over tomograms
    sample_placement_a: object; sample_result_a: object   # from whichever tomogram's batch ran last
    sample_params_a: object = None   # that SAME tomogram's own SimParams (box must match the sample)
    n_trials_a: int = 0
    n_svp_real: int = 0     # summed across tomograms
    n_igm_real: int = 0     # summed across tomograms
    mean_dist_real: float = 0.0
    mean_dist_sim: float = 0.0
    radius: float = NN_DEFAULT_RADIUS_NM   # interaction radius (nm, centre-centre)


def pct_of_total(counts) -> np.ndarray:
    """Per-bin % of the total (all zeros if the total is 0)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    return counts / total * 100.0 if total > 0 else np.zeros_like(counts)


def nn_result_to_dataframe(result: NearestNeighbourResult) -> pd.DataFrame:
    edges = result.bin_edges
    return pd.DataFrame({
        'bin_lo':              edges[:-1],
        'bin_hi':              edges[1:],
        'count_observed':      result.real_counts,
        'count_simulated_avg': result.sim_counts_avg,
        'count_simulated_std': result.sim_counts_std,
        'pct_observed':        pct_of_total(result.real_counts),
        'pct_simulated':       pct_of_total(result.sim_counts_avg),
        'radius_nm':           result.radius,
    })


def nn_raw_result_to_dataframe(tomograms: list) -> pd.DataFrame:
    """One row per observed interaction pair (tomogram['pairs'], set by the
    last NearestNeighbourWorker run), with the rule(s) it met."""
    cols = ['tomogram', 'svp_index', 'igm_index', 'distance_nm',
            'svp_nearest_igm', 'igm_nearest_svp', 'within_radius']
    frames = []
    for tomo in tomograms:
        pairs = tomo.get('pairs')
        if pairs is None:
            continue
        frames.append(pd.DataFrame({
            'tomogram': tomo['name'],
            'svp_index': pairs['svp_idx'], 'igm_index': pairs['igm_idx'],
            'distance_nm': pairs['dist'],
            'svp_nearest_igm': pairs['is_svp_nn'], 'igm_nearest_svp': pairs['is_igm_nn'],
            'within_radius': pairs['in_radius'],
        }))
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def tomogram_interaction_pairs(tomo: dict, radius: float) -> dict:
    """interaction_pairs for one loaded observed tomogram (inner SVPs only —
    all of them, since observed data has no border shell)."""
    svp_pos, _radii, svp_border, igm_pos, _thetas = tomo['placement']
    return interaction_pairs(np.asarray(svp_pos)[~np.asarray(svp_border)], igm_pos, radius)


class NearestNeighbourWorker(QThread):
    """Runs one simulated batch PER loaded tomogram (each box/particle-count
    matched to that tomogram). Bin edges are fixed up front from the largest
    box diagonal (the longest possible pair distance), so each trial is
    binned in its worker process. Tomograms are combined by matching trial
    index (as in RealDataCompareWorker): trial t of every tomogram is summed
    into one combined experiment. Each tomogram's observed pairs are stored
    on it as tomo['pairs'] so the viewer and raw export show exactly what
    was plotted."""
    finished  = pyqtSignal(object)   # NearestNeighbourResult
    progress  = pyqtSignal(str)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, params_a: SimParams, n_trials_a: int, tomograms: list, bin_width: float,
                 radius: float = NN_DEFAULT_RADIUS_NM):
        super().__init__()
        self.params_a, self.n_trials_a = params_a, n_trials_a
        self.tomograms = tomograms
        self.bin_width = bin_width
        self.radius = radius
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            for tomo in self.tomograms:
                tomo['pairs'] = tomogram_interaction_pairs(tomo, self.radius)
            real_dists_pooled = np.concatenate([t['pairs']['dist'] for t in self.tomograms])
            max_diag = max(math.sqrt(t['params'].box_x ** 2 + t['params'].box_y ** 2
                                     + t['params'].box_z ** 2) for t in self.tomograms)
            if len(real_dists_pooled):
                max_diag = max(max_diag, float(real_dists_pooled.max()))
            # Fixed-width bins anchored at 0, via integer multiples of
            # bin_width to avoid np.arange's float-step drift. Nearest-
            # neighbour pairs can be far apart, so bins run to the largest
            # box diagonal, +1 bin of headroom.
            n_edge_bins = int(np.ceil(max(max_diag, 1.0) / self.bin_width)) + 1
            edges = np.arange(n_edge_bins + 1) * self.bin_width

            combined = np.zeros((self.n_trials_a, n_edge_bins))
            sim_dist_sum = 0.0
            sample_placement = sample_result = sample_params = None
            n_tomo = len(self.tomograms)
            for t_i, tomo in enumerate(self.tomograms):
                def _cb(i, n, t_i=t_i):
                    self.progress.emit(f"Tomogram {t_i + 1}/{n_tomo} ({tomo['name']}), trial {i}/{n}")

                out = _run_n_trials_collect_nn(tomo['params'], self.n_trials_a, edges,
                                               self._stop_event, _cb, radius=self.radius)
                if out is None:
                    self.cancelled.emit(); return
                combined += out['counts']
                sim_dist_sum += out['dist_sum']
                # Keep the sample's OWN tomogram's params alongside it — the
                # box drawn in the post-run 3D preview must match whichever
                # tomogram's placement is actually being shown.
                sample_placement, sample_result = out['sample_placement'], out['sample_result']
                sample_params = tomo['params']

            real_counts, _ = np.histogram(real_dists_pooled, bins=edges)
            # Summed over all tomograms and trials, / n_trials == mean of the
            # combined-trial rows.
            sim_counts_avg = combined.mean(axis=0)
            sim_counts_std = combined.std(axis=0)

            mean_dist_real = float(np.mean(real_dists_pooled)) if len(real_dists_pooled) else float('nan')
            total_sim = combined.sum()
            mean_dist_sim = sim_dist_sum / total_sim if total_sim else float('nan')

            n_svp_real = sum(len(t['real']['svp_positions']) for t in self.tomograms)
            n_igm_real = sum(len(t['real']['igm_positions']) for t in self.tomograms)

            self.finished.emit(NearestNeighbourResult(
                params_a=self.params_a,
                bin_edges=edges, sim_counts_avg=sim_counts_avg, sim_counts_std=sim_counts_std,
                real_counts=real_counts.astype(float), real_dists=real_dists_pooled,
                sample_placement_a=sample_placement, sample_result_a=sample_result,
                sample_params_a=sample_params,
                n_trials_a=self.n_trials_a,
                n_svp_real=n_svp_real, n_igm_real=n_igm_real,
                mean_dist_real=mean_dist_real, mean_dist_sim=mean_dist_sim,
                radius=self.radius,
            ))
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IgM-SVP Interaction Simulation")
        self._result:         Optional[SimResult]        = None
        self._worker:         Optional[SimulationWorker] = None
        self._csv_path:       Optional[str]              = None
        self._params:         Optional[SimParams]        = None
        self._hbv_star_path:  Optional[str]              = None
        self._igm_star_path:  Optional[str]              = None
        self._nn_hbv_star_path: Optional[str]            = None
        self._nn_igm_star_path: Optional[str]            = None
        self._svp_actors:     list                       = []
        self._igm_actors:     list                       = []
        self._igm_arm_actors: list                       = []
        self._bootstrap_worker: Optional['BootstrapWorker'] = None
        self._bootstrap_result: Optional['BootstrapResult'] = None
        self.plotter2:         object                     = None
        self._svp_actors2:    list                        = []
        self._igm_actors2:    list                        = []
        self._igm_arm_actors2: list                       = []
        self._nn_viewer_svp_actors:     list               = []
        self._nn_viewer_igm_actors:     list               = []
        self._nn_viewer_igm_arm_actors: list               = []
        self._star_real:      Optional[dict]               = None
        self._animation_timer: Optional[QTimer]            = None
        self._animation_start: Optional[tuple]              = None
        self._animation_end:   Optional[tuple]              = None
        self._animation_frame: int                          = 0
        self._animation_n_frames: int                       = 30
        self._animation_random_params: Optional[SimParams] = None
        self._dilution_worker: Optional['DilutionWorker']  = None
        self._dilution_result: Optional['DilutionResult']  = None
        self._star_orig_result: Optional[SimResult]         = None
        self._animation_random_full: Optional[dict]          = None
        self._recording: bool                                = False
        self._recording_path: Optional[str]                  = None
        self._compare_hbv_star_path:   Optional[str]                    = None
        self._compare_igm_star_path:   Optional[str]                    = None
        self._compare_tomograms:       list                             = []
        self._compare_worker:          Optional['RealDataCompareWorker'] = None
        self._compare_result:          Optional['RealDataCompareResult'] = None
        self._nn_tomograms:            list                               = []
        self._nn_worker:               Optional['NearestNeighbourWorker'] = None
        self._nn_result:               Optional['NearestNeighbourResult'] = None
        self._h1_info_source:          Optional[str]                     = None
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── left scroll panel ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(180)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(6, 6, 6, 6)
        left_l.setSpacing(8)

        # ── mode selector ────────────────────────────────────────────────
        mode_w  = QWidget()
        mode_hl = QHBoxLayout(mode_w)
        mode_hl.setContentsMargins(0, 0, 0, 0)
        mode_hl.addWidget(QLabel("Mode:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Manual / Bootstrap", "Dilution Comparison",
                                "STAR Exact + Randomize", "Observed Data Comparison",
                                "Nearest Neighbour Analysis"])
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_hl.addWidget(self.cmb_mode, stretch=1)
        left_l.addWidget(mode_w)

        # ── load from experiment ─────────────────────────────────────────
        exp_grp  = QGroupBox("Load from Experiment")
        exp_form = QFormLayout(exp_grp)
        exp_form.setLabelAlignment(Qt.AlignRight)

        hbv_w = QWidget()
        hbv_l = QHBoxLayout(hbv_w)
        hbv_l.setContentsMargins(0, 0, 0, 0)
        self.le_hbv_star = QLineEdit()
        self.le_hbv_star.setReadOnly(True)
        self.le_hbv_star.setPlaceholderText("(HBV / SVP STAR file)")
        btn_hbv = QPushButton("Browse…")
        btn_hbv.clicked.connect(self._browse_hbv_star)
        hbv_l.addWidget(self.le_hbv_star)
        hbv_l.addWidget(btn_hbv)
        exp_form.addRow("SVP STAR", hbv_w)

        igm_w = QWidget()
        igm_l = QHBoxLayout(igm_w)
        igm_l.setContentsMargins(0, 0, 0, 0)
        self.le_igm_star = QLineEdit()
        self.le_igm_star.setReadOnly(True)
        self.le_igm_star.setPlaceholderText("(IgM STAR file)")
        btn_igm = QPushButton("Browse…")
        btn_igm.clicked.connect(self._browse_igm_star)
        igm_l.addWidget(self.le_igm_star)
        igm_l.addWidget(btn_igm)
        exp_form.addRow("IgM STAR", igm_w)

        self.btn_apply_star = QPushButton("Apply to Parameters")
        self.btn_apply_star.setEnabled(False)
        self.btn_apply_star.setStyleSheet(
            "QPushButton{background-color:#1976D2;color:white;"
            "font-weight:bold;padding:4px;border-radius:4px;}"
            "QPushButton:hover{background-color:#1565C0;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_apply_star.clicked.connect(self._apply_star_params)
        exp_form.addRow("", self.btn_apply_star)

        # Real Data Comparison's tomogram-pair loader lives here (rather than
        # down inside the "Real Data Comparison" box below) so loading data
        # is the first thing visible when switching to that mode — visible
        # only in Compare mode (see _on_mode_changed). Its own nested group
        # box (distinct title) makes clear this is a SEPARATE, Compare-mode-
        # only, multi-tomogram loader — NOT the same as the single SVP/IgM
        # STAR fields above, which are shared by Manual/STAR-Exact mode via
        # "Apply to Parameters" and only ever hold one pair. Each tomogram
        # added here keeps its own box/particle-count-matched SimParams (see
        # _add_compare_tomogram), mirroring the Nearest Neighbour tab's
        # multi-tomogram pattern.
        self.compare_tomo_loader_w = QGroupBox("Observed Data Comparison — tomogram pairs")
        compare_loader_form = QFormLayout(self.compare_tomo_loader_w)
        compare_loader_form.setLabelAlignment(Qt.AlignRight)

        compare_hbv_w = QWidget()
        compare_hbv_l = QHBoxLayout(compare_hbv_w)
        compare_hbv_l.setContentsMargins(0, 0, 0, 0)
        self.le_compare_hbv_star = QLineEdit()
        self.le_compare_hbv_star.setReadOnly(True)
        self.le_compare_hbv_star.setPlaceholderText("(HBV / SVP STAR file)")
        self.btn_compare_browse_hbv = QPushButton("Browse…")
        self.btn_compare_browse_hbv.clicked.connect(self._browse_compare_hbv_star)
        compare_hbv_l.addWidget(self.le_compare_hbv_star)
        compare_hbv_l.addWidget(self.btn_compare_browse_hbv)
        compare_loader_form.addRow("SVP STAR", compare_hbv_w)

        compare_igm_w = QWidget()
        compare_igm_l = QHBoxLayout(compare_igm_w)
        compare_igm_l.setContentsMargins(0, 0, 0, 0)
        self.le_compare_igm_star = QLineEdit()
        self.le_compare_igm_star.setReadOnly(True)
        self.le_compare_igm_star.setPlaceholderText("(IgM STAR file)")
        self.btn_compare_browse_igm = QPushButton("Browse…")
        self.btn_compare_browse_igm.clicked.connect(self._browse_compare_igm_star)
        compare_igm_l.addWidget(self.le_compare_igm_star)
        compare_igm_l.addWidget(self.btn_compare_browse_igm)
        compare_loader_form.addRow("IgM STAR", compare_igm_w)

        self.btn_add_compare_tomogram = QPushButton("➕ Add Tomogram")
        self.btn_add_compare_tomogram.setStyleSheet(
            "QPushButton{background-color:#5E35B1;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#4527A0;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_add_compare_tomogram.clicked.connect(self._add_compare_tomogram)
        compare_loader_form.addRow("", self.btn_add_compare_tomogram)

        self.lst_compare_tomograms = QListWidget()
        self.lst_compare_tomograms.setMaximumHeight(90)
        self.lst_compare_tomograms.itemSelectionChanged.connect(
            lambda: self.btn_remove_compare_tomogram.setEnabled(
                bool(self.lst_compare_tomograms.selectedItems())))
        compare_loader_form.addRow(self.lst_compare_tomograms)

        self.btn_remove_compare_tomogram = QPushButton("✕ Remove Selected")
        self.btn_remove_compare_tomogram.setEnabled(False)
        self.btn_remove_compare_tomogram.clicked.connect(self._remove_compare_tomogram)
        compare_loader_form.addRow("", self.btn_remove_compare_tomogram)

        self.compare_tomo_loader_w.setVisible(False)
        exp_form.addRow(self.compare_tomo_loader_w)

        left_l.addWidget(exp_grp)

        # ── Nearest Neighbour Analysis mode ──────────────────────────────
        # Self-contained STAR-pair loader (independent of exp_grp above, so
        # browsing here doesn't affect what other modes would use) — placed
        # right under exp_grp so browsing and adding tomograms both happen
        # at the top, with no need to scroll past Parameters/other modes.
        self.nn_grp = QGroupBox("Nearest Neighbour Analysis")
        nn_form = QFormLayout(self.nn_grp)
        nn_form.setLabelAlignment(Qt.AlignRight)

        nn_hbv_w = QWidget()
        nn_hbv_l = QHBoxLayout(nn_hbv_w)
        nn_hbv_l.setContentsMargins(0, 0, 0, 0)
        self.le_nn_hbv_star = QLineEdit()
        self.le_nn_hbv_star.setReadOnly(True)
        self.le_nn_hbv_star.setPlaceholderText("(HBV / SVP STAR file)")
        self.btn_nn_browse_hbv = QPushButton("Browse…")
        self.btn_nn_browse_hbv.clicked.connect(self._browse_nn_hbv_star)
        nn_hbv_l.addWidget(self.le_nn_hbv_star)
        nn_hbv_l.addWidget(self.btn_nn_browse_hbv)
        nn_form.addRow("SVP STAR", nn_hbv_w)

        nn_igm_w = QWidget()
        nn_igm_l = QHBoxLayout(nn_igm_w)
        nn_igm_l.setContentsMargins(0, 0, 0, 0)
        self.le_nn_igm_star = QLineEdit()
        self.le_nn_igm_star.setReadOnly(True)
        self.le_nn_igm_star.setPlaceholderText("(IgM STAR file)")
        self.btn_nn_browse_igm = QPushButton("Browse…")
        self.btn_nn_browse_igm.clicked.connect(self._browse_nn_igm_star)
        nn_igm_l.addWidget(self.le_nn_igm_star)
        nn_igm_l.addWidget(self.btn_nn_browse_igm)
        nn_form.addRow("IgM STAR", nn_igm_w)

        self.btn_add_nn_tomogram = QPushButton("➕ Add Tomogram")
        self.btn_add_nn_tomogram.setStyleSheet(
            "QPushButton{background-color:#5E35B1;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#4527A0;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_add_nn_tomogram.clicked.connect(self._add_nn_tomogram)
        nn_form.addRow("", self.btn_add_nn_tomogram)

        self.lst_nn_tomograms = QListWidget()
        self.lst_nn_tomograms.setMaximumHeight(90)
        self.lst_nn_tomograms.itemSelectionChanged.connect(
            lambda: self.btn_remove_nn_tomogram.setEnabled(
                bool(self.lst_nn_tomograms.selectedItems())))
        nn_form.addRow(self.lst_nn_tomograms)

        self.btn_remove_nn_tomogram = QPushButton("✕ Remove Selected")
        self.btn_remove_nn_tomogram.setEnabled(False)
        self.btn_remove_nn_tomogram.clicked.connect(self._remove_nn_tomogram)
        nn_form.addRow("", self.btn_remove_nn_tomogram)

        self.sb_nn_bin_width = QDoubleSpinBox()
        self.sb_nn_bin_width.setRange(0.1, 500.0)
        self.sb_nn_bin_width.setDecimals(1)
        self.sb_nn_bin_width.setValue(5.0)
        nn_form.addRow("Distance bin width (nm)", self.sb_nn_bin_width)

        self.sb_nn_radius = QDoubleSpinBox()
        self.sb_nn_radius.setRange(0.0, 1000.0)
        self.sb_nn_radius.setDecimals(1)
        self.sb_nn_radius.setValue(NN_DEFAULT_RADIUS_NM)
        self.sb_nn_radius.setToolTip(
            "An SVP–IgM pair is counted as an interaction if the IgM is that "
            "SVP's nearest IgM, OR the SVP is that IgM's nearest SVP, OR their "
            "centres are within this radius. Each pair is counted once, even "
            "if it meets several rules. Applies to observed and simulated "
            "data, and to the lines in the 3D viewer.")
        nn_form.addRow("Interaction radius (nm, centre–centre)", self.sb_nn_radius)

        self.sb_nn_xmax = QDoubleSpinBox()
        self.sb_nn_xmax.setRange(0.0, 100000.0)
        self.sb_nn_xmax.setDecimals(1)
        self.sb_nn_xmax.setValue(0.0)
        self.sb_nn_xmax.setToolTip(
            "Crop the plot's x-axis at this distance (view only — bin data, "
            "CSV exports, and stats are unaffected). 0 = show full range.")
        self.sb_nn_xmax.valueChanged.connect(self._on_nn_plot_opts_changed)
        nn_form.addRow("X-axis max (nm, 0 = auto)", self.sb_nn_xmax)

        self.sb_nn_curve_smooth = QSpinBox()
        self.sb_nn_curve_smooth.setRange(1, 20)
        self.sb_nn_curve_smooth.setValue(3)
        self.sb_nn_curve_smooth.setToolTip(
            "The observed-data curve is fit to bins this many times wider than "
            "the display bin width above (view only — bars, CSV exports, and "
            "stats are unaffected) — reduces jaggedness from having few observed "
            "tomograms without changing the displayed histogram.")
        self.sb_nn_curve_smooth.valueChanged.connect(self._on_nn_plot_opts_changed)
        nn_form.addRow("Curve smoothing (combine N bins)", self.sb_nn_curve_smooth)

        self.cb_nn_show_curve = QCheckBox("Show observed-data curve")
        self.cb_nn_show_curve.setChecked(True)
        self.cb_nn_show_curve.stateChanged.connect(self._on_nn_plot_opts_changed)
        nn_form.addRow("", self.cb_nn_show_curve)

        self.lbl_nn_fit = QLabel("")
        self.lbl_nn_fit.setWordWrap(True)
        self.lbl_nn_fit.setStyleSheet("color:#B71C1C;")
        nn_form.addRow(self.lbl_nn_fit)

        self.sb_nn_trials_a = QSpinBox()
        self.sb_nn_trials_a.setRange(2, 100000)
        self.sb_nn_trials_a.setValue(1000)
        nn_form.addRow("Trials: simulated side", self.sb_nn_trials_a)

        self.btn_run_nn = QPushButton("▶ Run Nearest Neighbour Analysis")
        self.btn_run_nn.setEnabled(False)
        self.btn_run_nn.setStyleSheet(
            "QPushButton{background-color:#00838F;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#006064;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_run_nn.clicked.connect(self._run_nn)
        nn_form.addRow("", self.btn_run_nn)

        self.btn_stop_nn = QPushButton("■ Stop")
        self.btn_stop_nn.setEnabled(False)
        self.btn_stop_nn.setStyleSheet(
            "QPushButton{background-color:#E53935;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#d32f2f;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_stop_nn.clicked.connect(self._stop_nn)
        nn_form.addRow("", self.btn_stop_nn)

        self.btn_export_nn_raw = QPushButton("💾 Export Raw Interaction Pairs")
        self.btn_export_nn_raw.setEnabled(False)
        self.btn_export_nn_raw.setToolTip(
            "One row per observed interaction pair (tomogram, SVP, IgM): its "
            "distance and which rule(s) it met (SVP's nearest IgM, IgM's "
            "nearest SVP, within radius).")
        self.btn_export_nn_raw.clicked.connect(self._export_nn_raw)
        nn_form.addRow("", self.btn_export_nn_raw)

        self.btn_open_nn_viewer = QPushButton("🔎 Open 3D Viewer")
        self.btn_open_nn_viewer.setEnabled(False)
        self.btn_open_nn_viewer.setToolTip(
            "Interactive 3D view of the loaded observed tomogram(s): every IgM "
            "and SVP labeled, a line for every counted interaction pair, and "
            "nearest-IgM distances — toggle between tomograms if more than one "
            "is loaded.")
        self.btn_open_nn_viewer.clicked.connect(self._open_nn_viewer)
        nn_form.addRow("", self.btn_open_nn_viewer)

        self.nn_status_label = QLabel(
            "Browse to SVP and IgM STAR files above, then click \"Add "
            "Tomogram\" — repeat (browsing to new STAR files each time) to "
            "add multiple tomograms. Border concentration/thickness are "
            "always treated as 0 for this mode (observed tomograms have no "
            "border-shell concept). Interactions counted: each SVP's nearest "
            "IgM, each IgM's nearest SVP, and every SVP–IgM pair within the "
            "interaction radius (each pair counted once). Edit particle "
            "counts / threshold / SVP diameter as desired below, then \"Run\" "
            "to draw N simulated trials per tomogram (box/particle-count "
            "matched to that tomogram). Observed bars (red) are raw "
            "interaction counts pooled over all loaded tomograms; simulated "
            "bars (grey) are raw counts summed over all trials ÷ N.")
        self.nn_status_label.setWordWrap(True)
        nn_form.addRow(self.nn_status_label)

        left_l.addWidget(self.nn_grp)
        self.nn_grp.setVisible(False)

        # ── simulation parameters ─────────────────────────────────────────
        self.grp_params = QGroupBox("Parameters")
        form = QFormLayout(self.grp_params)
        form.setLabelAlignment(Qt.AlignRight)

        # 1
        self.sb_box_x = QDoubleSpinBox()
        self.sb_box_x.setRange(10.0, 5000.0)
        self.sb_box_x.setSingleStep(10.0)
        self.sb_box_x.setValue(200.0)
        form.addRow("Box X (nm)", self.sb_box_x)

        self.sb_box_y = QDoubleSpinBox()
        self.sb_box_y.setRange(10.0, 5000.0)
        self.sb_box_y.setSingleStep(10.0)
        self.sb_box_y.setValue(200.0)
        form.addRow("Box Y (nm)", self.sb_box_y)

        self.sb_box_z = QDoubleSpinBox()
        self.sb_box_z.setRange(10.0, 5000.0)
        self.sb_box_z.setSingleStep(10.0)
        self.sb_box_z.setValue(200.0)
        form.addRow("Box Z (nm)", self.sb_box_z)

        # 2
        self.sb_n_svp = QSpinBox()
        self.sb_n_svp.setRange(1, 10000)
        self.sb_n_svp.setValue(100)
        form.addRow("Inner SVPs", self.sb_n_svp)

        # 3 — IgM subtype group
        igm_grp  = QGroupBox("IgM Subtypes")
        igm_form = QFormLayout(igm_grp)
        igm_form.setLabelAlignment(Qt.AlignRight)

        mode_w = QWidget()
        mode_l = QHBoxLayout(mode_w)
        mode_l.setContentsMargins(0, 0, 0, 0)
        self.rb_igm_count = QRadioButton("Count")
        self.rb_igm_prop  = QRadioButton("Proportion (%)")
        self.rb_igm_count.setChecked(True)
        mode_l.addWidget(self.rb_igm_count)
        mode_l.addWidget(self.rb_igm_prop)
        igm_form.addRow("Mode:", mode_w)

        self.sb_igm1 = QSpinBox(); self.sb_igm1.setRange(0, 5000); self.sb_igm1.setValue(7)
        self.sb_igm2 = QSpinBox(); self.sb_igm2.setRange(0, 5000); self.sb_igm2.setValue(7)
        self.sb_igm3 = QSpinBox(); self.sb_igm3.setRange(0, 5000); self.sb_igm3.setValue(6)

        self.sb_igm1p = QDoubleSpinBox()
        self.sb_igm1p.setRange(0.0, 100.0); self.sb_igm1p.setValue(35.0)
        self.sb_igm1p.setSuffix(" %"); self.sb_igm1p.setVisible(False)
        self.sb_igm2p = QDoubleSpinBox()
        self.sb_igm2p.setRange(0.0, 100.0); self.sb_igm2p.setValue(35.0)
        self.sb_igm2p.setSuffix(" %"); self.sb_igm2p.setVisible(False)
        self.sb_igm3p = QDoubleSpinBox()
        self.sb_igm3p.setRange(0.0, 100.0); self.sb_igm3p.setValue(30.0)
        self.sb_igm3p.setSuffix(" %"); self.sb_igm3p.setVisible(False)

        self.sb_n_igm = QSpinBox()
        self.sb_n_igm.setRange(1, 10000); self.sb_n_igm.setValue(20)
        self.sb_n_igm.setVisible(False)

        igm_form.addRow("Type 1:", self.sb_igm1)
        igm_form.addRow("Type 2:", self.sb_igm2)
        igm_form.addRow("Type 3:", self.sb_igm3)
        igm_form.addRow("Type 1 %:", self.sb_igm1p)
        igm_form.addRow("Type 2 %:", self.sb_igm2p)
        igm_form.addRow("Type 3 % (auto):", self.sb_igm3p)
        igm_form.addRow("Total (prop. mode):", self.sb_n_igm)

        self.lbl_igm_total = QLabel("Total IgMs: 20")
        igm_form.addRow("", self.lbl_igm_total)

        form.addRow(igm_grp)

        def _refresh_igm_total():
            if self.rb_igm_count.isChecked():
                n = self.sb_igm1.value() + self.sb_igm2.value() + self.sb_igm3.value()
            else:
                n = self.sb_n_igm.value()
            self.lbl_igm_total.setText(f"Total IgMs: {n}")

        def _on_igm_mode_changed():
            count_mode = self.rb_igm_count.isChecked()
            for w in (self.sb_igm1, self.sb_igm2, self.sb_igm3):
                w.setVisible(count_mode)
            for w in (self.sb_igm1p, self.sb_igm2p, self.sb_igm3p, self.sb_n_igm):
                w.setVisible(not count_mode)
            _refresh_igm_total()

        def _on_prop_changed():
            p1 = self.sb_igm1p.value()
            p2 = self.sb_igm2p.value()
            p3 = max(0.0, 100.0 - p1 - p2)
            self.sb_igm3p.blockSignals(True)
            self.sb_igm3p.setValue(p3)
            self.sb_igm3p.blockSignals(False)
            _refresh_igm_total()

        self.rb_igm_count.toggled.connect(_on_igm_mode_changed)
        self.sb_igm1.valueChanged.connect(_refresh_igm_total)
        self.sb_igm2.valueChanged.connect(_refresh_igm_total)
        self.sb_igm3.valueChanged.connect(_refresh_igm_total)
        self.sb_igm1p.valueChanged.connect(_on_prop_changed)
        self.sb_igm2p.valueChanged.connect(_on_prop_changed)
        self.sb_n_igm.valueChanged.connect(_refresh_igm_total)

        # 4
        self.sb_svp_diam = QDoubleSpinBox()
        self.sb_svp_diam.setRange(5.0, 500.0)
        self.sb_svp_diam.setSingleStep(1.0)
        self.sb_svp_diam.setValue(19.0)
        form.addRow("SVP diameter (nm)", self.sb_svp_diam)

        # 5
        csv_w = QWidget()
        csv_l = QHBoxLayout(csv_w)
        csv_l.setContentsMargins(0, 0, 0, 0)
        self.le_csv = QLineEdit()
        self.le_csv.setReadOnly(True)
        self.le_csv.setPlaceholderText("(optional)")
        btn_csv = QPushButton("Browse...")
        btn_csv.clicked.connect(self._browse_csv)
        csv_l.addWidget(self.le_csv)
        csv_l.addWidget(btn_csv)
        form.addRow("SVP size CSV", csv_w)

        # 6
        self.sb_threshold = QDoubleSpinBox()
        self.sb_threshold.setRange(0.0, 200.0)
        self.sb_threshold.setSingleStep(0.5)
        self.sb_threshold.setValue(0.0)
        self.sb_threshold.setToolTip(
            "Optional extra margin added on top of the physical contact "
            "distance (hub + arm + blob reach, and SVP radius + spike). "
            "0 = exact physical touch, matching the preset geometry exactly."
        )
        form.addRow("Interaction threshold (nm)", self.sb_threshold)

        # 7
        self.le_border_conc = QLineEdit("1e-6")
        form.addRow("Border conc. (SVPs/nm³)", self.le_border_conc)

        # 8
        self.sb_border_thick = QDoubleSpinBox()
        self.sb_border_thick.setRange(0.0, 1000.0)
        self.sb_border_thick.setSingleStep(5.0)
        self.sb_border_thick.setValue(50.0)
        form.addRow("Border thickness (nm)", self.sb_border_thick)

        left_l.addWidget(self.grp_params)

        # ── Display Box (optional, purely cosmetic — never affects placement) ──
        self.disp_grp  = QGroupBox("Display Box (optional)")
        disp_form = QFormLayout(self.disp_grp)
        disp_form.setLabelAlignment(Qt.AlignRight)

        self.cb_outer_enabled = QCheckBox("Enable")
        disp_form.addRow("", self.cb_outer_enabled)

        self.sb_outer_x = QDoubleSpinBox()
        self.sb_outer_x.setRange(10.0, 5000.0)
        self.sb_outer_x.setSingleStep(10.0)
        self.sb_outer_x.setValue(400.0)
        disp_form.addRow("Display X (nm)", self.sb_outer_x)

        self.sb_outer_y = QDoubleSpinBox()
        self.sb_outer_y.setRange(10.0, 5000.0)
        self.sb_outer_y.setSingleStep(10.0)
        self.sb_outer_y.setValue(400.0)
        disp_form.addRow("Display Y (nm)", self.sb_outer_y)

        self.sb_outer_z = QDoubleSpinBox()
        self.sb_outer_z.setRange(10.0, 5000.0)
        self.sb_outer_z.setSingleStep(10.0)
        self.sb_outer_z.setValue(400.0)
        disp_form.addRow("Display Z (nm)", self.sb_outer_z)

        self.cmb_outer_mode = QComboBox()
        self.cmb_outer_mode.addItems(["Center", "Min corner", "Max corner", "Custom"])
        disp_form.addRow("Position", self.cmb_outer_mode)

        self.sb_outer_off_x = QDoubleSpinBox()
        self.sb_outer_off_x.setRange(-5000.0, 5000.0)
        self.sb_outer_off_y = QDoubleSpinBox()
        self.sb_outer_off_y.setRange(-5000.0, 5000.0)
        self.sb_outer_off_z = QDoubleSpinBox()
        self.sb_outer_off_z.setRange(-5000.0, 5000.0)
        for w in (self.sb_outer_off_x, self.sb_outer_off_y, self.sb_outer_off_z):
            w.setVisible(False)
        self.lbl_outer_off_x = QLabel("Custom offset X")
        self.lbl_outer_off_y = QLabel("Custom offset Y")
        self.lbl_outer_off_z = QLabel("Custom offset Z")
        self.lbl_outer_off_x.setVisible(False)
        self.lbl_outer_off_y.setVisible(False)
        self.lbl_outer_off_z.setVisible(False)
        disp_form.addRow(self.lbl_outer_off_x, self.sb_outer_off_x)
        disp_form.addRow(self.lbl_outer_off_y, self.sb_outer_off_y)
        disp_form.addRow(self.lbl_outer_off_z, self.sb_outer_off_z)

        def _on_outer_mode_changed():
            custom = self.cmb_outer_mode.currentText() == "Custom"
            for w in (self.sb_outer_off_x, self.sb_outer_off_y, self.sb_outer_off_z,
                      self.lbl_outer_off_x, self.lbl_outer_off_y, self.lbl_outer_off_z):
                w.setVisible(custom)
        self.cmb_outer_mode.currentTextChanged.connect(_on_outer_mode_changed)

        left_l.addWidget(self.disp_grp)

        # ── Bootstrap (null-model repeated-placement test) ─────────────────
        self.boot_grp  = QGroupBox("Bootstrap (Null-Model Test)")
        boot_form = QFormLayout(self.boot_grp)
        boot_form.setLabelAlignment(Qt.AlignRight)

        self.sb_boot_trials = QSpinBox()
        self.sb_boot_trials.setRange(2, 100000)
        self.sb_boot_trials.setValue(1000)
        boot_form.addRow("Number of trials", self.sb_boot_trials)

        self.btn_boot_run = QPushButton("▶ Run Bootstrap")
        self.btn_boot_run.setStyleSheet(
            "QPushButton{background-color:#7B1FA2;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#6A1B9A;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_boot_run.clicked.connect(self._run_bootstrap)
        boot_form.addRow("", self.btn_boot_run)

        self.btn_boot_stop = QPushButton("■ Stop Bootstrap")
        self.btn_boot_stop.setEnabled(False)
        self.btn_boot_stop.setStyleSheet(
            "QPushButton{background-color:#E53935;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#d32f2f;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_boot_stop.clicked.connect(self._stop_bootstrap)
        boot_form.addRow("", self.btn_boot_stop)

        self.boot_status_label = QLabel("")
        self.boot_status_label.setWordWrap(True)
        boot_form.addRow(self.boot_status_label)

        left_l.addWidget(self.boot_grp)

        # ── Dilution Comparison mode ────────────────────────────────────
        self.dilution_grp  = QGroupBox("Dilution Comparison")
        dilution_form = QFormLayout(self.dilution_grp)
        dilution_form.setLabelAlignment(Qt.AlignRight)

        self.cmb_dilution_factor = QComboBox()
        self.cmb_dilution_factor.addItems([f"1/{k}" for k in range(1, 21)])
        dilution_form.addRow("Dilution factor", self.cmb_dilution_factor)

        self.sb_dilution_trials_a = QSpinBox()
        self.sb_dilution_trials_a.setRange(2, 100000)   # needs >=2 to have a distribution (z-score spread)
        self.sb_dilution_trials_a.setValue(1000)
        dilution_form.addRow("Trials: same volume (A)", self.sb_dilution_trials_a)

        self.sb_dilution_trials_b = QSpinBox()
        self.sb_dilution_trials_b.setRange(1, 100000)   # can be 1 — e.g. to compare against one real tomogram
        self.sb_dilution_trials_b.setValue(1000)
        dilution_form.addRow("Trials: reduced volume (B)", self.sb_dilution_trials_b)

        self.btn_run_dilution = QPushButton("▶ Run Dilution Comparison")
        self.btn_run_dilution.setStyleSheet(
            "QPushButton{background-color:#00838F;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#006064;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_run_dilution.clicked.connect(self._run_dilution)
        dilution_form.addRow("", self.btn_run_dilution)

        self.btn_stop_dilution = QPushButton("■ Stop")
        self.btn_stop_dilution.setEnabled(False)
        self.btn_stop_dilution.setStyleSheet(
            "QPushButton{background-color:#E53935;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#d32f2f;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_stop_dilution.clicked.connect(self._stop_dilution)
        dilution_form.addRow("", self.btn_stop_dilution)

        self.dilution_status_label = QLabel(
            "Uses the Parameters group above as the original, undiluted "
            "configuration — dilution only changes particle counts and box "
            "X/Y; SVP diameter, threshold, border conc, etc. stay exactly "
            "as set there. Trials B can be set to 1 to compare against a "
            "single observed tomogram instead of a second simulated batch.\n"
            "z-score = how many standard deviations B's mean falls from A's "
            "distribution's mean (|z|>2 unusual, |z|>3 rare). "
            "χ²-p = chi-square test of whether A's and B's whole-number-bin "
            "count SHAPES differ (a separate question from whether their "
            "means differ).")
        self.dilution_status_label.setWordWrap(True)
        dilution_form.addRow(self.dilution_status_label)

        left_l.addWidget(self.dilution_grp)
        self.dilution_grp.setVisible(False)

        # ── STAR Exact + Randomize mode ─────────────────────────────────
        self.star_grp  = QGroupBox("STAR Exact + Randomize")
        star_form = QFormLayout(self.star_grp)
        star_form.setLabelAlignment(Qt.AlignRight)

        self.btn_render_star = QPushButton("🔬 Render Observed Positions")
        self.btn_render_star.setStyleSheet(
            "QPushButton{background-color:#5E35B1;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#4527A0;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_render_star.clicked.connect(self._render_star_exact)
        star_form.addRow("", self.btn_render_star)

        self.btn_randomize_star = QPushButton("🔀 Randomize")
        self.btn_randomize_star.setVisible(False)
        self.btn_randomize_star.setStyleSheet(
            "QPushButton{background-color:#F4511E;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#E64A19;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_randomize_star.clicked.connect(self._randomize_star_exact)
        star_form.addRow("", self.btn_randomize_star)

        self.cb_randomize_movie = QCheckBox("🎥 Save as movie (.mp4)")
        self.cb_randomize_movie.setVisible(False)
        self.cb_randomize_movie.setEnabled(PYVISTA_OK)
        star_form.addRow("", self.cb_randomize_movie)

        self.star_status_label = QLabel(
            "Load SVP/IgM STAR files above (\"Load from Experiment\"), then "
            "render their observed positions. SVP diameter and interaction "
            "threshold are read from the Parameters group above.")
        self.star_status_label.setWordWrap(True)
        star_form.addRow(self.star_status_label)

        left_l.addWidget(self.star_grp)
        self.star_grp.setVisible(False)

        # ── Real Data Comparison mode ────────────────────────────────────
        self.compare_grp = QGroupBox("Observed Data Comparison")
        compare_form = QFormLayout(self.compare_grp)
        compare_form.setLabelAlignment(Qt.AlignRight)

        self.sb_compare_trials_a = QSpinBox()
        self.sb_compare_trials_a.setRange(2, 100000)
        self.sb_compare_trials_a.setValue(1000)
        compare_form.addRow("Trials: simulated side", self.sb_compare_trials_a)

        self.btn_run_compare = QPushButton("▶ Run Comparison")
        self.btn_run_compare.setEnabled(False)
        self.btn_run_compare.setStyleSheet(
            "QPushButton{background-color:#00838F;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#006064;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_run_compare.clicked.connect(self._run_compare)
        compare_form.addRow("", self.btn_run_compare)

        self.btn_stop_compare = QPushButton("■ Stop")
        self.btn_stop_compare.setEnabled(False)
        self.btn_stop_compare.setStyleSheet(
            "QPushButton{background-color:#E53935;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#d32f2f;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_stop_compare.clicked.connect(self._stop_compare)
        compare_form.addRow("", self.btn_stop_compare)

        self.btn_compare_stats = QPushButton("ⓘ Mean comparison")
        self.btn_compare_stats.setEnabled(False)
        self.btn_compare_stats.setToolTip(
            "Mean, 95% interval, effect size and p-value for observed vs simulated "
            "(SVPs per IgM and IgMs per SVP), with how both were derived.")
        self.btn_compare_stats.clicked.connect(self._open_compare_stats_dialog)
        compare_form.addRow("", self.btn_compare_stats)

        self.cb_compare_no_border = QCheckBox("No border shell on simulated side (match observed data)")
        self.cb_compare_no_border.setChecked(True)
        self.cb_compare_no_border.setToolTip(
            "Observed STAR-file data has no border-shell concept (every SVP counts "
            "toward the IgMs-per-SVP denominator). Checking this forces the "
            "simulated side's border_conc to 0 so both sides' denominators match. "
            "Uncheck to keep whatever border conc./thickness is set above — then "
            "the comparison is inner-only SVPs (sim) vs all SVPs (observed).")
        compare_form.addRow("", self.cb_compare_no_border)

        self.compare_status_label = QLabel(
            "Browse to SVP/IgM STAR files above and \"Add Tomogram\" — repeat "
            "for as many observed tomograms as you want pooled together (each "
            "keeps its own box/particle-count, matched by its own simulated "
            "batch). Adjust SVP diameter / threshold / border conc. as "
            "desired, then \"Run Comparison\" to draw N independent simulated "
            "trials per tomogram and compare the pooled distributions "
            "against the pooled observations.\n"
            "z-score = how many std. devs the observed value falls from the simulated "
            "distribution's mean (|z|>2 unusual, |z|>3 rare). χ²-p = whether the "
            "observed per-particle bin-count SHAPE differs from the simulated one.")
        self.compare_status_label.setWordWrap(True)
        compare_form.addRow(self.compare_status_label)

        left_l.addWidget(self.compare_grp)
        self.compare_grp.setVisible(False)

        btn_bar = QWidget()
        btn_hl  = QHBoxLayout(btn_bar)
        btn_hl.setContentsMargins(0, 0, 0, 0)
        btn_hl.setSpacing(4)

        self.btn_run = QPushButton("▶ Run")
        self.btn_run.setStyleSheet(
            "QPushButton{background-color:#4CAF50;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#45a049;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_run.clicked.connect(self._run_simulation)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton{background-color:#F44336;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#d32f2f;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_stop.clicked.connect(self._stop_simulation)

        self.btn_refresh = QPushButton("↺ Refresh")
        self.btn_refresh.setStyleSheet(
            "QPushButton{background-color:#1976D2;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#1565C0;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_refresh.clicked.connect(self._refresh_view)

        self.btn_home = QPushButton("⌂ Home")
        self.btn_home.setStyleSheet(
            "QPushButton{background-color:#FF9800;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#F57C00;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_home.clicked.connect(lambda: self._reset_camera_home())

        self.btn_save = QPushButton("📷 Save")
        self.btn_save.setStyleSheet(
            "QPushButton{background-color:#00897B;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#00695C;}"
            "QPushButton:disabled{background-color:#aaa;}"
        )
        self.btn_save.clicked.connect(self._save_screenshots)

        self.btn_geom = QPushButton("⚙ Geometry Parameters")
        self.btn_geom.setStyleSheet(
            "QPushButton{background-color:#5E35B1;color:white;"
            "font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background-color:#4527A0;}"
        )
        self.btn_geom.clicked.connect(self._open_geometry_dialog)

        btn_hl.addWidget(self.btn_run)
        btn_hl.addWidget(self.btn_stop)
        btn_hl.addWidget(self.btn_refresh)
        btn_hl.addWidget(self.btn_home)
        btn_hl.addWidget(self.btn_save)
        btn_hl.addWidget(self.btn_geom)
        left_l.addWidget(btn_bar)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        left_l.addWidget(self.status_label)
        left_l.addStretch(1)

        scroll.setWidget(left_w)

        # ── right: vertical splitter ─────────────────────────────────────
        right_split = QSplitter(Qt.Vertical)

        if PYVISTA_OK:
            self.plotter = QtInteractor(self)
            self.plotter.set_background('white')
            self.plotter2 = QtInteractor(self)
            self.plotter2.set_background('white')

            left_viewer_w = QWidget()
            left_viewer_l = QVBoxLayout(left_viewer_w)
            left_viewer_l.setContentsMargins(0, 0, 0, 0)
            left_viewer_l.setSpacing(2)
            self.lbl_viewer_a = QLabel("")
            self.lbl_viewer_a.setWordWrap(True)
            self.lbl_viewer_a.setVisible(False)
            left_viewer_l.addWidget(self.lbl_viewer_a)
            left_viewer_l.addWidget(self.plotter, stretch=1)

            self.viewer_b_container = QWidget()
            right_viewer_l = QVBoxLayout(self.viewer_b_container)
            right_viewer_l.setContentsMargins(0, 0, 0, 0)
            right_viewer_l.setSpacing(2)
            self.lbl_viewer_b = QLabel("")
            self.lbl_viewer_b.setWordWrap(True)
            right_viewer_l.addWidget(self.lbl_viewer_b)
            right_viewer_l.addWidget(self.plotter2, stretch=1)

            self.viewer_split = QSplitter(Qt.Horizontal)
            self.viewer_split.addWidget(left_viewer_w)
            self.viewer_split.addWidget(self.viewer_b_container)
            self.viewer_b_container.setVisible(False)   # only shown in Dilution Comparison mode

            # Real Data Comparison mode only: pick which loaded tomogram
            # (and its own matched simulated sample) Viewer A/B show — hidden
            # otherwise, and hidden even in Compare mode until >1 tomogram's
            # result is available (see _on_compare_done).
            self.compare_viewer_toggle_w = QWidget()
            toggle_l = QHBoxLayout(self.compare_viewer_toggle_w)
            toggle_l.setContentsMargins(0, 0, 0, 4)
            toggle_l.addWidget(QLabel("Tomogram:"))
            self.cmb_compare_tomo_viewer = QComboBox()
            toggle_l.addWidget(self.cmb_compare_tomo_viewer, stretch=1)
            self.compare_viewer_toggle_w.setVisible(False)

            viewer_area_w = QWidget()
            viewer_area_l = QVBoxLayout(viewer_area_w)
            viewer_area_l.setContentsMargins(0, 0, 0, 0)
            viewer_area_l.setSpacing(2)
            viewer_area_l.addWidget(self.compare_viewer_toggle_w)
            viewer_area_l.addWidget(self.viewer_split, stretch=1)
            right_split.addWidget(viewer_area_w)
        else:
            self._fig3d = Figure(facecolor='white')
            self._ax3d  = self._fig3d.add_subplot(111, projection='3d')
            self._ax3d.set_facecolor('white')
            self._canvas3d = FigureCanvas(self._fig3d)
            right_split.addWidget(self._canvas3d)

        bot_split = QSplitter(Qt.Horizontal)

        self._fig_h1    = Figure(tight_layout=True)
        self._ax_h1     = self._fig_h1.add_subplot(111)
        self._canvas_h1 = FigureCanvas(self._fig_h1)

        h1_container = QWidget()
        h1_layout = QVBoxLayout(h1_container)
        h1_layout.setContentsMargins(0, 0, 0, 0)
        h1_layout.setSpacing(0)
        h1_header = QHBoxLayout()
        h1_header.addStretch(1)
        self.btn_h1_info = QPushButton("ⓘ")
        self.btn_h1_info.setFixedSize(22, 22)
        self.btn_h1_info.setToolTip("Network sub-analysis (SVPs per IgM)")
        self.btn_h1_info.setVisible(False)
        self.btn_h1_info.setEnabled(False)
        self.btn_h1_info.clicked.connect(self._open_h1_info_dialog)
        h1_header.addWidget(self.btn_h1_info)
        h1_layout.addLayout(h1_header)
        h1_layout.addWidget(self._canvas_h1)
        bot_split.addWidget(h1_container)

        self._fig_h2    = Figure(tight_layout=True)
        self._ax_h2     = self._fig_h2.add_subplot(111)
        self._canvas_h2 = FigureCanvas(self._fig_h2)
        bot_split.addWidget(self._canvas_h2)

        right_split.addWidget(bot_split)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

        # ── color legend below the splitter ──────────────────────────────
        legend_canvas = self._build_legend_canvas()

        right_container = QWidget()
        right_vl = QVBoxLayout(right_container)
        right_vl.setContentsMargins(0, 0, 0, 0)
        right_vl.setSpacing(0)
        right_vl.addWidget(right_split, stretch=1)
        right_vl.addWidget(legend_canvas)

        # ── main horizontal splitter — makes left panel resizable ─────────
        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(scroll)
        main_split.addWidget(right_container)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([300, 980])
        root.addWidget(main_split, stretch=1)
        self._on_mode_changed()

    def _on_mode_changed(self):
        mode = self.cmb_mode.currentText()
        is_manual   = mode == "Manual / Bootstrap"
        is_dilution = mode == "Dilution Comparison"
        is_star     = mode == "STAR Exact + Randomize"
        is_compare  = mode == "Observed Data Comparison"
        is_nn       = mode == "Nearest Neighbour Analysis"

        self.grp_params.setVisible(is_manual or is_dilution or is_compare or is_nn)
        self.disp_grp.setVisible(is_manual)
        self.boot_grp.setVisible(is_manual)
        self.btn_run.setVisible(is_manual)
        self.btn_stop.setVisible(is_manual)

        self.dilution_grp.setVisible(is_dilution)
        self.star_grp.setVisible(is_star)
        self.compare_grp.setVisible(is_compare)
        self.compare_tomo_loader_w.setVisible(is_compare)
        self.nn_grp.setVisible(is_nn)
        if PYVISTA_OK and not is_compare:
            self.compare_viewer_toggle_w.setVisible(False)

        self.btn_h1_info.setVisible(is_dilution or is_compare)
        self.btn_h1_info.setEnabled(
            (is_dilution and self._dilution_result is not None) or
            (is_compare and self._compare_result is not None))

        # Neither Nearest Neighbour nor Real Data Comparison mode locks the
        # shared box spinboxes — each loaded tomogram keeps its own box
        # internally (see _add_nn_tomogram / _add_compare_tomogram), since
        # multiple tomograms can have different sizes.

        if PYVISTA_OK:
            self.viewer_b_container.setVisible(is_dilution or is_compare or is_nn)
            self.lbl_viewer_a.setVisible(is_dilution or is_compare or is_nn)

    # ─────────────────────────────────────────────────────────────────────
    def _browse_hbv_star(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HBV / SVP STAR file", "",
            "STAR files (*.star);;All files (*)"
        )
        if path:
            self._hbv_star_path = path
            self.le_hbv_star.setText(os.path.basename(path))
            self._check_star_ready()

    def _browse_igm_star(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open IgM STAR file", "",
            "STAR files (*.star);;All files (*)"
        )
        if path:
            self._igm_star_path = path
            self.le_igm_star.setText(os.path.basename(path))
            self._check_star_ready()

    def _check_star_ready(self):
        self.btn_apply_star.setEnabled(
            bool(self._hbv_star_path) and bool(self._igm_star_path)
        )

    def _browse_nn_hbv_star(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HBV / SVP STAR file", "",
            "STAR files (*.star);;All files (*)"
        )
        if path:
            self._nn_hbv_star_path = path
            self.le_nn_hbv_star.setText(os.path.basename(path))

    def _browse_nn_igm_star(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open IgM STAR file", "",
            "STAR files (*.star);;All files (*)"
        )
        if path:
            self._nn_igm_star_path = path
            self.le_nn_igm_star.setText(os.path.basename(path))

    def _browse_compare_hbv_star(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HBV / SVP STAR file", "",
            "STAR files (*.star);;All files (*)"
        )
        if path:
            self._compare_hbv_star_path = path
            self.le_compare_hbv_star.setText(os.path.basename(path))

    def _browse_compare_igm_star(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open IgM STAR file", "",
            "STAR files (*.star);;All files (*)"
        )
        if path:
            self._compare_igm_star_path = path
            self.le_compare_igm_star.setText(os.path.basename(path))

    def _apply_star_params(self):
        try:
            p = extract_params_from_stars(self._hbv_star_path, self._igm_star_path)
        except Exception as exc:
            QMessageBox.critical(self, "STAR Load Error", str(exc))
            return

        filled:  List[str] = []
        skipped: List[str] = []

        def _set_float(sb: QDoubleSpinBox, val: Optional[float], label: str):
            if val is None:
                skipped.append(label)
                return
            clamped = max(sb.minimum(), min(sb.maximum(), val))
            sb.setValue(clamped)
            filled.append(f"{label}={clamped}")

        def _set_int(sb: QSpinBox, val: Optional[int], label: str):
            if val is None:
                skipped.append(label)
                return
            clamped = max(sb.minimum(), min(sb.maximum(), val))
            sb.setValue(clamped)
            filled.append(f"{label}={clamped}")

        _set_float(self.sb_box_x,     p['box_x'],        "Box X")
        _set_float(self.sb_box_y,     p['box_y'],        "Box Y")
        _set_float(self.sb_box_z,     p['box_z'],        "Box Z")
        _set_int  (self.sb_n_svp,     p['n_svp'],        "SVPs")
        _set_float(self.sb_svp_diam,  p['svp_diameter'], "SVP diam")
        _set_float(self.sb_threshold, p['threshold'],    "Threshold")

        # Apply IgM count to whichever mode is active
        n_igm = p.get('n_igm')
        if n_igm is not None:
            if self.rb_igm_count.isChecked():
                total = self.sb_igm1.value() + self.sb_igm2.value() + self.sb_igm3.value()
                if total > 0:
                    n1 = max(0, round(n_igm * self.sb_igm1.value() / total))
                    n2 = max(0, round(n_igm * self.sb_igm2.value() / total))
                    n3 = max(0, n_igm - n1 - n2)
                else:
                    n1, n2, n3 = n_igm, 0, 0
                self.sb_igm1.setValue(n1)
                self.sb_igm2.setValue(n2)
                self.sb_igm3.setValue(n3)
            else:
                clamped = max(self.sb_n_igm.minimum(),
                              min(self.sb_n_igm.maximum(), n_igm))
                self.sb_n_igm.setValue(clamped)
            filled.append(f"IgMs={n_igm}")
        else:
            skipped.append("IgMs")

        if p.get('border_conc') is not None:
            self.le_border_conc.setText(f"{p['border_conc']:.3e}")
            filled.append(f"Border conc={p['border_conc']:.3e}")
        else:
            skipped.append("Border conc")

        lines = [f"Loaded from STAR — set: {', '.join(filled)}."]
        if skipped:
            lines.append(f"Not auto-filled (set manually): {', '.join(skipped)}.")
        if p.get('warnings'):
            lines.extend(p['warnings'])
        self.status_label.setText("  ".join(lines))

    # ─────────────────────────────────────────────────────────────────────
    # STAR Exact + Randomize mode
    # ─────────────────────────────────────────────────────────────────────

    def _render_star_exact(self):
        if not self._hbv_star_path or not self._igm_star_path:
            QMessageBox.critical(self, "Input Error",
                "Load both SVP and IgM STAR files first (\"Load from Experiment\" above).")
            return
        try:
            real = extract_real_coords_from_stars(
                self._hbv_star_path, self._igm_star_path, self.sb_svp_diam.value())
        except Exception as exc:
            QMessageBox.critical(self, "STAR Load Error", str(exc))
            return

        self._star_real = real
        self._params = SimParams(
            box_x=real['box_x'], box_y=real['box_y'], box_z=real['box_z'],
            n_svp=real['n_svp'], n_igm=real['n_igm'],
            svp_diameter=self.sb_svp_diam.value(), threshold=self.sb_threshold.value(),
        )
        self._update_3d_placement((real['svp_positions'], real['svp_radii'],
                                   real['svp_is_border'], real['igm_positions'], real['igm_thetas']))
        result = compute_binding(real['svp_positions'], real['svp_radii'], real['svp_is_border'],
                                 real['igm_positions'], real['igm_thetas'], real['igm_subtypes'],
                                 self._params)
        self._result = result
        self._star_orig_result = result
        self._update_3d(result)
        ground_truth = _ground_truth_for_current_stars(
            self._igm_star_path, self._hbv_star_path, real['n_igm'])
        self._update_histograms(result, igm_ground_truth=ground_truth)
        n_bound = sum(len(s) > 0 for s in result.igm_bound_svps)
        self.star_status_label.setText(
            f"Observed positions: {real['n_svp']} SVPs, {real['n_igm']} IgMs  |  "
            f"IgMs bound: {n_bound}/{real['n_igm']}"
        )
        self.btn_randomize_star.setVisible(True)
        self.btn_randomize_star.setEnabled(True)
        self.cb_randomize_movie.setVisible(True)

    def _randomize_star_exact(self):
        if self._star_real is None or self._params is None:
            return
        n_svp = self._star_real['n_svp']
        n_igm = self._star_real['n_igm']
        random_params = SimParams(
            box_x=self._params.box_x, box_y=self._params.box_y, box_z=self._params.box_z,
            n_svp=n_svp, n_igm=n_igm, n_igm_type1=n_igm, n_igm_type2=0, n_igm_type3=0,
            svp_diameter=self.sb_svp_diam.value(), threshold=self.sb_threshold.value(),
            border_conc=0.0,   # no border shell — must match the real data's particle count exactly
        )
        svp_pos, svp_radii, svp_border, igm_pos, igm_thetas, igm_subtypes = \
            place_particles_interleaved(random_params)

        self._animation_start = (self._star_real['svp_positions'], self._star_real['igm_positions'])
        self._animation_end   = (svp_pos, igm_pos)
        self._animation_random_full = dict(
            svp_positions=svp_pos, svp_radii=svp_radii, svp_is_border=svp_border,
            igm_positions=igm_pos, igm_thetas=igm_thetas, igm_subtypes=igm_subtypes,
        )
        self._animation_frame = 0

        self._recording = False
        self._recording_path = None
        if PYVISTA_OK and self.cb_randomize_movie.isChecked():
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Randomization Movie", "randomization.mp4", "MP4 Files (*.mp4)")
            if path:
                # Save the true pre-randomization colored state as a
                # standalone SVG too, alongside the movie — BEFORE the movie
                # is opened, so the magnified capture never runs mid-recording.
                base = path[:-4] if path.lower().endswith('.mp4') else path
                try:
                    self.plotter.render()
                    _save_plotter_hires_svg(self.plotter, f"{base}_initial.svg")
                    self._begin_recording(path)
                    # Hold on the TRUE pre-randomization colors/positions first,
                    # before the grey-out below — the movie should open on the
                    # real starting state, not already grey. Must re-render
                    # before EVERY write_frame() call, not just once before the
                    # loop — otherwise the GL render window can stop being
                    # "current" between calls and write_frame() raises
                    # RenderWindowUnavailable.
                    for _ in range(30):
                        self.plotter.render()
                        self.plotter.write_frame()
                except Exception as exc:
                    self._abort_randomize(exc)
                    return

        if PYVISTA_OK:
            # Neutral grey for every particle while it's mid-flight — the
            # pre-randomization bound/unbound colors describe interactions
            # that no longer apply once positions start changing, and the
            # true post-randomization colors aren't known until the move
            # finishes and binding is recomputed (_on_randomize_animation_done).
            grey = _hex_rgb(_SVP_COLOR_UNBOUND)
            for actor in self._svp_actors:
                if actor is not None:
                    actor.GetProperty().SetColor(*grey)
            for actor in list(self._igm_actors) + list(self._igm_arm_actors):
                if actor is not None:
                    actor.GetProperty().SetColor(*grey)
            self.plotter.render()

        self.btn_randomize_star.setEnabled(False)
        self.star_status_label.setText("Randomizing — animating particle positions ...")

        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_tick)
        self._animation_timer.start(33)   # ~30fps

    def _begin_recording(self, path: str):
        """Open the movie writer and freeze the 3D viewport (no mouse
        interaction, no resizing) until _finish_recording — every frame must
        have the same size, and interacting mid-recording is a crash path."""
        self.plotter.setFixedSize(self.plotter.size())
        self.plotter.setEnabled(False)
        self._recording = True
        self._recording_path = path
        self.plotter.open_movie(path, framerate=30)

    def _finish_recording(self):
        """Close the movie writer (if open) and unfreeze the viewport. Safe
        to call more than once."""
        if PYVISTA_OK:
            writer = getattr(self.plotter, 'mwriter', None)
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    traceback.print_exc()
                self.plotter.mwriter = None
            self.plotter.setMinimumSize(0, 0)
            self.plotter.setMaximumSize(16777215, 16777215)
            self.plotter.setEnabled(True)
        self._recording = False

    def _abort_randomize(self, exc: Exception):
        """Any error during randomize/recording: stop cleanly (timer off,
        movie writer closed) instead of letting the exception escape a Qt
        slot, which makes PyQt5 abort the whole program."""
        traceback.print_exc()
        if self._animation_timer is not None:
            self._animation_timer.stop()
        self._finish_recording()
        self.btn_randomize_star.setEnabled(True)
        self.star_status_label.setText(f"Randomize failed: {exc}")
        QMessageBox.critical(self, "Randomize Error", f"{type(exc).__name__}: {exc}")

    def _animation_tick(self):
        try:
            self._animation_tick_inner()
        except Exception as exc:
            self._abort_randomize(exc)

    def _animation_tick_inner(self):
        if not PYVISTA_OK:
            self._animation_timer.stop()
            self._on_randomize_animation_done()
            return
        self._animation_frame += 1
        t = min(1.0, self._animation_frame / self._animation_n_frames)
        start_svp, start_igm = self._animation_start
        end_svp, end_igm = self._animation_end
        self.plotter.suppress_rendering = True
        try:
            for i, actor in enumerate(self._svp_actors):
                if actor is not None and i < len(start_svp):
                    delta = t * (end_svp[i] - start_svp[i])
                    actor.SetPosition(float(delta[0]), float(delta[1]), float(delta[2]))
            for i, actor in enumerate(self._igm_actors):
                if actor is not None and i < len(start_igm):
                    delta = t * (end_igm[i] - start_igm[i])
                    actor.SetPosition(float(delta[0]), float(delta[1]), float(delta[2]))
                    # Bound IgMs are split into a hub actor (above) and a
                    # separate arms+tips actor — move both by the identical
                    # delta every frame so core+arms+tips translate together
                    # instead of the arms staying frozen until the mesh
                    # rebuild at the end of the tween.
                    if i < len(self._igm_arm_actors) and self._igm_arm_actors[i] is not None:
                        self._igm_arm_actors[i].SetPosition(
                            float(delta[0]), float(delta[1]), float(delta[2]))
        finally:
            self.plotter.suppress_rendering = False
        self.plotter.render()
        if self._recording:
            self.plotter.write_frame()
        if t >= 1.0:
            self._animation_timer.stop()
            self._on_randomize_animation_done()

    def _on_randomize_animation_done(self):
        try:
            self._on_randomize_animation_done_inner()
        except Exception as exc:
            self._abort_randomize(exc)

    def _on_randomize_animation_done_inner(self):
        rnd = self._animation_random_full
        result = compute_binding(rnd['svp_positions'], rnd['svp_radii'], rnd['svp_is_border'],
                                 rnd['igm_positions'], rnd['igm_thetas'], rnd['igm_subtypes'],
                                 self._params)
        n_igm = len(result.igm_positions)
        n_bound_rand = sum(len(s) > 0 for s in result.igm_bound_svps)
        n_bound_orig = (sum(len(s) > 0 for s in self._star_orig_result.igm_bound_svps)
                        if self._star_orig_result else 0)
        self._result = result
        self._update_3d(result)   # actors are already at their randomized positions — just recolor
        self._update_histograms(result)
        movie_note = ""
        if self._recording:
            base = (self._recording_path[:-4] if self._recording_path.lower().endswith('.mp4')
                    else self._recording_path)
            # Save the true recolored final state as a standalone SVG too,
            # alongside the movie.
            self.plotter.render()
            _save_plotter_hires_svg(self.plotter, f"{base}_final.svg")
            # Hold on the final, newly-colored interaction state for about a
            # second (at the same 30fps used during the tween) instead of
            # letting the movie end right as positions land, still grey.
            # Re-render before EVERY write_frame() call (see the matching
            # comment in _randomize_star_exact) — required to keep the GL
            # render window "current" across repeated captures.
            for _ in range(30):
                self.plotter.render()
                self.plotter.write_frame()
            self._finish_recording()
            movie_note = (f"  |  Movie saved to {self._recording_path}"
                         f"  |  Initial/final images saved to {base}_initial.svg/.png / {base}_final.svg/.png")
        self.star_status_label.setText(
            f"Original: {n_bound_orig}/{n_igm} IgMs bound  →  "
            f"Randomized: {n_bound_rand}/{n_igm} IgMs bound{movie_note}"
        )
        self.btn_randomize_star.setEnabled(True)

    # ─────────────────────────────────────────────────────────────────────
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SVP size CSV", "",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            diams = load_svp_csv(path)
            self._csv_path = path
            self.le_csv.setText(os.path.basename(path))
            self.status_label.setText(
                f"CSV loaded: {len(diams)} diameters "
                f"(mean={diams.mean():.1f}, std={diams.std():.1f} nm)"
            )
        except ValueError as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            self._csv_path = None
            self.le_csv.clear()

    def _gather_params(self) -> Optional[SimParams]:
        """Read every parameter widget and build a validated SimParams, or
        show an error dialog and return None. Shared by the single-run and
        bootstrap run paths so they can never disagree on what the current
        parameters mean."""
        try:
            border_conc = float(self.le_border_conc.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error",
                                 "Border conc. must be a valid float (e.g. 1e-6).")
            return None

        svp_diameters = None
        if self._csv_path:
            try:
                svp_diameters = load_svp_csv(self._csv_path)
            except ValueError as exc:
                QMessageBox.critical(self, "CSV Error", str(exc))
                return None

        if self.rb_igm_count.isChecked():
            n1 = self.sb_igm1.value()
            n2 = self.sb_igm2.value()
            n3 = self.sb_igm3.value()
        else:
            total_igm = self.sb_n_igm.value()
            p1 = self.sb_igm1p.value() / 100.0
            p2 = self.sb_igm2p.value() / 100.0
            p3 = max(0.0, 1.0 - p1 - p2)
            n1 = max(0, round(total_igm * p1))
            n2 = max(0, round(total_igm * p2))
            n3 = max(0, total_igm - n1 - n2)

        outer_mode_map = {"Center": "center", "Min corner": "min_corner",
                          "Max corner": "max_corner", "Custom": "custom"}

        params = SimParams(
            box_x         = self.sb_box_x.value(),
            box_y         = self.sb_box_y.value(),
            box_z         = self.sb_box_z.value(),
            n_svp         = self.sb_n_svp.value(),
            n_igm         = n1 + n2 + n3,
            n_igm_type1   = n1,
            n_igm_type2   = n2,
            n_igm_type3   = n3,
            svp_diameter  = self.sb_svp_diam.value(),
            svp_diameters = svp_diameters,
            threshold     = self.sb_threshold.value(),
            border_conc   = border_conc,
            border_thick  = self.sb_border_thick.value(),
            outer_box_enabled = self.cb_outer_enabled.isChecked(),
            outer_box_x   = self.sb_outer_x.value(),
            outer_box_y   = self.sb_outer_y.value(),
            outer_box_z   = self.sb_outer_z.value(),
            outer_box_mode = outer_mode_map[self.cmb_outer_mode.currentText()],
            outer_offset_x = self.sb_outer_off_x.value(),
            outer_offset_y = self.sb_outer_off_y.value(),
            outer_offset_z = self.sb_outer_off_z.value(),
        )

        if params.outer_box_enabled:
            if not (params.outer_box_x > params.box_x and
                    params.outer_box_y > params.box_y and
                    params.outer_box_z > params.box_z):
                QMessageBox.critical(self, "Input Error",
                    "Display box dimensions must be strictly larger than the "
                    "simulation box in all 3 axes.")
                return None
            if not outer_box_contains_inner(params):
                QMessageBox.critical(self, "Input Error",
                    "Display box offset does not fully contain the simulation "
                    "box; adjust the custom offset values.")
                return None

        return params

    def _run_simulation(self):
        params = self._gather_params()
        if params is None:
            return
        self._params = params
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_boot_run.setEnabled(False)
        self.btn_run_dilution.setEnabled(False)
        self.btn_run_compare.setEnabled(False)
        self.btn_run_nn.setEnabled(False)
        self.status_label.setText("Starting simulation ...")

        self._worker = SimulationWorker(params)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.placed.connect(self._on_placed)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _stop_simulation(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self.status_label.setText("Stopping...")

    def _run_bootstrap(self):
        params = self._gather_params()
        if params is None:
            return
        self._params = params
        n_trials = self.sb_boot_trials.value()
        self.btn_boot_run.setEnabled(False)
        self.btn_boot_stop.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.btn_run_dilution.setEnabled(False)
        self.btn_run_compare.setEnabled(False)
        self.btn_run_nn.setEnabled(False)
        self.boot_status_label.setText(f"Starting bootstrap ({n_trials} trials) ...")

        self._bootstrap_worker = BootstrapWorker(params, n_trials)
        self._bootstrap_worker.progress.connect(self.boot_status_label.setText)
        self._bootstrap_worker.finished.connect(self._on_bootstrap_done)
        self._bootstrap_worker.error.connect(self._on_bootstrap_error)
        self._bootstrap_worker.cancelled.connect(self._on_bootstrap_cancelled)
        self._bootstrap_worker.start()

    def _stop_bootstrap(self):
        if self._bootstrap_worker and self._bootstrap_worker.isRunning():
            self._bootstrap_worker.stop()
            self.btn_boot_stop.setEnabled(False)
            self.boot_status_label.setText("Stopping bootstrap...")

    def _on_bootstrap_done(self, result: 'BootstrapResult'):
        self._bootstrap_result = result
        self.btn_boot_run.setEnabled(True)
        self.btn_boot_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self._update_bootstrap_histograms(result)

        sample_note = ""
        if result.sample_placement is not None and result.sample_result is not None:
            # Show one actual trial's placed particles + binding outcome in
            # the 3D view so the bootstrap isn't just abstract statistics.
            self._update_3d_placement(result.sample_placement)
            self._result = result.sample_result
            self._update_3d(result.sample_result)
            sample_note = f"  |  showing trial {result.sample_trial_index}/{result.n_trials} in 3D view"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Bootstrap CSV", "bootstrap_results.csv", "CSV Files (*.csv)")
        if path:
            bootstrap_result_to_dataframe(result).to_csv(path, index=False)
            self.boot_status_label.setText(
                f"Bootstrap complete ({result.n_trials} trials) — saved to {path}{sample_note}")
        else:
            self.boot_status_label.setText(
                f"Bootstrap complete ({result.n_trials} trials) — not saved{sample_note}")

    def _on_bootstrap_error(self, msg: str):
        self.btn_boot_run.setEnabled(True)
        self.btn_boot_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.boot_status_label.setText(f"Bootstrap error: {msg}")
        QMessageBox.critical(self, "Bootstrap Error", msg)

    def _on_bootstrap_cancelled(self):
        self.btn_boot_run.setEnabled(True)
        self.btn_boot_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.boot_status_label.setText("Bootstrap cancelled.")

    # ─────────────────────────────────────────────────────────────────────
    # Dilution Comparison mode
    # ─────────────────────────────────────────────────────────────────────

    def _run_dilution(self):
        orig = self._gather_params()
        if orig is None:
            return
        k = int(self.cmb_dilution_factor.currentText().split('/')[1])
        n_trials_a = self.sb_dilution_trials_a.value()
        n_trials_b = self.sb_dilution_trials_b.value()
        params_a, params_b = derive_dilution_params(orig, k)

        self.btn_run_dilution.setEnabled(False)
        self.btn_stop_dilution.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.btn_boot_run.setEnabled(False)
        self.btn_run_compare.setEnabled(False)
        self.btn_run_nn.setEnabled(False)
        self.dilution_status_label.setText(
            f"Starting dilution comparison (1/{k}, A: {n_trials_a} trials, B: {n_trials_b} trials) ...")

        self._dilution_worker = DilutionWorker(params_a, params_b, n_trials_a, n_trials_b)
        self._dilution_worker.progress.connect(self.dilution_status_label.setText)
        self._dilution_worker.finished.connect(self._on_dilution_done)
        self._dilution_worker.error.connect(self._on_dilution_error)
        self._dilution_worker.cancelled.connect(self._on_dilution_cancelled)
        self._dilution_worker.start()

    def _stop_dilution(self):
        if self._dilution_worker and self._dilution_worker.isRunning():
            self._dilution_worker.stop()
            self.btn_stop_dilution.setEnabled(False)
            self.dilution_status_label.setText("Stopping dilution comparison...")

    def _on_dilution_done(self, result: 'DilutionResult'):
        self._dilution_result = result
        self._h1_info_source = 'dilution'
        self.btn_h1_info.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_stop_dilution.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self._update_dilution_histograms(result)

        self._params = result.params_a
        if result.sample_placement_a is not None:
            self._update_3d_placement(result.sample_placement_a, plotter=self.plotter,
                                      svp_actors_attr='_svp_actors', igm_actors_attr='_igm_actors',
                                      params=result.params_a)
            self._update_3d(result.sample_result_a, plotter=self.plotter,
                            svp_actors_attr='_svp_actors', igm_actors_attr='_igm_actors')
        if PYVISTA_OK and result.sample_placement_b is not None:
            self._update_3d_placement(result.sample_placement_b, plotter=self.plotter2,
                                      svp_actors_attr='_svp_actors2', igm_actors_attr='_igm_actors2',
                                      params=result.params_b)
            self._update_3d(result.sample_result_b, plotter=self.plotter2,
                            svp_actors_attr='_svp_actors2', igm_actors_attr='_igm_actors2')

        if PYVISTA_OK:
            self.lbl_viewer_a.setText(scenario_summary_text(
                "A", "Original volume, diluted concentration", result.params_a))
            self.lbl_viewer_b.setText(scenario_summary_text(
                "B", "Same dilution, smaller volume", result.params_b))

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Dilution Comparison CSV", "dilution_results.csv", "CSV Files (*.csv)")
        saved_note = ""
        if path:
            dilution_result_to_dataframe(result).to_csv(path, index=False)
            saved_note = f"  |  saved to {path}"
        b_trial_word = "trial" if result.n_trials_b == 1 else "trials"
        self.dilution_status_label.setText(
            f"Done (A: {result.n_trials_a} trials, B: {result.n_trials_b} {b_trial_word}).  "
            f"z(SVPs/IgM)={result.z1:.2f}{z_tag(result.z1)} (zp={result.zpvalue1:.3g})  "
            f"z(IgMs/SVP)={result.z2:.2f}{z_tag(result.z2)} (zp={result.zpvalue2:.3g})  |  "
            f"χ²-p(SVPs/IgM)={result.chi2_pvalue1:.3g}  "
            f"χ²-p(IgMs/SVP)={result.chi2_pvalue2:.3g}{saved_note}"
        )

    def _on_dilution_error(self, msg: str):
        self.btn_run_dilution.setEnabled(True)
        self.btn_stop_dilution.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.dilution_status_label.setText(f"Dilution comparison error: {msg}")
        QMessageBox.critical(self, "Dilution Comparison Error", msg)

    def _on_dilution_cancelled(self):
        self.btn_run_dilution.setEnabled(True)
        self.btn_stop_dilution.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.dilution_status_label.setText("Dilution comparison cancelled.")

    # Every histogram in the app uses these same 6 fixed categorical bins —
    # 0, 1, 2, 3, 4, and "5 or more" collapsed into one tail bucket — instead
    # of one bin per observed integer value.
    _HIST_BIN_LABELS = ['0', '1', '2', '3', '4', '≥5']

    @staticmethod
    def _rebin5(arr):
        """Collapse a whole-number-indexed array (index == count value) into
        the 6 fixed bins above, by summing everything from index 5 onward
        into the last bin."""
        arr = np.asarray(arr, dtype=float)
        out = np.zeros(6)
        n = min(len(arr), 5)
        out[:n] = arr[:n]
        if len(arr) > 5:
            out[5] = arr[5:].sum()
        return out

    @staticmethod
    def _pct(arr):
        """Normalize to percent of the array's own total (so each series —
        e.g. the 'A' bars vs the 'B' bars in a comparison plot — sums to
        100% independently of the other series' sample size)."""
        arr = np.asarray(arr, dtype=float)
        total = arr.sum()
        return arr / total * 100.0 if total > 0 else np.zeros_like(arr)

    def _rebin5_pct_with_std(self, avg, std):
        """Rebin a mean-per-bin array (and its trial-to-trial std) to the 6
        fixed bins, then convert to percent-of-total. Variances (not stds)
        are summed across bins being collapsed together — correct for
        independent bins — then both avg and std are scaled by the same
        linear factor (1/total*100) so the error bars stay proportionate."""
        avg6 = self._rebin5(avg)
        std6 = np.sqrt(self._rebin5(np.asarray(std, dtype=float) ** 2))
        total = avg6.sum()
        if total <= 0:
            return avg6, std6
        scale = 100.0 / total
        return avg6 * scale, std6 * scale

    def _plot_binned_overlap(self, ax, avg_a, std_a, avg_b, std_b, z, zp, chi2_p,
                              xlabel, ylabel, label_a, label_b, bin_labels=None,
                              color_a='steelblue', color_b='darkorange'):
        """Grouped bar chart over the bins, A vs B, with error bars for
        trial-to-trial variability. Callers pass already rebinned,
        already percent-normalized arrays (see _rebin5_pct_with_std) — each
        series sums to 100% independently UNLESS the caller has cropped some
        bins out of both arrays after rebinning (e.g. the binding/network
        sub-analysis popup drops the '0' bin), in which case the remaining
        bars keep their original 'percent of the full set' meaning and won't
        sum to 100%  — that's intentional, not a bug. Title shows the
        z-score (is B's MEAN/pooled-count unusual relative to A's
        distribution?) and the chi-square homogeneity p-value (is the SHAPE
        of the two binned distributions different, over whichever bins were
        passed in?) — two different questions, not a single combined
        significance test. bin_labels defaults to the full 6-label set
        (self._HIST_BIN_LABELS) for the main charts; callers showing a
        cropped bin range pass the matching cropped label list. Shared by
        Dilution Comparison, Real Data Comparison, and the binding/network
        popup (where B may have zero error bars — a single observation has
        no spread — rendering as a solid bar next to A's bars-with-whiskers)."""
        if bin_labels is None:
            bin_labels = self._HIST_BIN_LABELS
        ax.cla()
        x = np.arange(len(avg_a))
        width = 0.4
        ax.bar(x - width / 2, avg_a, width, yerr=std_a, capsize=2,
              facecolor=mcolors.to_rgba(color_a, 0.35),
              edgecolor=color_a, linewidth=1.5, label=label_a)
        ax.bar(x + width / 2, avg_b, width, yerr=std_b, capsize=2,
              facecolor=mcolors.to_rgba(color_b, 0.35),
              edgecolor=color_b, linewidth=1.5, label=label_b)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_title(f"z(B vs A mean) = {z:.2f}{z_tag(z)}  (p={zp:.3g})    "
                    f"χ²-p(shape) = {chi2_p:.3g}", fontsize=9)
        ax.legend(fontsize=7)

    def _update_dilution_histograms(self, result: 'DilutionResult'):
        avg1_a, std1_a = self._rebin5_pct_with_std(result.bin_avg1_a, result.bin_std1_a)
        avg1_b, std1_b = self._rebin5_pct_with_std(result.bin_avg1_b, result.bin_std1_b)
        self._plot_binned_overlap(self._ax_h1, avg1_a, std1_a, avg1_b, std1_b,
                             result.z1, result.zpvalue1,
                             result.chi2_pvalue1, "SVPs per IgM", "% of IgMs (per trial)",
                             'A: original volume, diluted concentration',
                             'B: same dilution, smaller volume')
        self._fig_h1.tight_layout()
        self._canvas_h1.draw()

        avg2_a, std2_a = self._rebin5_pct_with_std(result.bin_avg2_a, result.bin_std2_a)
        avg2_b, std2_b = self._rebin5_pct_with_std(result.bin_avg2_b, result.bin_std2_b)
        self._plot_binned_overlap(self._ax_h2, avg2_a, std2_a, avg2_b, std2_b,
                             result.z2, result.zpvalue2,
                             result.chi2_pvalue2, "IgMs per SVP", "% of SVPs (per trial)",
                             'A: original volume, diluted concentration',
                             'B: same dilution, smaller volume')
        self._fig_h2.tight_layout()
        self._canvas_h2.draw()

    # ─────────────────────────────────────────────────────────────────────
    # Real Data Comparison mode
    # ─────────────────────────────────────────────────────────────────────

    def _add_compare_tomogram(self):
        """Load one real STAR-pair tomogram and add it to the pooled list —
        mirrors _add_nn_tomogram exactly: each tomogram gets its own
        box/particle-count-matched SimParams (so its later simulated batch
        occupies the identical volume with the identical counts), rather
        than sharing one locked box across every tomogram."""
        if not self._compare_hbv_star_path or not self._compare_igm_star_path:
            QMessageBox.critical(self, "Input Error",
                "Browse to both SVP and IgM STAR files above first.")
            return
        try:
            real = extract_real_coords_from_stars(
                self._compare_hbv_star_path, self._compare_igm_star_path, self.sb_svp_diam.value())
        except Exception as exc:
            QMessageBox.critical(self, "STAR Load Error", str(exc))
            return

        try:
            border_conc = float(self.le_border_conc.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error",
                                 "Border conc. must be a valid float (e.g. 1e-6).")
            return

        params = SimParams(
            box_x=real['box_x'], box_y=real['box_y'], box_z=real['box_z'],
            n_svp=real['n_svp'], n_igm=real['n_igm'],
            svp_diameter=self.sb_svp_diam.value(), threshold=self.sb_threshold.value(),
            border_conc=border_conc, border_thick=self.sb_border_thick.value())

        result = compute_binding(real['svp_positions'], real['svp_radii'], real['svp_is_border'],
                                 real['igm_positions'], real['igm_thetas'], real['igm_subtypes'],
                                 params)
        placement = (real['svp_positions'], real['svp_radii'], real['svp_is_border'],
                     real['igm_positions'], real['igm_thetas'])

        ground_truth = _ground_truth_for_current_stars(
            self._compare_igm_star_path, self._compare_hbv_star_path, real['n_igm'])
        data1 = (list(ground_truth) if ground_truth is not None
                 else [len(s) for s in result.igm_bound_svps])
        data2 = [len(result.svp_bound_igms[j]) for j in range(result.n_inner)]
        max_k1 = MAX_SVP_PER_IGM
        bin1 = (np.bincount(data1, minlength=max_k1 + 1)[:max_k1 + 1] if data1
                else np.zeros(max_k1 + 1))
        max_k2 = max(data2) if data2 else 0
        bin2 = np.bincount(data2, minlength=max_k2 + 1) if data2 else np.zeros(1)

        name = os.path.splitext(os.path.basename(self._compare_hbv_star_path))[0]
        data1_source = ('curated ground truth (lowcc_IgM.ods)' if ground_truth is not None
                        else 'geometric contact (threshold)')
        entry = dict(name=name, params=params, result=result, placement=placement,
                     data1=data1, data2=data2, bin1=bin1, bin2=bin2,
                     n_svp=real['n_svp'], n_igm=real['n_igm'], data1_source=data1_source)
        self._compare_tomograms.append(entry)
        self.lst_compare_tomograms.addItem(f"{name}  ({real['n_svp']} SVP / {real['n_igm']} IgM)")

        self.btn_run_compare.setEnabled(True)
        self.compare_status_label.setText(
            f"{len(self._compare_tomograms)} tomogram(s) loaded. Adjust Parameters above, "
            f"then Run Comparison.")

    def _remove_compare_tomogram(self):
        row = self.lst_compare_tomograms.currentRow()
        if row < 0:
            return
        self.lst_compare_tomograms.takeItem(row)
        del self._compare_tomograms[row]
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_remove_compare_tomogram.setEnabled(False)
        self.compare_status_label.setText(f"{len(self._compare_tomograms)} tomogram(s) loaded.")

    def _run_compare(self):
        if not self._compare_tomograms:
            QMessageBox.critical(self, "Input Error", "Add at least one observed tomogram first.")
            return
        n_trials_a = self.sb_compare_trials_a.value()

        self.btn_run_compare.setEnabled(False)
        self.btn_compare_stats.setEnabled(False)
        self.btn_stop_compare.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.btn_boot_run.setEnabled(False)
        self.btn_run_dilution.setEnabled(False)
        self.btn_run_nn.setEnabled(False)
        self.compare_status_label.setText(
            f"Starting comparison ({n_trials_a} simulated trials/tomogram × "
            f"{len(self._compare_tomograms)} tomogram(s)) ...")

        self._compare_worker = RealDataCompareWorker(
            n_trials_a, self._compare_tomograms, self.cb_compare_no_border.isChecked())
        self._compare_worker.progress.connect(self.compare_status_label.setText)
        self._compare_worker.finished.connect(self._on_compare_done)
        self._compare_worker.error.connect(self._on_compare_error)
        self._compare_worker.cancelled.connect(self._on_compare_cancelled)
        self._compare_worker.start()

    def _stop_compare(self):
        if self._compare_worker and self._compare_worker.isRunning():
            self._compare_worker.stop()
            self.btn_stop_compare.setEnabled(False)
            self.compare_status_label.setText("Stopping comparison...")

    def _show_compare_tomogram(self, entry: dict, n_trials_a: int):
        """Render one tomogram's real placement + its own matched simulated
        sample into Viewer A (sim)/Viewer B (real) — shared by the initial
        render in _on_compare_done and by _on_compare_tomo_selected when the
        user switches the "Tomogram:" dropdown."""
        if not PYVISTA_OK:
            return
        self._params = entry['sim_params']
        self._update_3d_placement(entry['sim_placement'], plotter=self.plotter,
                                  svp_actors_attr='_svp_actors', igm_actors_attr='_igm_actors',
                                  params=entry['sim_params'])
        self._update_3d(entry['sim_result'], plotter=self.plotter,
                        svp_actors_attr='_svp_actors', igm_actors_attr='_igm_actors')
        self.lbl_viewer_a.setText(compare_scenario_summary_text(
            "A", f"Simulated: {entry['name']} ({n_trials_a} trials)", entry['sim_params']))

        self._update_3d_placement(entry['real_placement'], plotter=self.plotter2,
                                  svp_actors_attr='_svp_actors2', igm_actors_attr='_igm_actors2',
                                  params=entry['real_params'])
        self._update_3d(entry['real_result'], plotter=self.plotter2,
                        svp_actors_attr='_svp_actors2', igm_actors_attr='_igm_actors2')
        n_bound = sum(len(s) > 0 for s in entry['real_result'].igm_bound_svps)
        self.lbl_viewer_b.setText(compare_scenario_summary_text(
            "B", f"Observed: {entry['name']}", entry['real_params'],
            n_extra=f"  |  IgMs bound: {n_bound}/{len(entry['real_result'].igm_positions)}"))

    def _on_compare_tomo_selected(self, idx: int):
        if (idx < 0 or self._compare_result is None or not self._compare_result.per_tomogram
                or idx >= len(self._compare_result.per_tomogram)):
            return
        self._show_compare_tomogram(self._compare_result.per_tomogram[idx],
                                    self._compare_result.n_trials_a)

    def _on_compare_done(self, result: 'RealDataCompareResult'):
        self._compare_result = result
        self.btn_compare_stats.setEnabled(True)
        self._h1_info_source = 'compare'
        self.btn_h1_info.setEnabled(True)
        self.btn_run_compare.setEnabled(True)
        self.btn_stop_compare.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self._update_compare_histograms(result)

        if PYVISTA_OK and result.per_tomogram:
            self.cmb_compare_tomo_viewer.blockSignals(True)
            self.cmb_compare_tomo_viewer.clear()
            self.cmb_compare_tomo_viewer.addItems([t['name'] for t in result.per_tomogram])
            self.cmb_compare_tomo_viewer.setCurrentIndex(len(result.per_tomogram) - 1)
            self.cmb_compare_tomo_viewer.blockSignals(False)
            try:
                self.cmb_compare_tomo_viewer.currentIndexChanged.disconnect()
            except TypeError:
                pass   # nothing connected yet on the first run
            self.cmb_compare_tomo_viewer.currentIndexChanged.connect(self._on_compare_tomo_selected)
            self.compare_viewer_toggle_w.setVisible(len(result.per_tomogram) > 1)
            # Default view: whichever tomogram's batch ran last (same
            # "sample" convention the old single-preview code used).
            self._show_compare_tomogram(result.per_tomogram[-1], result.n_trials_a)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison CSV", "observed_data_comparison.csv", "CSV Files (*.csv)")
        saved_note = ""
        if path:
            real_data_compare_result_to_dataframe(result).to_csv(path, index=False)
            saved_note = f"  |  saved to {path}"
        self.compare_status_label.setText(
            f"Done ({result.n_trials_a} simulated trials/tomogram × {len(self._compare_tomograms)} "
            f"tomogram(s): {result.n_svp_real} SVPs, {result.n_igm_real} IgMs).  "
            f"z(SVPs/IgM)={result.z1:.2f}{z_tag(result.z1)} (zp={result.zpvalue1:.3g})  "
            f"z(IgMs/SVP)={result.z2:.2f}{z_tag(result.z2)} (zp={result.zpvalue2:.3g})  |  "
            f"χ²-p(SVPs/IgM)={result.chi2_pvalue1:.3g}  "
            f"χ²-p(IgMs/SVP)={result.chi2_pvalue2:.3g}{saved_note}")

    def _on_compare_error(self, msg: str):
        self.btn_run_compare.setEnabled(True)
        self.btn_stop_compare.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.compare_status_label.setText(f"Comparison error: {msg}")
        QMessageBox.critical(self, "Observed Data Comparison Error", msg)

    def _on_compare_cancelled(self):
        self.btn_run_compare.setEnabled(True)
        self.btn_stop_compare.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.compare_status_label.setText("Comparison cancelled.")

    def _update_compare_histograms(self, result: 'RealDataCompareResult'):
        # Real (red, left) vs. simulated (grey, right) — a is real, b is simulated.
        n_tomo = len(self._compare_tomograms)
        real_label = f"Observed ({n_tomo} tomogram{'s' if n_tomo != 1 else ''}, pooled)"
        sim_label = f"Simulated ({result.n_trials_a} trials/tomogram, pooled)"
        avg1_sim, std1_sim = self._rebin5_pct_with_std(result.bin_avg1_a, result.bin_std1_a)
        avg1_real, std1_real = self._rebin5_pct_with_std(result.real_bin1, np.zeros_like(result.real_bin1))
        self._plot_binned_overlap(self._ax_h1, avg1_real, std1_real, avg1_sim, std1_sim,
                                  result.z1, result.zpvalue1, result.chi2_pvalue1,
                                  "SVPs per IgM", "% of IgMs",
                                  real_label, sim_label,
                                  color_a=REAL_COLOR, color_b=SIM_COLOR)
        self._fig_h1.tight_layout(); self._canvas_h1.draw()

        avg2_sim, std2_sim = self._rebin5_pct_with_std(result.bin_avg2_a, result.bin_std2_a)
        avg2_real, std2_real = self._rebin5_pct_with_std(result.real_bin2, np.zeros_like(result.real_bin2))
        self._plot_binned_overlap(self._ax_h2, avg2_real, std2_real, avg2_sim, std2_sim,
                                  result.z2, result.zpvalue2, result.chi2_pvalue2,
                                  "IgMs per SVP", "% of SVPs",
                                  real_label, sim_label,
                                  color_a=REAL_COLOR, color_b=SIM_COLOR)
        self._fig_h2.tight_layout(); self._canvas_h2.draw()

    def _open_compare_stats_dialog(self):
        if self._compare_result is not None:
            MeanComparisonDialog(self, self._compare_result).exec_()

    def _open_h1_info_dialog(self):
        if self._h1_info_source == 'dilution' and self._dilution_result is not None:
            r = self._dilution_result
            # Derived directly from the correct per-trial pooled >=k counts
            # (not by re-collapsing the 6-bin marginal avg/std, which would
            # overestimate the combined variance — see pct_from_pooled_ge).
            avg_a, std_a = pct_from_pooled_ge(r.pooled_ge2_a, r.params_a.n_igm)
            avg_b, std_b = pct_from_pooled_ge(r.pooled_ge2_b, r.params_b.n_igm)
            # Both sides are many simulated trials with different IgM
            # counts, so compare per-trial network PERCENTAGES with Welch's
            # t-test (unequal variances, unequal trial counts).
            pct_a = np.asarray(r.pooled_ge2_a, dtype=float) / max(r.params_a.n_igm, 1) * 100.0
            pct_b = np.asarray(r.pooled_ge2_b, dtype=float) / max(r.params_b.n_igm, 1) * 100.0
            p = float(scipy_stats.ttest_ind(pct_a, pct_b, equal_var=False).pvalue)
            dlg = NetworkInfoDialog(
                self, "Scenario A", "Scenario B",
                avg_a[1], std_a[1], avg_b[1], std_b[1], p, f"p = {p:.3g}",
                f"Welch's two-sample t-test on per-trial network % "
                f"(A: {len(pct_a)} trials, B: {len(pct_b)} trials), two-sided.")
            dlg.exec_()
        elif self._h1_info_source == 'compare' and self._compare_result is not None:
            r = self._compare_result
            # Same fix as the dilution branch above. The real side's single
            # pooled observation degrades to std=0 naturally (a length-1
            # array), now numerically consistent with r.z_ge1/r.z_ge2 (which
            # already use these same pooled_ge/ge_real values).
            avg_real, std_real = pct_from_pooled_ge(np.array([r.ge2_real]), r.n_igm_real)
            avg_sim, std_sim = pct_from_pooled_ge(r.pooled_ge2_a, r.n_igm_real)
            # Every tomogram's simulated batch is box/particle-count-matched
            # to that SAME tomogram (see _add_compare_tomogram), so the
            # per-trial simulated counts are a null distribution for the
            # single real count. A normal fit to that distribution (the
            # z-score test already computed by the worker) gives an exact
            # p-value even when the real count lies beyond every trial.
            n_sim = len(r.pooled_ge2_a)
            p = r.zp_ge2
            dlg = NetworkInfoDialog(
                self, "Observed", "Simulated",
                avg_real[1], std_real[1], avg_sim[1], std_sim[1], p, f"p = {p:.3g}",
                f"two-sided z-test: observed network count vs a normal distribution "
                f"fitted to the mean and SD of {n_sim} matched simulated trials "
                f"(z = {r.z_ge2:.2f}).",
                real_vs_sim=True)
            dlg.exec_()

    # ─────────────────────────────────────────────────────────────────────
    # Nearest Neighbour Analysis mode
    # ─────────────────────────────────────────────────────────────────────

    def _add_nn_tomogram(self):
        if not self._nn_hbv_star_path or not self._nn_igm_star_path:
            QMessageBox.critical(self, "Input Error",
                "Browse to both SVP and IgM STAR files above first.")
            return
        try:
            real = extract_real_coords_from_stars(
                self._nn_hbv_star_path, self._nn_igm_star_path, self.sb_svp_diam.value())
        except Exception as exc:
            QMessageBox.critical(self, "STAR Load Error", str(exc))
            return

        # Real STAR tomograms have no border-shell concept at all (every SVP
        # is "inner" — see extract_real_coords_from_stars), so the simulated
        # side must never generate border SVPs either: force these to 0
        # unconditionally, regardless of the Parameters panel's border
        # widgets (which are for the OTHER modes' from-scratch simulations).
        params = SimParams(
            box_x=real['box_x'], box_y=real['box_y'], box_z=real['box_z'],
            n_svp=real['n_svp'], n_igm=real['n_igm'],
            svp_diameter=self.sb_svp_diam.value(), threshold=self.sb_threshold.value(),
            border_conc=0.0, border_thick=0.0)
        result = compute_binding(real['svp_positions'], real['svp_radii'], real['svp_is_border'],
                                 real['igm_positions'], real['igm_thetas'], real['igm_subtypes'],
                                 params)
        placement = (real['svp_positions'], real['svp_radii'], real['svp_is_border'],
                     real['igm_positions'], real['igm_thetas'])
        # Real STAR data has no border-shell concept — svp_is_border is all-False
        # (see extract_real_coords_from_stars), so every real SVP is "inner".
        inner_svps = real['svp_positions'][~real['svp_is_border']]
        # nn_idx/dists feed the viewer's nearest-distance labels; the plotted
        # interaction pairs are computed at run time with the chosen radius
        # (NearestNeighbourWorker sets entry['pairs']).
        dists, nn_idx = nearest_neighbor_distances_and_indices(inner_svps, real['igm_positions'])

        name = os.path.splitext(os.path.basename(self._nn_hbv_star_path))[0]
        entry = dict(name=name, hbv_path=self._nn_hbv_star_path, igm_path=self._nn_igm_star_path,
                    real=real, params=params, result=result, placement=placement,
                    dists=dists, nn_idx=nn_idx)
        self._nn_tomograms.append(entry)
        self.lst_nn_tomograms.addItem(f"{name}  ({real['n_svp']} SVP / {real['n_igm']} IgM)")

        self.btn_run_nn.setEnabled(True)
        self.nn_status_label.setText(
            f"{len(self._nn_tomograms)} tomogram(s) loaded. Adjust Parameters above, then Run.")

    def _remove_nn_tomogram(self):
        row = self.lst_nn_tomograms.currentRow()
        if row < 0:
            return
        self.lst_nn_tomograms.takeItem(row)
        del self._nn_tomograms[row]
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.btn_remove_nn_tomogram.setEnabled(False)
        self.nn_status_label.setText(f"{len(self._nn_tomograms)} tomogram(s) loaded.")

    def _run_nn(self):
        if not self._nn_tomograms:
            QMessageBox.critical(self, "Input Error", "Add at least one tomogram first.")
            return
        params_a = self._gather_params()
        if params_a is None:
            return
        n_trials_a = self.sb_nn_trials_a.value()
        bin_width = self.sb_nn_bin_width.value()
        radius = self.sb_nn_radius.value()

        self.btn_run_nn.setEnabled(False)
        self.btn_stop_nn.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.btn_boot_run.setEnabled(False)
        self.btn_run_dilution.setEnabled(False)
        self.btn_run_compare.setEnabled(False)
        self.nn_status_label.setText(
            f"Starting analysis ({n_trials_a} simulated trials per tomogram × "
            f"{len(self._nn_tomograms)} tomogram(s)) ...")

        self._nn_worker = NearestNeighbourWorker(
            params_a, n_trials_a, self._nn_tomograms, bin_width, radius)
        self._nn_worker.progress.connect(self.nn_status_label.setText)
        self._nn_worker.finished.connect(self._on_nn_done)
        self._nn_worker.error.connect(self._on_nn_error)
        self._nn_worker.cancelled.connect(self._on_nn_cancelled)
        self._nn_worker.start()

    def _stop_nn(self):
        if self._nn_worker and self._nn_worker.isRunning():
            self._nn_worker.stop()
            self.btn_stop_nn.setEnabled(False)
            self.nn_status_label.setText("Stopping analysis...")

    def _on_nn_done(self, result: 'NearestNeighbourResult'):
        self._nn_result = result
        self.btn_run_nn.setEnabled(True)
        self.btn_stop_nn.setEnabled(False)
        self.btn_export_nn_raw.setEnabled(True)
        self.btn_open_nn_viewer.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self._update_nn_plot(result)

        self._params = result.sample_params_a or result.params_a
        if PYVISTA_OK and result.sample_placement_a is not None:
            self._update_3d_placement(result.sample_placement_a, plotter=self.plotter,
                                      svp_actors_attr='_svp_actors', igm_actors_attr='_igm_actors',
                                      params=result.sample_params_a)
            self._update_3d(result.sample_result_a, plotter=self.plotter,
                            svp_actors_attr='_svp_actors', igm_actors_attr='_igm_actors')
        if PYVISTA_OK and result.sample_params_a is not None:
            self.lbl_viewer_a.setText(compare_scenario_summary_text(
                "A", f"Simulated sample ({result.n_trials_a} trials/tomogram)",
                result.sample_params_a))

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Nearest Neighbour CSV", "nearest_neighbour_results.csv", "CSV Files (*.csv)")
        saved_note = ""
        if path:
            nn_result_to_dataframe(result).to_csv(path, index=False)
            saved_note = f"  |  saved to {path}"
        self.nn_status_label.setText(
            f"Done ({result.n_trials_a} simulated trials/tomogram × "
            f"{len(self._nn_tomograms)} tomogram(s): {result.n_svp_real} SVPs, "
            f"{result.n_igm_real} IgMs total).  "
            f"Mean interaction distance — observed: {result.mean_dist_real:.1f} nm, "
            f"simulated: {result.mean_dist_sim:.1f} nm  |  interactions — observed: "
            f"{result.real_counts.sum():.0f}, simulated: {result.sim_counts_avg.sum():.1f} "
            f"(radius {result.radius:g} nm){saved_note}")

    def _on_nn_error(self, msg: str):
        self.btn_run_nn.setEnabled(True)
        self.btn_stop_nn.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.nn_status_label.setText(f"Analysis error: {msg}")
        QMessageBox.critical(self, "Nearest Neighbour Analysis Error", msg)

    def _on_nn_cancelled(self):
        self.btn_run_nn.setEnabled(True)
        self.btn_stop_nn.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.nn_status_label.setText("Analysis cancelled.")

    def _export_nn_raw(self):
        if not self._nn_tomograms:
            QMessageBox.critical(self, "Input Error", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Raw Interaction Pairs", "nearest_neighbour_raw.csv", "CSV Files (*.csv)")
        if path:
            nn_raw_result_to_dataframe(self._nn_tomograms).to_csv(path, index=False)

    def _open_nn_viewer(self):
        if not self._nn_tomograms:
            return
        if not PYVISTA_OK:
            QMessageBox.critical(self, "3D Viewer Unavailable",
                "The interactive 3D viewer requires pyvista/pyvistaqt, which "
                "isn't available in this environment.")
            return
        radius = (self._nn_result.radius if self._nn_result is not None
                  else self.sb_nn_radius.value())
        dlg = NNViewerDialog(self, self._nn_tomograms, radius)
        try:
            dlg.exec_()
        finally:
            dlg.deleteLater()   # parented to self, so it would otherwise never be destroyed

    def _on_nn_plot_opts_changed(self):
        if self._nn_result is not None:
            self._update_nn_plot(self._nn_result)

    @staticmethod
    def _smooth_peak_curve(centers, values):
        """Fit a smooth spline through a histogram's per-bin peak values, for
        overlaying a trend curve on top of the raw bars. Degree drops
        gracefully for few bins, and the result is clipped at 0 since a
        spline can undershoot below zero between bins even when every
        input value is non-negative (these are bin counts)."""
        centers = np.asarray(centers, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(centers) < 2:
            return centers, values
        k = min(3, len(centers) - 1)
        fine_x = np.linspace(centers[0], centers[-1], 300)
        spline = make_interp_spline(centers, values, k=k)
        fine_y = np.clip(spline(fine_x), 0, None)
        return fine_x, fine_y

    @staticmethod
    def _rebin_for_curve(centers, values, group_size: int):
        """Group every `group_size` consecutive (already fine-binned) values
        together by averaging them (so the curve stays on the same per-bin
        scale as the bars) and averaging their centers — used only to feed the curve fit
        below with less noisy, wider bins; never changes the displayed bars.
        group_size=1 is a no-op."""
        centers = np.asarray(centers, dtype=float)
        values = np.asarray(values, dtype=float)
        if group_size <= 1 or len(values) == 0:
            return centers, values
        n_groups = int(np.ceil(len(values) / group_size))
        coarse_centers = np.zeros(n_groups)
        coarse_values = np.zeros(n_groups)
        for i in range(n_groups):
            sl = slice(i * group_size, min((i + 1) * group_size, len(values)))
            coarse_values[i] = values[sl].mean()
            coarse_centers[i] = centers[sl].mean()
        return coarse_centers, coarse_values

    @staticmethod
    def _fit_envelope_curve(centers, values, x_max=None):
        """Curve that touches every early local peak on the way up (via the
        running maximum of the bin values) then smoothly declines after the
        highest one. Unlike a parametric fit (which can only ever touch ONE
        point exactly), this hugs several separate tall early bins at once —
        needed because real NN-distance histograms often have multiple
        near-equal peaks in the first few bins, not one to describe with a
        single formula. `centers`/`values` may already be coarser than the
        displayed bars (see _rebin_for_curve) — `x_max` then lets the drawn
        curve still span the full original x-range rather than stopping at
        the last (possibly partial) coarse group's center. Returns
        (fine_x, fine_y, peak_y, peak_x), with peak_y/peak_x set to None if
        there's not enough data to draw a curve at all (degenerate/too-few
        bins)."""
        centers = np.asarray(centers, dtype=float)
        values = np.asarray(values, dtype=float)
        x_max = centers[-1] if x_max is None else x_max
        fine_x = np.linspace(0.0, x_max, 300) if len(centers) else np.zeros(0)
        if len(centers) < 2 or values.sum() <= 0:
            return fine_x, np.zeros_like(fine_x), None, None

        peak_idx = int(np.argmax(values))
        envelope = values.copy()
        envelope[:peak_idx + 1] = np.maximum.accumulate(values[:peak_idx + 1])

        raw_fine_x, raw_fine_y = MainWindow._smooth_peak_curve(centers, envelope)
        # Re-map onto the full [0, x_max] grid — `centers` may be coarser
        # than the display bars (see _rebin_for_curve), so the spline's own
        # native range can fall short of the full plotted x-axis. Flat
        # extrapolation at both ends is safe here since it can't violate the
        # monotonicity enforced just below.
        fine_y = np.interp(fine_x, raw_fine_x, raw_fine_y, left=raw_fine_y[0], right=raw_fine_y[-1])
        # Safety net against spline overshoot/ringing between points — keep
        # the curve strictly non-decreasing up to its peak and non-increasing
        # after, so it always reads as one clean rise-then-fall.
        peak_pos = int(np.argmax(fine_y))
        fine_y[:peak_pos + 1] = np.maximum.accumulate(fine_y[:peak_pos + 1])
        fine_y[peak_pos:] = np.minimum.accumulate(fine_y[peak_pos:])
        fine_y = np.clip(fine_y, 0, None)
        return fine_x, fine_y, float(values[peak_idx]), float(centers[peak_idx])

    def _plot_nn_lines(self, ax, bin_edges, real_vals, sim_vals, mean_real, mean_sim,
                       n_trials, xmax=None, radius=NN_DEFAULT_RADIUS_NM):
        ax.cla()
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]
        # Filled everywhere, but bins within the interaction zone (<= the
        # interaction distance) dominate with a stronger fill than bins
        # outside it.
        below = centers <= NN_INTERACTION_DISTANCE_NM
        real_fc = [mcolors.to_rgba(REAL_COLOR, 0.45 if b else 0.15) for b in below]
        sim_fc  = [mcolors.to_rgba(SIM_COLOR, 0.35 if b else 0.12) for b in below]
        ax.bar(centers, real_vals, width=bin_width, color=real_fc,
              edgecolor=REAL_COLOR, linewidth=1.2, label='Observed tomogram(s)', zorder=3)
        ax.bar(centers, sim_vals, width=bin_width, color=sim_fc,
              edgecolor=SIM_COLOR, linewidth=1.2,
              label=f'Simulated ({n_trials} trials)', zorder=3)
        show_curve = self.cb_nn_show_curve.isChecked()
        peak_y = peak_x = None
        if show_curve:
            coarse_centers, coarse_real = self._rebin_for_curve(
                centers, real_vals, self.sb_nn_curve_smooth.value())
            real_x, real_y, peak_y, peak_x = self._fit_envelope_curve(
                coarse_centers, coarse_real, x_max=bin_edges[-1])
            ax.plot(real_x, real_y, color=REAL_COLOR, linewidth=2, zorder=2)
        sim_x, sim_y = self._smooth_peak_curve(centers, sim_vals)
        ax.plot(sim_x, sim_y, color=SIM_COLOR, linewidth=2, zorder=2)
        ax.axvline(NN_INTERACTION_DISTANCE_NM, color='gray', linestyle='--', linewidth=1.5,
                  label=f'Interaction line ({NN_INTERACTION_DISTANCE_NM:.1f} nm)')
        ax.set_xlabel("SVP–IgM distance (nm)")
        ax.set_ylabel("% of interactions")
        pairs_label = f"NN + ≤ {radius:g} nm pairs"
        ax.set_title(f"{pairs_label}: observed mean={mean_real:.1f} nm, simulated mean={mean_sim:.1f} nm",
                     fontsize=9)
        if xmax:
            # View-only truncation: crop the x-range and rescale y to fit
            # whatever's still visible — underlying bin data is untouched.
            ax.set_xlim(0, xmax)
            visible = centers <= xmax
            y_max = max(real_vals[visible].max() if visible.any() else 0.0,
                       sim_vals[visible].max() if visible.any() else 0.0, 1.0)
            ax.set_ylim(0, y_max * 1.1)
            ax.set_xticks(np.arange(50, xmax + 1, 50))
        else:
            # Fixed tick spacing (50, 100, 150, ...) independent of the user-set
            # bin width, so the axis reads consistently no matter how the
            # data is binned.
            full_max = bin_edges[-1]
            ax.set_xticks(np.arange(50, full_max + 1, 50))
        ax.legend(fontsize=8)
        return peak_y, peak_x

    def _update_nn_plot(self, result: 'NearestNeighbourResult'):
        # Each side as % of its own total interactions (result keeps raw counts).
        peak_y, peak_x = self._plot_nn_lines(
            self._ax_h1, result.bin_edges,
            pct_of_total(result.real_counts), pct_of_total(result.sim_counts_avg),
            result.mean_dist_real, result.mean_dist_sim, result.n_trials_a,
            xmax=self.sb_nn_xmax.value(), radius=result.radius)
        self._fig_h1.tight_layout(); self._canvas_h1.draw()
        # Second canvas stays blank in this mode — only one distance-distribution plot.
        self._ax_h2.cla()
        self._canvas_h2.draw()
        if peak_y is not None:
            self.lbl_nn_fit.setText(
                "Observed-data curve: envelope through the early peaks "
                f"(highest: {peak_y:.2f}% at {peak_x:.1f} nm), smoothed decline after.")
        elif not self.cb_nn_show_curve.isChecked():
            self.lbl_nn_fit.setText("Observed-data curve hidden — showing bars only.")
        else:
            self.lbl_nn_fit.setText(
                "Not enough bins to draw a curve — showing bars only.")

    def _refresh_view(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        if self._bootstrap_worker and self._bootstrap_worker.isRunning():
            self._bootstrap_worker.stop()
        if self._dilution_worker and self._dilution_worker.isRunning():
            self._dilution_worker.stop()
        if self._compare_worker and self._compare_worker.isRunning():
            self._compare_worker.stop()
        if self._nn_worker and self._nn_worker.isRunning():
            self._nn_worker.stop()
        if self._animation_timer is not None and self._animation_timer.isActive():
            self._animation_timer.stop()
        self._finish_recording()
        self._recording_path = None
        self._result = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_randomize_star.setEnabled(True)
        self._compare_tomograms = []
        self.lst_compare_tomograms.clear()
        self.btn_remove_compare_tomogram.setEnabled(False)
        self.btn_run_compare.setEnabled(False)
        self.btn_compare_stats.setEnabled(False)
        if PYVISTA_OK:
            self.cmb_compare_tomo_viewer.clear()
            self.compare_viewer_toggle_w.setVisible(False)
        self._nn_tomograms = []
        self.lst_nn_tomograms.clear()
        self.btn_remove_nn_tomogram.setEnabled(False)
        self.btn_run_nn.setEnabled(False)
        self._nn_result = None
        self.btn_export_nn_raw.setEnabled(False)
        self.btn_open_nn_viewer.setEnabled(False)
        self._h1_info_source = None
        self.btn_h1_info.setEnabled(False)
        if PYVISTA_OK:
            self.plotter.clear()
            self.plotter.render()
            self.plotter2.clear()
            self.plotter2.render()
        else:
            self._ax3d.cla()
            self._ax3d.set_facecolor('white')
            self._canvas3d.draw()
        self._ax_h1.cla()
        self._canvas_h1.draw()
        self._ax_h2.cla()
        self._canvas_h2.draw()
        self.status_label.setText("Ready.")

    def _save_screenshots(self):
        from PyQt5.QtWidgets import QFileDialog
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save Screenshots", "simulation",
            "PNG Files (*.png);;SVG Files (*.svg)")
        if not path:
            return
        ext = _pick_save_ext(path, chosen_filter)
        base = path[:-4] if path.lower().endswith(('.png', '.svg')) else path
        saved = []
        if PYVISTA_OK:
            if ext == 'svg':
                # High-res shaded render wrapped in an .svg (plus a .png) —
                # see _save_plotter_hires_svg.
                p3d = base + '_3d.svg'
                _save_plotter_hires_svg(self.plotter, p3d)
                saved.append('3d')
                if self.plotter2.isVisible():
                    p3d_b = base + '_3d_b.svg'
                    _save_plotter_hires_svg(self.plotter2, p3d_b)
                    saved.append('3d_b')
            else:
                p3d = base + '_3d.png'
                self.plotter.screenshot(p3d)
                saved.append('3d')
                if self.plotter2.isVisible():
                    p3d_b = base + '_3d_b.png'
                    self.plotter2.screenshot(p3d_b)
                    saved.append('3d_b')
        ph1 = f"{base}_hist_svp.{ext}"
        ph2 = f"{base}_hist_igm.{ext}"
        savefig_kwargs = dict(bbox_inches='tight', facecolor=self._fig_h1.get_facecolor())
        if ext == 'png':
            savefig_kwargs['dpi'] = 150
        self._fig_h1.savefig(ph1, format=ext, **savefig_kwargs)
        savefig_kwargs['facecolor'] = self._fig_h2.get_facecolor()
        self._fig_h2.savefig(ph2, format=ext, **savefig_kwargs)
        saved += ['hist_svp', 'hist_igm']
        self.status_label.setText(f"Saved: {', '.join(saved)}  →  {base}_*.{ext}")

    def closeEvent(self, event):
        """Shut everything down in order before Qt tears the window apart:
        background workers, the randomize animation/movie writer, then the
        VTK render windows (destroying them implicitly at exit segfaults)."""
        for attr in ('_worker', '_bootstrap_worker', '_dilution_worker',
                     '_compare_worker', '_nn_worker'):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                # Wait until it has really finished: stop() is honored at the
                # next completed trial (a few seconds at most), and Qt aborts
                # the program if a QThread is still running at exit.
                worker.stop()
                worker.wait()
        if self._animation_timer is not None:
            self._animation_timer.stop()
        self._finish_recording()
        if PYVISTA_OK:
            for plotter in (self.plotter, self.plotter2):
                if plotter is not None:
                    plotter.close()
        super().closeEvent(event)

    def _open_geometry_dialog(self):
        """Reachable regardless of the current cmb_mode — the button lives
        on the always-visible button bar, not inside any one mode's panel."""
        dlg = GeometryParametersDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.status_label.setText(
                "Geometry parameters applied — click Run to use the new geometry."
            )

    def _on_placed(self, data):
        """Phase 1: show both SVPs (blue) and IgMs (purple) placed together."""
        self.status_label.setText("Placement done — computing binding ...")
        self._update_3d_placement(data)

    def _update_3d_placement(self, data, plotter=None, svp_actors_attr='_svp_actors',
                             igm_actors_attr='_igm_actors', params=None):
        """Render placement preview: interleave SVPs and IgMs by index so both appear together.
        plotter/svp_actors_attr/igm_actors_attr/params default to the primary
        viewport's attributes, so existing single-viewport call sites are
        unaffected; a second viewport (dilution comparison) passes explicit
        overrides (plotter2, '_svp_actors2', '_igm_actors2', params_b)."""
        if not PYVISTA_OK:
            return
        plotter = plotter if plotter is not None else self.plotter
        params  = params  if params  is not None else self._params
        svp_positions, svp_radii, _svp_border, igm_positions, igm_thetas = data
        plotter.clear()
        Lx, Ly, Lz = params.box_x, params.box_y, params.box_z
        box_mesh = pv.Box(bounds=(0, Lx, 0, Ly, 0, Lz))
        plotter.add_mesh(box_mesh.extract_feature_edges(feature_angle=30),
                         color='#333333', line_width=2, name='inner_box')
        if params.outer_box_enabled:
            off_x, off_y, off_z = compute_outer_box_offset(params)
            Ox = params.outer_box_x
            Oy = params.outer_box_y
            Oz = params.outer_box_z
            outer_mesh = pv.Box(bounds=(off_x, off_x + Ox, off_y, off_y + Oy,
                                        off_z, off_z + Oz))
            plotter.add_mesh(outer_mesh.extract_feature_edges(feature_angle=30),
                             color='#FFC107', line_width=1, name='outer_box')
        n_svp = len(svp_positions)
        n_igm = len(igm_positions)
        svp_actors = [None] * n_svp
        igm_actors = [None] * n_igm
        for i in range(max(n_svp, n_igm)):
            if i < n_svp:
                svp_actors[i] = plotter.add_mesh(
                    build_svp_mesh(svp_positions[i], float(svp_radii[i])),
                    color=_SVP_COLOR_UNBOUND, smooth_shading=True)
            if i < n_igm:
                try:
                    igm_actors[i] = plotter.add_mesh(
                        build_igm_mesh(igm_positions[i], igm_thetas[i]),
                        color=_IGM_COLOR_UNBOUND, smooth_shading=True, opacity=0.85)
                except Exception:
                    pass
        setattr(self, svp_actors_attr, svp_actors)
        setattr(self, igm_actors_attr, igm_actors)
        # Parallel per-IgM "arms" actor list, populated only for particles
        # currently rendered bound (see _update_3d_pyvista) — derived name
        # keeps every existing call site's svp_actors_attr/igm_actors_attr
        # pair as the single source of truth, no new kwarg needed.
        setattr(self, igm_actors_attr.replace('_igm_actors', '_igm_arm_actors'), [None] * n_igm)
        self._reset_camera_home(plotter=plotter, params=params)

    def _on_done(self, result: SimResult):
        self._result = result
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        n_bound  = sum(len(s) > 0 for s in result.igm_bound_svps)
        n_border = len(result.svp_positions) - result.n_inner
        self.status_label.setText(
            f"Done.  Inner SVPs: {result.n_inner}  |  "
            f"Border SVPs: {n_border}  |  "
            f"IgMs bound: {n_bound}/{len(result.igm_positions)}"
        )
        self._update_3d(result)
        self._update_histograms(result)

    def _on_cancelled(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.status_label.setText("Simulation cancelled.")

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_boot_run.setEnabled(True)
        self.btn_run_dilution.setEnabled(True)
        self.btn_run_compare.setEnabled(bool(self._compare_tomograms))
        self.btn_run_nn.setEnabled(bool(self._nn_tomograms))
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Simulation Error", msg)

    # ─────────────────────────────────────────────────────────────────────
    # 3-D visualisation
    # ─────────────────────────────────────────────────────────────────────

    def _reset_camera_home(self, plotter=None, params=None):
        if not PYVISTA_OK:
            return
        plotter = plotter if plotter is not None else self.plotter
        params  = params  if params  is not None else self._params
        Lx = params.box_x if params else 500.0
        Ly = params.box_y if params else 500.0
        Lz = params.box_z if params else 500.0
        if params and params.outer_box_enabled:
            off_x, off_y, off_z = compute_outer_box_offset(params)
            Ox = params.outer_box_x
            Oy = params.outer_box_y
            Oz = params.outer_box_z
            # Union of inner and outer box extents — defensive against any
            # edge case where the outer box doesn't fully contain the inner
            # one, so Home framing never crops either box.
            min_x, max_x = min(0.0, off_x), max(Lx, off_x + Ox)
            min_y, max_y = min(0.0, off_y), max(Ly, off_y + Oy)
            min_z, max_z = min(0.0, off_z), max(Lz, off_z + Oz)
            cx, cy, cz = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
            d = max(max_x - min_x, max_y - min_y) * 1.8
        else:
            cx, cy, cz = Lx / 2, Ly / 2, Lz / 2
            d = max(Lx, Ly) * 1.8
        plotter.camera_position = [
            (cx, cy, cz + d),   # directly above
            (cx, cy, cz),       # look at box centre
            (0,  1,  0),        # Y points up on screen → X goes left-right
        ]
        plotter.render()

    def _update_3d(self, result: SimResult, plotter=None, svp_actors_attr='_svp_actors',
                   igm_actors_attr='_igm_actors'):
        if PYVISTA_OK:
            self._update_3d_pyvista(result, plotter=plotter, svp_actors_attr=svp_actors_attr,
                                    igm_actors_attr=igm_actors_attr)
        else:
            self._update_3d_mpl(result)

    def _update_3d_pyvista(self, result: SimResult, plotter=None, svp_actors_attr='_svp_actors',
                           igm_actors_attr='_igm_actors'):
        # Actors were already built during the placement phase and stored in
        # self._svp_actors / self._igm_actors (or the *_attr-named pair for a
        # second viewport). Unbound IgMs and all SVPs just get a VTK color
        # update — no mesh rebuild. Bound IgMs get their geometry REBUILT:
        # compute_binding() may have rewritten their theta so one arm points
        # at the SVP they're bound to, and that only takes effect if the
        # mesh is regenerated at the new theta (color alone can't show it).
        plotter = plotter if plotter is not None else self.plotter
        svp_actors = getattr(self, svp_actors_attr)
        igm_actors = getattr(self, igm_actors_attr)
        arm_actors_attr = igm_actors_attr.replace('_igm_actors', '_igm_arm_actors')
        igm_arm_actors = getattr(self, arm_actors_attr, None)
        if igm_arm_actors is None or len(igm_arm_actors) != len(igm_actors):
            igm_arm_actors = [None] * len(igm_actors)
        plotter.suppress_rendering = True
        try:
            for j, actor in enumerate(svp_actors):
                if actor is not None:
                    actor.GetProperty().SetColor(
                        *_hex_rgb(svp_color(len(result.svp_bound_igms[j]))))
            for i, actor in enumerate(igm_actors):
                if actor is None:
                    continue
                arm_actor = igm_arm_actors[i]
                if result.igm_bound_svps[i]:
                    # Newly (or still) bound: rebuild as two actors — pale
                    # hub + darker arms — so the two-tone scheme shows.
                    plotter.remove_actor(actor)
                    if arm_actor is not None:
                        plotter.remove_actor(arm_actor)
                    hub_mesh, arms_mesh = build_igm_mesh_split(
                        result.igm_positions[i], result.igm_thetas[i])
                    igm_actors[i] = plotter.add_mesh(
                        hub_mesh, color=_IGM_HUB_COLOR_BOUND, smooth_shading=True, opacity=0.85)
                    igm_arm_actors[i] = plotter.add_mesh(
                        arms_mesh, color=_IGM_ARM_COLOR_BOUND, smooth_shading=True, opacity=0.85)
                elif arm_actor is not None:
                    # Was bound (split), now unbound: drop the arm actor and
                    # rebuild the hub actor back into one flat grey mesh so a
                    # later re-bind doesn't have to special-case "hub-only".
                    plotter.remove_actor(arm_actor)
                    igm_arm_actors[i] = None
                    plotter.remove_actor(actor)
                    igm_actors[i] = plotter.add_mesh(
                        build_igm_mesh(result.igm_positions[i], result.igm_thetas[i]),
                        color=_IGM_COLOR_UNBOUND, smooth_shading=True, opacity=0.85)
                else:
                    # Still unbound: color-only update, no mesh rebuild.
                    actor.GetProperty().SetColor(*_hex_rgb(_IGM_COLOR_UNBOUND))
        finally:
            plotter.suppress_rendering = False
        setattr(self, arm_actors_attr, igm_arm_actors)
        plotter.render()

    def _update_3d_mpl(self, result: SimResult):
        self._ax3d.cla()
        self._ax3d.set_facecolor('white')
        self._ax3d.set_xlabel('X (nm)', color='black')
        self._ax3d.set_ylabel('Y (nm)', color='black')
        self._ax3d.set_zlabel('Z (nm)', color='black')
        Lx, Ly, Lz = self._params.box_x, self._params.box_y, self._params.box_z

        for j, (pos, r) in enumerate(zip(result.svp_positions,
                                          result.svp_radii)):
            self._ax3d.scatter(pos[0], pos[1], pos[2],
                               s=(r * 2) ** 2,
                               c=svp_color(len(result.svp_bound_igms[j])),
                               alpha=0.8)

        for i, pos in enumerate(result.igm_positions):
            # Single marker can't show a two-tone hub/arm split — use the
            # arm color (more visually dominant) for bound IgMs; intentional
            # simplification for this rarely-used non-pyvista fallback.
            color = (_IGM_ARM_COLOR_BOUND if len(result.igm_bound_svps[i]) > 0
                     else _IGM_COLOR_UNBOUND)
            self._ax3d.scatter(pos[0], pos[1], pos[2],
                               s=80, c=color, marker='^', alpha=0.9)

        self._ax3d.set_xlim(0, Lx)
        self._ax3d.set_ylim(0, Ly)
        self._ax3d.set_zlim(0, Lz)
        self._canvas3d.draw()

    # ─────────────────────────────────────────────────────────────────────
    # Histograms
    # ─────────────────────────────────────────────────────────────────────

    def _update_histograms(self, result: SimResult, igm_ground_truth: Optional[np.ndarray] = None):
        # Hist 1 — SVPs per IgM
        ax = self._ax_h1
        ax.cla()
        using_ground_truth = (igm_ground_truth is not None
                               and len(igm_ground_truth) == len(result.igm_bound_svps))
        data1   = (list(igm_ground_truth) if using_ground_truth
                   else [len(s) for s in result.igm_bound_svps])
        counts1 = np.zeros(6, dtype=int)
        for v in data1:
            counts1[min(v, 5)] += 1
        pct1  = self._pct(counts1)
        mean1 = float(np.mean(data1)) if data1 else 0.0
        std1  = float(np.std(data1))  if data1 else 0.0
        ax.bar(range(6), pct1, color='steelblue')
        ax.set_xlabel("SVPs per IgM")
        ax.set_ylabel("% of IgMs")
        ax.set_xticks(range(6))
        ax.set_xticklabels(self._HIST_BIN_LABELS)
        title_suffix = " [ground truth]" if using_ground_truth else ""
        ax.set_title(f"SVPs per IgM  [mean={mean1:.1f} \u00b1 {std1:.1f}]{title_suffix}")
        self._fig_h1.tight_layout()
        self._canvas_h1.draw()

        # Hist 2 — IgMs per inner SVP
        ax2 = self._ax_h2
        ax2.cla()
        n_inner = result.n_inner
        data2   = [len(result.svp_bound_igms[j]) for j in range(n_inner)]
        counts2 = np.zeros(6, dtype=int)
        for v in data2:
            counts2[min(v, 5)] += 1
        pct2  = self._pct(counts2)
        mean2 = float(np.mean(data2)) if data2 else 0.0
        std2  = float(np.std(data2))  if data2 else 0.0
        ax2.bar(range(6), pct2, color='salmon')
        ax2.set_xlabel("IgMs per SVP")
        ax2.set_ylabel("% of SVPs")
        ax2.set_xticks(range(6))
        ax2.set_xticklabels(self._HIST_BIN_LABELS)
        ax2.set_title(f"IgMs per SVP (inner)  [mean={mean2:.1f} \u00b1 {std2:.1f}]")
        self._fig_h2.tight_layout()
        self._canvas_h2.draw()

    def _update_bootstrap_histograms(self, result: 'BootstrapResult'):
        """Reuses the same two canvases as _update_histograms, but plots each
        whole-number bin's count AVERAGED across all bootstrap trials (e.g.
        "on average, 4.3 IgMs had exactly 2 SVPs bound"), with error bars for
        trial-to-trial variability — not the raw per-trial-mean distribution."""
        ax = self._ax_h1
        ax.cla()
        pct1, err1 = self._rebin5_pct_with_std(result.bin_avg1, result.bin_std1)
        ax.bar(range(6), pct1, yerr=err1, capsize=3, color='steelblue')
        ax.set_xlabel("SVPs per IgM")
        ax.set_ylabel("% of IgMs (per trial)")
        ax.set_xticks(range(6))
        ax.set_xticklabels(self._HIST_BIN_LABELS)
        grand1 = float(np.mean(result.trial_means1))
        lo1, hi1 = np.percentile(result.trial_means1, [2.5, 97.5])
        ax.set_title(f"Bootstrap SVPs/IgM  [grand mean={grand1:.2f}, "
                     f"95% CI=({lo1:.2f}, {hi1:.2f})]", fontsize=9)
        self._fig_h1.tight_layout()
        self._canvas_h1.draw()

        ax2 = self._ax_h2
        ax2.cla()
        pct2, err2 = self._rebin5_pct_with_std(result.bin_avg2, result.bin_std2)
        ax2.bar(range(6), pct2, yerr=err2, capsize=3, color='salmon')
        ax2.set_xlabel("IgMs per SVP")
        ax2.set_ylabel("% of SVPs (per trial)")
        ax2.set_xticks(range(6))
        ax2.set_xticklabels(self._HIST_BIN_LABELS)
        grand2 = float(np.mean(result.trial_means2))
        lo2, hi2 = np.percentile(result.trial_means2, [2.5, 97.5])
        ax2.set_title(f"Bootstrap IgMs/SVP  [grand mean={grand2:.2f}, "
                      f"95% CI=({lo2:.2f}, {hi2:.2f})]", fontsize=9)
        self._fig_h2.tight_layout()
        self._canvas_h2.draw()

    # ─────────────────────────────────────────────────────────────────────
    # Color legend
    # ─────────────────────────────────────────────────────────────────────

    def _build_legend_canvas(self):
        fig = Figure(figsize=(10, 0.85), facecolor='#f5f5f5')
        ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()

        handles = [
            mpatches.Patch(facecolor=_SVP_COLOR_UNBOUND, edgecolor='#555', label='SVP: unbound'),
            mpatches.Patch(facecolor=_SVP_COLOR_BOUND, edgecolor='#555', label='SVP: bound'),
            mpatches.Patch(facecolor=_IGM_COLOR_UNBOUND, edgecolor='#555', label='IgM: unbound'),
            mpatches.Patch(facecolor=_IGM_HUB_COLOR_BOUND, edgecolor='#555', label='IgM: bound (hub)'),
            mpatches.Patch(facecolor=_IGM_ARM_COLOR_BOUND, edgecolor='#555', label='IgM: bound (arms)'),
            mpatches.Patch(facecolor='#FFC107', edgecolor='#555', label='Display box (if enabled)'),
        ]

        ax.legend(
            handles=handles,
            loc='center',
            ncol=6,
            fontsize=9,
            frameon=True,
            framealpha=0.9,
            handlelength=1.6,
            handleheight=1.2,
            columnspacing=1.5,
            handletextpad=0.6,
            title='Color Key',
            title_fontsize=9,
        )

        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(75)
        return canvas


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

CRASH_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'igm_svp_sim_crash.log')


def _install_crash_handlers():
    """faulthandler: on a segfault, dump every thread's Python stack to the
    terminal (it supports a single output, and the terminal is where a crash
    is noticed). excepthook: an uncaught exception inside a Qt slot would
    make PyQt5 abort the program — instead print it, append it to
    CRASH_LOG_PATH, show it, and keep running."""
    faulthandler.enable(all_threads=True)
    try:
        log = open(CRASH_LOG_PATH, 'a', buffering=1)
    except OSError:
        log = None

    def _excepthook(exc_type, exc, tb):
        text = ''.join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        if log is not None:
            log.write(text)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Unexpected Error",
                                 f"{exc_type.__name__}: {exc}\n\n"
                                 f"Details were written to {CRASH_LOG_PATH}")
    sys.excepthook = _excepthook


if __name__ == '__main__':
    _install_crash_handlers()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    sys.exit(app.exec_())