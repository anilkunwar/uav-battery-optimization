# =============================================================================
# Streamlit App: FPV and other types of UAV Drones LiPo 3D Thermal Runaway
# Multi‑Simulation Platform
# =============================================================================
# UPGRADED VERSION 3.6.0 – Definitive 3D Camera Persistence
# =============================================================================
# - st.form for structural controls (n_slices, show_cross, slice_axis, etc.)
# - Native Plotly Frames for time slider (no Python rerun on time change)
# - Pre‑allocated constant traces (trace count never changes)
# - use_container_width=True for all plotly charts
# =============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
from numba import njit, prange
import time
import json
import zipfile
from io import BytesIO
from datetime import datetime
import pandas as pd
import hashlib
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

# Optional OS-level monitoring
try:
    import psutil
    import os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# -----------------------------------------------------------------------------
# 0. Custom CSS for bolder sliders (UI enhancement)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stSlider label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    .stSelectbox label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    .stNumberInput label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stAlert {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Configuration & Colormap Library (50+)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FPV LiPo Thermal Runaway Platform", layout="wide")
st.title("🔥 FPV LiPo 3D Thermal Runaway Multi‑Simulation Platform")
st.markdown("""
**Run multiple scenarios • Compare thermal responses • Cloud‑style storage**  
Run → Save → Compare • Publication‑ready figures • Advanced post‑processing  
*Upgraded v3.6.0 — Definitive 3D camera persistence*
""")

COLORMAPS = {
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
    'magma': 'magma', 'cividis': 'cividis', 'hot': 'hot', 'cool': 'cool',
    'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn',
    'winter': 'winter', 'copper': 'copper', 'bone': 'bone', 'gray': 'gray',
    'pink': 'pink', 'afmhot': 'afmhot', 'gist_heat': 'gist_heat',
    'gist_gray': 'gist_gray', 'binary': 'binary', 'coolwarm': 'coolwarm',
    'bwr': 'bwr', 'seismic': 'seismic', 'RdBu': 'RdBu', 'RdGy': 'RdGy',
    'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG', 'PuOr': 'PuOr',
    'twilight': 'twilight', 'twilight_shifted': 'twilight_shifted',
    'hsv': 'hsv', 'tab10': 'tab10', 'tab20': 'tab20', 'Set1': 'Set1',
    'Set2': 'Set2', 'Set3': 'Set3', 'Paired': 'Paired', 'Accent': 'Accent',
    'Dark2': 'Dark2', 'jet': 'jet', 'turbo': 'turbo', 'rainbow': 'rainbow',
    'nipy_spectral': 'nipy_spectral', 'gist_ncar': 'gist_ncar',
    'gist_rainbow': 'gist_rainbow', 'gist_earth': 'gist_earth',
    'gist_stern': 'gist_stern', 'ocean': 'ocean', 'terrain': 'terrain',
    'gnuplot': 'gnuplot', 'gnuplot2': 'gnuplot2', 'CMRmap': 'CMRmap',
    'cubehelix': 'cubehelix', 'brg': 'brg', 'rocket': 'rocket',
    'mako': 'mako', 'crest': 'crest', 'flare': 'flare', 'icefire': 'icefire',
    'vlag': 'vlag'
}
cmap_list = list(COLORMAPS.keys())

# -----------------------------------------------------------------------------
# 1.5 Drone Battery Models Database (based on standard drone classifications)
# -----------------------------------------------------------------------------
DRONE_BATTERY_MODELS = {
    "Model A: 3S 2200mAh 50C (Default FPV/RC)": {
        "S": 3, "V_nom": 11.1, "Cap_mAh": 2200, "C_rating": 50,
        "L_mm": 102, "W_mm": 34, "H_mm": 23, "Weight_g": 170,
        "kx": 25.0, "ky": 25.0, "kz": 1.5, "h_conv": 15.0
    },
    "Model B: 4S 1300mAh 100C (FPV Racing)": {
        "S": 4, "V_nom": 14.8, "Cap_mAh": 1300, "C_rating": 100,
        "L_mm": 75, "W_mm": 35, "H_mm": 25, "Weight_g": 160,
        "kx": 25.0, "ky": 25.0, "kz": 1.5, "h_conv": 25.0
    },
    "Model C: 6S 1300mAh 100C (Freestyle/Pro FPV)": {
        "S": 6, "V_nom": 22.2, "Cap_mAh": 1300, "C_rating": 100,
        "L_mm": 90, "W_mm": 40, "H_mm": 30, "Weight_g": 210,
        "kx": 25.0, "ky": 25.0, "kz": 1.5, "h_conv": 25.0
    },
    "Model D: 4S 5000mAh 25C (Photography Drone)": {
        "S": 4, "V_nom": 14.8, "Cap_mAh": 5000, "C_rating": 25,
        "L_mm": 150, "W_mm": 50, "H_mm": 30, "Weight_g": 520,
        "kx": 20.0, "ky": 20.0, "kz": 1.2, "h_conv": 10.0
    },
    "Model E: 8S 23000mAh 25C (Industrial/Heavy Lift)": {
        "S": 8, "V_nom": 29.6, "Cap_mAh": 23000, "C_rating": 25,
        "L_mm": 182, "W_mm": 92, "H_mm": 88, "Weight_g": 3292,
        "kx": 25.0, "ky": 25.0, "kz": 1.5, "h_conv": 10.0
    }
}

def matplotlib_to_plotly(cmap_name, pl_entries=11):
    try:
        cmap = plt.get_cmap(cmap_name)
    except:
        cmap = plt.get_cmap('viridis')
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = (np.array(cmap(k*h)[:3])*255).astype(np.int_)
        pl_colorscale.append([k*h, f'rgb({C[0]},{C[1]},{C[2]})'])
    return pl_colorscale


# -----------------------------------------------------------------------------
# Helper to resolve colour‑bar range (user‑defined or auto)
# -----------------------------------------------------------------------------
def resolve_cbar_range(style_params, data=None):
    if style_params.get('use_custom_cbar_range', False):
        vmin = float(style_params.get('cbar_t_min', 290.0))
        vmax = float(style_params.get('cbar_t_max', 900.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax
    if data is not None:
        return float(np.nanmin(data)), float(np.nanmax(data))
    return None, None


# -----------------------------------------------------------------------------
# 2. Journal & Styling Templates (unchanged)
# -----------------------------------------------------------------------------
class JournalTemplates:
    @staticmethod
    def get_journal_styles():
        return {
            'nature': {
                'figure_width_single': 8.9, 'figure_width_double': 18.3,
                'font_family': 'Arial', 'font_size_small': 7,
                'font_size_medium': 8, 'font_size_large': 9,
                'line_width': 0.5, 'axes_linewidth': 0.5,
                'tick_width': 0.5, 'tick_length': 2,
                'grid_alpha': 0.1, 'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5, 'figure_width_double': 11.4,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.75, 'axes_linewidth': 0.75,
                'tick_width': 0.75, 'tick_length': 3,
                'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                              '#77AC30', '#4DBEEE', '#A2142F', '#FF00FF',
                              '#00FFFF', '#FFA500']
            },
            'j_power_sources': {
                'figure_width_single': 8.6, 'figure_width_double': 17.8,
                'font_family': 'Arial', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4,
                'grid_alpha': 0.2, 'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000',
                              '#44AA99', '#882255', '#117733', '#999933',
                              '#AA4499', '#88CCEE']
            },
            'custom': {
                'figure_width_single': 6.0, 'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans', 'font_size_small': 10,
                'font_size_medium': 12, 'font_size_large': 14,
                'line_width': 1.5, 'axes_linewidth': 1.5,
                'tick_width': 1.0, 'tick_length': 5,
                'grid_alpha': 0.3, 'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10))
            }
        }

# -----------------------------------------------------------------------------
# 3. PublicationEnhancer Class (unchanged)
# -----------------------------------------------------------------------------
class PublicationEnhancer:
    @staticmethod
    def create_custom_colormaps():
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        plasma_enhanced = LinearSegmentedColormap.from_list('plasma_enhanced', [
            (0.0, '#0c0887'), (0.1, '#4b03a1'), (0.3, '#8b0aa5'),
            (0.5, '#b83289'), (0.7, '#db5c68'), (0.9, '#f48849'),
            (1.0, '#fec325')
        ])
        coolwarm_enhanced = LinearSegmentedColormap.from_list('coolwarm_enhanced', [
            (0.0, '#3a4cc0'), (0.25, '#8abcdd'), (0.5, '#f7f7f7'),
            (0.75, '#f0b7a4'), (1.0, '#b40426')
        ])
        thermal_map = LinearSegmentedColormap.from_list('thermal_map', [
            (0.0, '#2c7bb6'), (0.2, '#abd9e9'), (0.4, '#ffffbf'),
            (0.6, '#fdae61'), (0.8, '#d7191c'), (1.0, '#800026')
        ])
        return {
            'plasma_enhanced': plasma_enhanced,
            'coolwarm_enhanced': coolwarm_enhanced,
            'thermal_map': thermal_map
        }

    @staticmethod
    def add_scale_bar(ax, length_physical, location='lower right',
                      color='white', linewidth=2, label='m'):
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if location == 'lower right':
            x_pos = 0.90; y_pos = 0.08
            ha = 'center'; va = 'center'
        elif location == 'lower left':
            x_pos = 0.10; y_pos = 0.08
            ha = 'center'; va = 'center'
        else:
            x_pos = 0.90; y_pos = 0.92
            ha = 'center'; va = 'center'
        bar_x_start = xlim[0] + x_pos * x_range - length_physical/2
        bar_x_end   = xlim[0] + x_pos * x_range + length_physical/2
        bar_y = ylim[0] + y_pos * y_range
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                color=color, linewidth=linewidth, solid_capstyle='butt')
        ax.text((bar_x_start+bar_x_end)/2, bar_y + 0.02*y_range,
                f'{length_physical:.2f} {label}', ha='center', va='bottom',
                color=color, fontsize=8, fontweight='bold')
        return ax

    @staticmethod
    def add_error_shading(ax, x, y_mean, y_std, color='blue', alpha=0.3, label=''):
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        color=color, alpha=alpha, label=label)
        return ax

    @staticmethod
    def add_confidence_band(ax, x, y_data, confidence=0.95, color='blue', alpha=0.2):
        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        n = len(y_data)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0
        y_err = t_val * y_std / np.sqrt(n)
        ax.fill_between(x, y_mean - y_err, y_mean + y_err,
                        color=color, alpha=alpha, label=f'{int(confidence*100)}% CI')
        return ax, y_mean, y_err

    @staticmethod
    def create_fancy_legend(ax, handles, labels, **kwargs):
        legend = ax.legend(handles, labels, **kwargs)
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.9)
        return legend

# -----------------------------------------------------------------------------
# 4. Advanced Styling Controls (unchanged)
# -----------------------------------------------------------------------------
def get_styling_controls():
    style = {}
    st.sidebar.header("🎨 Advanced Post‑Processing")
    with st.sidebar.expander("📐 Font & Text", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['title_font_size'] = st.slider("Title Size", 8, 32, 16)
            style['label_font_size'] = st.slider("Label Size", 8, 28, 14)
            style['tick_font_size'] = st.slider("Tick Size", 6, 20, 12)
        with col2:
            style['title_weight'] = st.selectbox("Title Weight", ['normal','bold','light','semibold'], index=1)
            style['label_weight'] = st.selectbox("Label Weight", ['normal','bold','light'], index=1)
            style['title_color'] = st.color_picker("Title Color", "#000000")
    with st.sidebar.expander("📏 Lines & Borders", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['line_width'] = st.slider("Line Width", 0.5, 5.0, 2.0, 0.5)
            style['spine_width'] = st.slider("Spine Width", 1.0, 4.0, 2.5, 0.5)
            style['tick_width'] = st.slider("Tick Width", 0.5, 3.0, 2.0, 0.5)
        with col2:
            style['tick_length'] = st.slider("Tick Length", 2, 15, 6)
            style['spine_color'] = st.color_picker("Spine Color", "#000000")
            style['grid_width'] = st.slider("Grid Width", 0.1, 2.0, 0.5, 0.1)
    with st.sidebar.expander("🌐 Grid & Background", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['show_grid'] = st.checkbox("Show Grid", True)
            style['grid_style'] = st.selectbox("Grid Style", ['-', '--', '-.', ':'], index=1)
            style['grid_alpha'] = st.slider("Grid Alpha", 0.0, 1.0, 0.3, 0.05)
            style['grid_zorder'] = st.slider("Grid Z‑Order", 0, 10, 0)
        with col2:
            style['figure_facecolor'] = st.color_picker("Figure Background", "#FFFFFF")
            style['axes_facecolor'] = st.color_picker("Axes Background", "#FFFFFF")
            style['show_minor_ticks'] = st.checkbox("Show Minor Ticks", True)
    with st.sidebar.expander("📊 Legend & Annotation", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['legend_fontsize'] = st.slider("Legend Size", 8, 20, 12)
            style['legend_location'] = st.selectbox("Legend Location",
                ['best','upper right','upper left','lower right','lower left','center'], index=0)
        with col2:
            style['show_legend'] = st.checkbox("Show Legend", True)
            style['legend_frame'] = st.checkbox("Legend Frame", True)

    # 🎨 Colorbar – with fixed T_min / T_max controls
    with st.sidebar.expander("🎨 Colorbar", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['colorbar_fontsize'] = st.slider("Colorbar Font", 8, 20, 12)
            style['colorbar_width']    = st.slider("Colorbar Width", 0.2, 1.0, 0.6, 0.05)
            style['colorbar_extend']   = st.selectbox("Colorbar Extend", ['neither','both','min','max'], index=0)
        with col2:
            style['colorbar_shrink']   = st.slider("Colorbar Shrink", 0.5, 1.0, 0.8, 0.05)
            style['colorbar_pad']      = st.slider("Colorbar Pad", 0.0, 0.2, 0.05, 0.01)
            style['cmap_normalization'] = st.selectbox("Norm", ['linear','log','power'], index=0)
            if style['cmap_normalization'] == 'power':
                style['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)

        st.markdown("**🌡️ Fixed Temperature Range**")
        style['use_custom_cbar_range'] = st.checkbox(
            "Lock color scale to user-defined T_min/T_max",
            value=False,
            help="When ON, the colorbar stays constant across frames/simulations. "
                 "When OFF, the colorbar auto-scales to the data range."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            style['cbar_t_min'] = st.number_input(
                "T_min (K)", min_value=200.0, max_value=2000.0,
                value=290.0, step=10.0, format="%.1f",
                disabled=not style['use_custom_cbar_range']
            )
        with c2:
            style['cbar_t_max'] = st.number_input(
                "T_max (K)", min_value=300.0, max_value=2500.0,
                value=900.0, step=10.0, format="%.1f",
                disabled=not style['use_custom_cbar_range']
            )
        with c3:
            style['cbar_auto_from_global'] = st.button(
                "Use Global Min/Max",
                help="Scan the current field to fill T_min/T_max automatically (then you can fine-tune)."
            )

    with st.sidebar.expander("📐 Advanced Layout", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['layout_pad'] = st.slider("Layout Pad", 0.5, 3.0, 1.0, 0.1)
            style['wspace'] = st.slider("Horizontal Spacing", 0.1, 1.0, 0.3, 0.05)
        with col2:
            style['hspace'] = st.slider("Vertical Spacing", 0.1, 1.0, 0.4, 0.05)
            style['figure_dpi'] = st.select_slider("Figure DPI", options=[150,300,600,1200], value=600)
    with st.sidebar.expander("📰 Publication", expanded=False):
        style['journal_style'] = st.selectbox("Journal Style", ['nature','science','j_power_sources','custom'], index=0)
        style['use_latex'] = st.checkbox("Use LaTeX", False)
        style['vector_output'] = st.checkbox("Vector Export", True)
        style['apply_smoothing'] = st.checkbox("Apply Smoothing", True)
    style['cmap'] = st.sidebar.selectbox("Default Colormap", cmap_list, index=cmap_list.index('hot'))
    return style

# -----------------------------------------------------------------------------
# 5. FigureStyler & EnhancedFigureStyler (unchanged)
# -----------------------------------------------------------------------------
class FigureStyler:
    @staticmethod
    def apply_advanced_styling(fig, axes, style_params):
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        for ax in axes:
            if ax is not None:
                ax.tick_params(axis='both', which='major',
                               labelsize=style_params.get('tick_font_size', 12),
                               width=style_params.get('tick_width', 1.5),
                               length=style_params.get('tick_length', 6))
                for spine in ax.spines.values():
                    spine.set_linewidth(style_params.get('spine_width', 2.0))
                    spine.set_color(style_params.get('spine_color', 'black'))
                if style_params.get('show_grid', True):
                    ax.grid(True, alpha=style_params.get('grid_alpha', 0.3),
                            linestyle=style_params.get('grid_style', '--'),
                            linewidth=style_params.get('grid_width', 0.5),
                            zorder=style_params.get('grid_zorder', 0))
                if style_params.get('figure_facecolor'):
                    fig.set_facecolor(style_params['figure_facecolor'])
        return fig

class EnhancedFigureStyler(FigureStyler):
    @staticmethod
    def apply_publication_styling(fig, axes, style_params):
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        for ax in axes:
            if ax is not None:
                if style_params.get('show_minor_ticks', True):
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=6,
                               width=style_params.get('tick_width', 1.0))
                ax.tick_params(which='minor', length=3,
                               width=style_params.get('tick_width', 1.0)*0.5)
                if style_params.get('use_latex', False):
                    xlabel = ax.get_xlabel()
                    ylabel = ax.get_ylabel()
                    if xlabel:
                        ax.set_xlabel(f'${xlabel}$')
                    if ylabel:
                        ax.set_ylabel(f'${ylabel}$')
        fig.set_constrained_layout(True)
        try:
            fig.subplots_adjust(wspace=style_params.get('wspace', 0.3),
                                hspace=style_params.get('hspace', 0.4))
        except:
            pass
        return fig

# -----------------------------------------------------------------------------
# 6. Simulation Database (unchanged)
# -----------------------------------------------------------------------------
class SimulationDB:
    @staticmethod
    def generate_id(sim_params):
        safe = {k: v for k, v in sim_params.items()
                if k not in ['reaction_params']}
        param_str = json.dumps(safe, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    @staticmethod
    def save_simulation(sim_params, history, metadata, final_3D, snapshots_3d=None, snapshot_times=None):
        if 'thermal_simulations' not in st.session_state:
            st.session_state.thermal_simulations = {}
        sim_id = SimulationDB.generate_id(sim_params)
        st.session_state.thermal_simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'history': history,
            'metadata': metadata,
            'final_3D': final_3D,
            'snapshots_3d': snapshots_3d,
            'snapshot_times': snapshot_times,
            'created_at': datetime.now().isoformat()
        }
        return sim_id

    @staticmethod
    def get_all_simulations():
        return st.session_state.get('thermal_simulations', {})

    @staticmethod
    def get_simulation_list():
        sims = []
        for sim_id, data in SimulationDB.get_all_simulations().items():
            params = data['params']
            name = f"{params.get('label', 'LiPo')} h={params['h_conv']:.1f} trig={params.get('trigger_temp', 'ISC')}"
            sims.append({'id': sim_id, 'name': name, 'params': params})
        return sims

    @staticmethod
    def delete_simulation(sim_id):
        if 'thermal_simulations' in st.session_state:
            if sim_id in st.session_state.thermal_simulations:
                del st.session_state.thermal_simulations[sim_id]
                return True
        return False

# -----------------------------------------------------------------------------
# 7. Thermal Line Profiler 3D (unchanged)
# -----------------------------------------------------------------------------
class ThermalLineProfiler3D:
    @staticmethod
    def extract_profile(data, profile_type, center_physical, extents,
                        offset_ratio=0.5, angle_deg=45, length_fraction=0.8):
        nx, ny, nz = data.shape
        extent_x = extents['x']; extent_y = extents['y']; extent_z = extents['z']
        dx = (extent_x[1]-extent_x[0]) / nx
        dy = (extent_y[1]-extent_y[0]) / ny
        dz = (extent_z[1]-extent_z[0]) / nz

        cx_idx = int((center_physical[0] - extent_x[0]) / dx)
        cy_idx = int((center_physical[1] - extent_y[0]) / dy)
        cz_idx = int((center_physical[2] - extent_z[0]) / dz)
        cx_idx = max(0, min(nx-1, cx_idx))
        cy_idx = max(0, min(ny-1, cy_idx))
        cz_idx = max(0, min(nz-1, cz_idx))

        max_len = int(min(nx, ny, nz) * length_fraction)
        offset = int(max_len * offset_ratio * 0.5)

        if profile_type == 'x':
            y_pos = cy_idx + int(offset * 0.5)
            z_pos = cz_idx + int(offset * 0.5)
            y_pos = max(0, min(ny-1, y_pos))
            z_pos = max(0, min(nz-1, z_pos))
            profile = data[:, y_pos, z_pos]
            distance = np.linspace(extent_x[0], extent_x[1], nx)
            endpoints = (extent_x[0], y_pos*dy+extent_y[0], z_pos*dz+extent_z[0],
                         extent_x[1], y_pos*dy+extent_y[0], z_pos*dz+extent_z[0])
            return distance, profile, endpoints

        elif profile_type == 'y':
            x_pos = cx_idx + int(offset * 0.5)
            z_pos = cz_idx + int(offset * 0.5)
            x_pos = max(0, min(nx-1, x_pos))
            z_pos = max(0, min(nz-1, z_pos))
            profile = data[x_pos, :, z_pos]
            distance = np.linspace(extent_y[0], extent_y[1], ny)
            endpoints = (x_pos*dx+extent_x[0], extent_y[0], z_pos*dz+extent_z[0],
                         x_pos*dx+extent_x[0], extent_y[1], z_pos*dz+extent_z[0])
            return distance, profile, endpoints

        elif profile_type == 'z':
            x_pos = cx_idx + int(offset * 0.5)
            y_pos = cy_idx + int(offset * 0.5)
            x_pos = max(0, min(nx-1, x_pos))
            y_pos = max(0, min(ny-1, y_pos))
            profile = data[x_pos, y_pos, :]
            distance = np.linspace(extent_z[0], extent_z[1], nz)
            endpoints = (x_pos*dx+extent_x[0], y_pos*dy+extent_y[0], extent_z[0],
                         x_pos*dx+extent_x[0], y_pos*dy+extent_y[0], extent_z[1])
            return distance, profile, endpoints

        else:
            if profile_type == 'diag_xy':
                length = int(max(nx, ny) * length_fraction)
                start_idx = (cx_idx - length//2, cy_idx - length//2, cz_idx)
                end_idx   = (cx_idx + length//2, cy_idx + length//2, cz_idx)
            elif profile_type == 'diag_xz':
                length = int(max(nx, nz) * length_fraction)
                start_idx = (cx_idx - length//2, cy_idx, cz_idx - length//2)
                end_idx   = (cx_idx + length//2, cy_idx, cz_idx + length//2)
            elif profile_type == 'diag_yz':
                length = int(max(ny, nz) * length_fraction)
                start_idx = (cx_idx, cy_idx - length//2, cz_idx - length//2)
                end_idx   = (cx_idx, cy_idx + length//2, cz_idx + length//2)
            else:
                angle_rad = np.deg2rad(angle_deg)
                length = int(max(nx, ny) * length_fraction)
                dx_line = int(length * np.cos(angle_rad) / 2)
                dy_line = int(length * np.sin(angle_rad) / 2)
                start_idx = (cx_idx - dx_line, cy_idx - dy_line, cz_idx + offset)
                end_idx   = (cx_idx + dx_line, cy_idx + dy_line, cz_idx + offset)
            start_idx = np.clip(start_idx, 0, (nx-1, ny-1, nz-1))
            end_idx   = np.clip(end_idx,   0, (nx-1, ny-1, nz-1))
            n_pts = 200
            x_pts = np.linspace(start_idx[0], end_idx[0], n_pts)
            y_pts = np.linspace(start_idx[1], end_idx[1], n_pts)
            z_pts = np.linspace(start_idx[2], end_idx[2], n_pts)
            coords = np.vstack((x_pts, y_pts, z_pts))
            profile = interpolate.map_coordinates(data, coords, order=1, mode='nearest')
            phys_x = extent_x[0] + x_pts * dx
            phys_y = extent_y[0] + y_pts * dy
            phys_z = extent_z[0] + z_pts * dz
            dist = np.sqrt((phys_x - phys_x[0])**2 +
                           (phys_y - phys_y[0])**2 +
                           (phys_z - phys_z[0])**2)
            endpoints = (start_idx[0]*dx+extent_x[0], start_idx[1]*dy+extent_y[0], start_idx[2]*dz+extent_z[0],
                         end_idx[0]*dx+extent_x[0], end_idx[1]*dy+extent_y[0], end_idx[2]*dz+extent_z[0])
            return dist, profile, endpoints

# -----------------------------------------------------------------------------
# 7.5 Initial Condition Helpers – VECTORIZED (FIXED alpha-lock + realistic H)
# -----------------------------------------------------------------------------
R_GAS = 8.314

# Reaction parameter presets — H scaled to ~1e8–1e9 J/m³ for realistic runaway
REALISTIC_REACTION_PARAMS = np.array([
    [1.3508e5, 1.667e15, 2.50e8],      # SEI
    [1.5006e5, 2.500e13, 7.00e8],      # Electrolyte
    [1.3960e5, 6.667e13, 1.20e8],      # Separator
    [2.0000e5, 5.700e15, 3.50e8],      # Cathode
], dtype=np.float64)

AGGRESSIVE_REACTION_PARAMS = np.array([
    [1.3508e5, 5.000e15, 5.00e8],
    [1.5006e5, 1.000e14, 1.20e9],
    [1.3960e5, 2.000e14, 3.00e8],
    [2.0000e5, 1.000e16, 7.00e8],
], dtype=np.float64)

DEFAULT_REACTION_PARAMS = np.array([
    [1.3508e5, 1.667e15, 2.50e8],
    [1.5006e5, 2.500e13, 7.00e8],
    [1.3960e5, 6.667e13, 1.20e8],
    [2.0000e5, 5.700e15, 3.50e8],
], dtype=np.float64)

HIGH_RISE_REACTION_PARAMS = np.array([
    [1.3508e5, 1.667e15, 5.00e8],      # SEI
    [1.5006e5, 2.500e13, 1.40e9],      # Electrolyte
    [1.3960e5, 6.667e13, 2.40e8],      # Separator
    [2.0000e5, 5.700e15, 7.00e8],      # Cathode
], dtype=np.float64)

def initialize_temperature_field(Nx, Ny, Nz, T_amb, trigger_temp,
                                  trigger_radius, trigger_center=None):
    T = np.full((Nx, Ny, Nz), T_amb, dtype=np.float64)
    if trigger_center is None:
        cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    else:
        cx, cy, cz = trigger_center

    i = np.arange(Nx)[:, None, None]
    j = np.arange(Ny)[None, :, None]
    k = np.arange(Nz)[None, None, :]
    dist = np.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
    mask = dist <= trigger_radius
    r_norm = dist / max(trigger_radius, 1)
    T[mask] = trigger_temp - (trigger_temp - T_amb) * (r_norm[mask])**2
    return T

def initialize_reaction_degrees(Nx, Ny, Nz, trigger_radius, trigger_center=None):
    alphas = np.zeros((4, Nx, Ny, Nz), dtype=np.float64)
    alphas[1] += 1e-4
    alphas[2] += 1e-4
    alphas[3] += 1e-4

    if trigger_center is None:
        cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    else:
        cx, cy, cz = trigger_center

    i = np.arange(Nx)[:, None, None]
    j = np.arange(Ny)[None, :, None]
    k = np.arange(Nz)[None, None, :]
    dist = np.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
    mask = dist <= trigger_radius
    r_norm = dist / max(trigger_radius, 1)

    alphas[0, mask] = 0.05 - 0.02 * r_norm[mask]
    alphas[1, mask] = 0.02
    alphas[2, mask] = 0.02
    alphas[3, mask] = 0.02
    return alphas

# -----------------------------------------------------------------------------
# 8. Numba Kernel – with T_cap, conservative fuel consumption, sustained heater,
#    and NOW with localized ISC heater (q_loc, isc_active, loc_mask)
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def step_3d(T, alphas, dt,
            rho, Cp, kx, ky, kz, dx, dy, dz,
            q_normal, reaction_params, T_amb, h_conv, eps, sigma, R,
            q_heater, heater_active, heater_face,
            q_loc, isc_active, loc_mask, T_cap):
    Nx, Ny, Nz = T.shape
    T_new = T.copy()
    alphas_new = alphas.copy()
    for i in prange(1, Nx - 1):
        for j in prange(1, Ny - 1):
            for k in prange(1, Nz - 1):
                d2Tdx2 = (T[i+1,j,k] - 2*T[i,j,k] + T[i-1,j,k]) / (dx*dx)
                d2Tdy2 = (T[i,j+1,k] - 2*T[i,j,k] + T[i,j-1,k]) / (dy*dy)
                d2Tdz2 = (T[i,j,k+1] - 2*T[i,j,k] + T[i,j,k-1]) / (dz*dz)
                T_ijk = T[i,j,k]
                q_abuse = 0.0
                for r in range(4):
                    Ea = reaction_params[r,0]
                    A  = reaction_params[r,1]
                    H  = reaction_params[r,2]
                    alpha = alphas[r,i,j,k]
                    
                    if r == 0:
                        f_alpha = 1.0 - alpha
                    elif r == 1:
                        f_alpha = alpha * (1.0 - alpha) + 1e-4 * (1.0 - alpha)
                    else:
                        f_alpha = 1.0 - alpha
                        
                    rate = A * np.exp(-Ea / (R * max(T_ijk, 1.0)))
                    q_abuse += H * rate * f_alpha
                    dalpha = min(rate * f_alpha * dt, 1.0 - alpha)
                    alphas_new[r,i,j,k] = alpha + dalpha

                q_total = q_normal + q_abuse
                if isc_active and loc_mask[i,j,k]:
                    q_total += q_loc
                q_cap = rho * Cp * max(0.0, (T_cap - T_ijk)) / dt
                if q_total > q_cap:
                    q_total = q_cap
                T_new[i,j,k] = T_ijk + dt/(rho*Cp) * (
                    kx*d2Tdx2 + ky*d2Tdy2 + kz*d2Tdz2 + q_total
                )

    # Boundary conditions – with optional sustained heater flux
    for j in prange(Ny):
        for k in prange(Nz):
            T_surf = T[0,j,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            q_net = q_conv + q_rad
            if heater_active and heater_face == 0:
                q_net -= q_heater
            T_new[0,j,k] = T_new[1,j,k] - (dx/kx) * q_net

            T_surf = T[Nx-1,j,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            q_net = q_conv + q_rad
            if heater_active and heater_face == 1:
                q_net -= q_heater
            T_new[Nx-1,j,k] = T_new[Nx-2,j,k] - (dx/kx) * q_net

    for i in prange(Nx):
        for k in prange(Nz):
            T_surf = T[i,0,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            q_net = q_conv + q_rad
            if heater_active and heater_face == 2:
                q_net -= q_heater
            T_new[i,0,k] = T_new[i,1,k] - (dy/ky) * q_net

            T_surf = T[i,Ny-1,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            q_net = q_conv + q_rad
            if heater_active and heater_face == 3:
                q_net -= q_heater
            T_new[i,Ny-1,k] = T_new[i,Ny-2,k] - (dy/ky) * q_net

    for i in prange(Nx):
        for j in prange(Ny):
            T_surf = T[i,j,0]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            q_net = q_conv + q_rad
            if heater_active and heater_face == 4:
                q_net -= q_heater
            T_new[i,j,0] = T_new[i,j,1] - (dz/kz) * q_net

            T_surf = T[i,j,Nz-1]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            q_net = q_conv + q_rad
            if heater_active and heater_face == 5:
                q_net -= q_heater
            T_new[i,j,Nz-1] = T_new[i,j,Nz-2] - (dz/kz) * q_net

    return T_new, alphas_new

# -----------------------------------------------------------------------------
# 8.5 Domain Sketch Functions (with uirevision)
# -----------------------------------------------------------------------------
def plot_initial_domain_sketch(params):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    Lx, Lz = params['Lx'], params['Lz']
    Nx = params['Nx']
    cell_rect = plt.Rectangle((0, 0), Lx, Lz, linewidth=2.5, edgecolor='#2c3e50', facecolor='#3498db', alpha=0.25)
    ax.add_patch(cell_rect)
    margin = max(Lx, Lz) * 0.25
    air_rect = plt.Rectangle((-margin, -margin), Lx + 2*margin, Lz + 2*margin, 
                             linewidth=1.5, edgecolor='#7f8c8d', facecolor='none', linestyle='--')
    ax.add_patch(air_rect)
    cx, cz = Lx/2, Lz/2
    dx = Lx / (Nx - 1)
    # Only show hotspot if not using ISC heater (or always show? We'll keep it optional)
    if not params.get('local_heater_enabled', False):
        r_phys = params['trigger_radius'] * dx
        if r_phys > 0:
            hotspot = plt.Circle((cx, cz), r_phys, color='#e74c3c', alpha=0.8, label=f'Hotspot ({params["trigger_radius"]} cells)')
            ax.add_patch(hotspot)
    else:
        # Show ISC region
        r_phys = params['loc_radius'] * dx
        if r_phys > 0:
            isc_circle = plt.Circle((cx, cz), r_phys, color='#e67e22', alpha=0.5, label=f'ISC region (r={params["loc_radius"]} cells)')
            ax.add_patch(isc_circle)
            ax.text(cx, cz, 'ISC\nheater', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    arrow_style = dict(arrowstyle='->', color='#e67e22', lw=2, mutation_scale=20)
    ax.annotate('', xy=(Lx/2, Lz + margin*0.6), xytext=(Lx/2, Lz), arrowprops=arrow_style)
    ax.text(Lx/2 + 0.002, Lz + margin*0.6, 'Convection ($h$) + Radiation ($\\epsilon$)', fontsize=11, color='#d35400', ha='left', fontweight='bold')
    ax.annotate('', xy=(Lx/2, -margin*0.6), xytext=(Lx/2, 0), arrowprops=arrow_style)
    ax.text(Lx/2 + 0.002, -margin*0.6, 'Convection ($h$) + Radiation ($\\epsilon$)', fontsize=11, color='#d35400', ha='left', fontweight='bold')
    ax.annotate('', xy=(-margin*0.6, Lz/2), xytext=(0, Lz/2), arrowprops=arrow_style)
    ax.annotate('', xy=(Lx + margin*0.6, Lz/2), xytext=(Lx, Lz/2), arrowprops=arrow_style)
    ax.text(Lx/2, Lz/2, 'LiPo Cell Core\n(Anisotropic $k$)', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#2c3e50')
    ax.text(Lx/2, -margin*0.9, f'Ambient Air Domain ($T_\\infty = {params["T_amb"]}$ K)', 
            ha='center', va='center', fontsize=13, color='#7f8c8d', fontstyle='italic')
    ax.annotate('', xy=(Lx*0.85, Lz/2), xytext=(Lx*0.65, Lz/2), 
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.5))
    ax.text(Lx*0.75, Lz/2 + 0.003, '$k_x, k_y$ (High)', ha='center', fontsize=11, color='#2980b9', fontweight='bold')
    ax.annotate('', xy=(Lx/2, Lz*0.85), xytext=(Lx/2, Lz*0.65), 
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax.text(Lx/2 + 0.003, Lz*0.75, '$k_z$ (Low)', ha='left', fontsize=11, color='#27ae60', fontweight='bold')
    ax.set_xlim(-margin*1.2, Lx + margin*1.2)
    ax.set_ylim(-margin*1.2, Lz + margin*1.2)
    ax.set_xlabel('x (m) - Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('z (m) - Thickness', fontsize=12, fontweight='bold')
    ax.set_title('2D Cross-Section (X-Z Plane) of Initial Thermal Domain', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    return fig

def _add_physical_arrow(fig, base, axis, sign, length, color,
                        head_frac=0.35, head_rad_frac=0.15):
    head_len = length * head_frac
    shaft_len = length - head_len
    head_rad = length * head_rad_frac
    b = np.array(base, dtype=float)
    vec = np.zeros(3); vec[axis] = sign
    p1 = np.zeros(3); p1[(axis + 1) % 3] = 1.0
    p2 = np.zeros(3); p2[(axis + 2) % 3] = 1.0
    shaft_end = b + vec * shaft_len
    fig.add_trace(go.Scatter3d(
        x=[b[0], shaft_end[0]], y=[b[1], shaft_end[1]], z=[b[2], shaft_end[2]],
        mode='lines', line=dict(color=color, width=5),
        hoverinfo='skip', showlegend=False))
    theta = np.linspace(0, 2 * np.pi, 24)
    t = np.linspace(0, 1, 2)
    TH, TT = np.meshgrid(theta, t)
    R = head_rad * (1 - TT); H = head_len * TT
    perp = np.cos(TH)[..., None] * p1 + np.sin(TH)[..., None] * p2
    P = shaft_end + R[..., None] * perp + H[..., None] * vec
    fig.add_trace(go.Surface(
        x=P[..., 0], y=P[..., 1], z=P[..., 2],
        colorscale=[[0, color], [1, color]], showscale=False,
        opacity=0.9, hoverinfo='skip', showlegend=False))

def plot_3d_domain_sketch(params):
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    Nx = params['Nx']
    r_phys = params['trigger_radius'] * (Lx / (Nx - 1))
    # If ISC enabled, show ISC region instead
    local_heater = params.get('local_heater_enabled', False)
    if local_heater:
        r_phys = params['loc_radius'] * (Lx / (Nx - 1))
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 30)
    x_s = Lx/2 + r_phys * np.outer(np.cos(u), np.sin(v))
    y_s = Ly/2 + r_phys * np.outer(np.sin(u), np.sin(v))
    z_s = Lz/2 + r_phys * np.outer(np.ones(np.size(u)), np.cos(v))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0, Lx, Lx, 0, 0, 0, Lx, Lx, 0, 0, Lx, Lx, Lx, Lx, 0, 0],
        y=[0, 0, Ly, Ly, 0, 0, 0, Ly, Ly, 0, 0, 0, Ly, Ly, Ly, Ly],
        z=[0, 0, 0, 0, 0, Lz, Lz, Lz, Lz, Lz, Lz, 0, 0, Lz, Lz, 0],
        mode='lines', line=dict(color='#2c3e50', width=3),
        name='LiPo Cell Boundary', hoverinfo='skip'))
    fig.add_trace(go.Mesh3d(
        x=[0, Lx, Lx, 0, 0, Lx, Lx, 0], y=[0, 0, Ly, Ly, 0, 0, Ly, Ly],
        z=[0, 0, 0, 0, Lz, Lz, Lz, Lz],
        i=[0,0,4,4,2,2,5,5,1,1,3,3], j=[1,2,5,6,3,0,1,4,5,4,7,6],
        k=[2,3,6,7,0,1,4,0,4,5,6,2],
        color='#3498db', opacity=0.15,
        name='Cell Core', showscale=False))
    if r_phys > 0:
        if local_heater:
            color = '#e67e22'
            name = 'ISC Heater Region'
        else:
            color = '#e74c3c'
            name = 'Hotspot Trigger'
        fig.add_trace(go.Surface(
            x=x_s, y=y_s, z=z_s,
            colorscale=[[0, color], [1, color]],
            showscale=False, opacity=0.6, name=name))
    arrow_len = max(Lx, Ly, Lz) * 0.35
    faces = {
        'Top (+Z)':    ((Lx/2, Ly/2, Lz),   2, +1),
        'Bottom (-Z)': ((Lx/2, Ly/2, 0),    2, -1),
        'Right (+X)':  ((Lx,   Ly/2, Lz/2), 0, +1),
        'Left (-X)':   ((0,    Ly/2, Lz/2), 0, -1),
        'Front (+Y)':  ((Lx/2, Ly,   Lz/2), 1, +1),
        'Back (-Y)':   ((Lx/2, 0,    Lz/2), 1, -1),
    }
    for name, (base, ax_, sgn) in faces.items():
        _add_physical_arrow(fig, base, ax_, sgn, arrow_len, '#e67e22')
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                               line=dict(color='#e67e22', width=5),
                               name='Heat Loss (h·ΔT + εσ·ΔT⁴)'))
    m = arrow_len * 1.3
    fig.update_layout(
        scene=dict(
            uirevision='constant_sketch',
            xaxis=dict(title='x (m)', range=[-m, Lx + m], backgroundcolor='white', gridcolor='#eeeeee'),
            yaxis=dict(title='y (m)', range=[-m, Ly + m], backgroundcolor='white', gridcolor='#eeeeee'),
            zaxis=dict(title='z (m)', range=[-m, Lz + m], backgroundcolor='white', gridcolor='#eeeeee'),
            aspectmode='data'),
        uirevision='constant_sketch',
        title=dict(text='🔥 3D LiPo Cell Geometry & Boundary Conditions', x=0.5),
        height=700, margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
    return fig

# -----------------------------------------------------------------------------
# 8.6 Visualization Functions (with frames for time slider)
# -----------------------------------------------------------------------------
def create_mesh_aware_3d_thermal(T_3d, extents, style_params, 
                                  show_mesh=True, mesh_opacity=0.3,
                                  slice_axis='z', slice_position=0.5):
    Nx, Ny, Nz = T_3d.shape
    ext_x = extents['x']; ext_y = extents['y']; ext_z = extents['z']
    x = np.linspace(ext_x[0], ext_x[1], Nx)
    y = np.linspace(ext_y[0], ext_y[1], Ny)
    z = np.linspace(ext_z[0], ext_z[1], Nz)
    if slice_axis == 'z':
        slice_idx = int(Nz * slice_position)
        slice_idx = max(0, min(Nz-1, slice_idx))
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z_data = T_3d[:, :, slice_idx]
        Z_pos = np.full_like(X, z[slice_idx])
    elif slice_axis == 'y':
        slice_idx = int(Ny * slice_position)
        slice_idx = max(0, min(Ny-1, slice_idx))
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y_data = T_3d[:, slice_idx, :]
        Y_pos = np.full_like(X, y[slice_idx])
    else:
        slice_idx = int(Nx * slice_position)
        slice_idx = max(0, min(Nx-1, slice_idx))
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X_data = T_3d[slice_idx, :, :]
        X_pos = np.full_like(Y, x[slice_idx])
    cmap_name = style_params.get('cmap', 'hot')
    pl_colorscale = matplotlib_to_plotly(cmap_name, pl_entries=20)
    cmin, cmax = resolve_cbar_range(style_params, T_3d)
    fig = go.Figure()
    colorbar_config = dict(
        title=dict(text='Temperature (K)', side='right', font=dict(size=14)),
        thickness=15, len=0.8
    )
    if slice_axis == 'z':
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_pos,
            surfacecolor=Z_data,
            colorscale=pl_colorscale,
            cmin=cmin, cmax=cmax,
            colorbar=colorbar_config,
            showscale=True,
            opacity=0.9,
            lighting=dict(ambient=0.6, diffuse=0.4, specular=0.1)
        ))
        if show_mesh:
            step_x = max(1, Nx // 15)
            step_y = max(1, Ny // 15)
            for i in range(0, Nx, step_x):
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[i]], y=[ext_y[0], ext_y[1]], z=[z[slice_idx], z[slice_idx]],
                    mode='lines', line=dict(color='gray', width=1),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
            for j in range(0, Ny, step_y):
                fig.add_trace(go.Scatter3d(
                    x=[ext_x[0], ext_x[1]], y=[y[j], y[j]], z=[z[slice_idx], z[slice_idx]],
                    mode='lines', line=dict(color='gray', width=1),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
    elif slice_axis == 'y':
        fig.add_trace(go.Surface(
            x=X, y=Y_pos, z=Z,
            surfacecolor=Y_data,
            colorscale=pl_colorscale,
            cmin=cmin, cmax=cmax,
            colorbar=colorbar_config,
            showscale=True,
            opacity=0.9,
            lighting=dict(ambient=0.6, diffuse=0.4, specular=0.1)
        ))
        if show_mesh:
            step_x = max(1, Nx // 15)
            step_z = max(1, Nz // 15)
            for i in range(0, Nx, step_x):
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[i]], y=[y[slice_idx], y[slice_idx]], z=[ext_z[0], ext_z[1]],
                    mode='lines', line=dict(color='gray', width=1),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
            for k in range(0, Nz, step_z):
                fig.add_trace(go.Scatter3d(
                    x=[ext_x[0], ext_x[1]], y=[y[slice_idx], y[slice_idx]], z=[z[k], z[k]],
                    mode='lines', line=dict(color='gray', width=1),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
    else:
        fig.add_trace(go.Surface(
            x=X_pos, y=Y, z=Z,
            surfacecolor=X_data,
            colorscale=pl_colorscale,
            cmin=cmin, cmax=cmax,
            colorbar=colorbar_config,
            showscale=True,
            opacity=0.9,
            lighting=dict(ambient=0.6, diffuse=0.4, specular=0.1)
        ))
        if show_mesh:
            step_y = max(1, Ny // 15)
            step_z = max(1, Nz // 15)
            for j in range(0, Ny, step_y):
                fig.add_trace(go.Scatter3d(
                    x=[x[slice_idx], x[slice_idx]], y=[y[j], y[j]], z=[ext_z[0], ext_z[1]],
                    mode='lines', line=dict(color='gray', width=1),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
            for k in range(0, Nz, step_z):
                fig.add_trace(go.Scatter3d(
                    x=[x[slice_idx], x[slice_idx]], y=[ext_y[0], ext_y[1]], z=[z[k], z[k]],
                    mode='lines', line=dict(color='gray', width=1),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
    margin = max(ext_x[1]-ext_x[0], ext_y[1]-ext_y[0], ext_z[1]-ext_z[0]) * 0.1
    fig.add_trace(go.Scatter3d(
        x=[ext_x[0], ext_x[1], ext_x[1], ext_x[0], ext_x[0],
           ext_x[0], ext_x[1], ext_x[1], ext_x[0], ext_x[0],
           ext_x[1], ext_x[1], ext_x[1], ext_x[1], ext_x[0], ext_x[0]],
        y=[ext_y[0], ext_y[0], ext_y[1], ext_y[1], ext_y[0],
           ext_y[0], ext_y[0], ext_y[1], ext_y[1], ext_y[0],
           ext_y[0], ext_y[1], ext_y[1], ext_y[0], ext_y[0], ext_y[1]],
        z=[ext_z[0], ext_z[0], ext_z[0], ext_z[0], ext_z[0],
           ext_z[1], ext_z[1], ext_z[1], ext_z[1], ext_z[1],
           ext_z[1], ext_z[1], ext_z[0], ext_z[0], ext_z[0], ext_z[1]],
        mode='lines', line=dict(color='#2c3e50', width=3),
        name='Domain Boundary', hoverinfo='skip'
    ))
    T_min = np.min(T_3d); T_max = np.max(T_3d)
    title_size = style_params.get('title_font_size', 18)
    title_color = style_params.get('title_color', '#000000')
    label_size = style_params.get('label_font_size', 14)
    tick_size = style_params.get('tick_font_size', 12)
    spine_color = style_params.get('spine_color', '#000000')
    spine_width = style_params.get('spine_width', 1.0)
    bg_color = style_params.get('figure_facecolor', '#FFFFFF')
    axis_template = dict(
        tickfont=dict(size=tick_size, color=title_color),
        gridcolor=spine_color if style_params.get('show_grid', True) else 'rgba(0,0,0,0)',
        gridwidth=style_params.get('grid_width', 0.5),
        showgrid=style_params.get('show_grid', True),
        showline=True, linewidth=spine_width, linecolor=spine_color, zeroline=False
    )
    def make_axis(template, title_text, rng):
        d = template.copy()
        d['title'] = dict(text=title_text, font=dict(size=label_size, color=title_color))
        d['range'] = rng
        return d
    fig.update_layout(
        scene=dict(
            uirevision='constant_single',
            xaxis=make_axis(axis_template, 'X (m)', [ext_x[0]-margin, ext_x[1]+margin]),
            yaxis=make_axis(axis_template, 'Y (m)', [ext_y[0]-margin, ext_y[1]+margin]),
            zaxis=make_axis(axis_template, 'Z (m)', [ext_z[0]-margin, ext_z[1]+margin]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            bgcolor=bg_color
        ),
        uirevision='constant_single',
        title=dict(
            text=f'🔥 3D Thermal Field with Mesh | T: {T_min:.1f} - {T_max:.1f} K',
            x=0.5,
            font=dict(size=title_size, color=title_color, family='Arial')
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color
    )
    return fig

# NEW: Multi-slice with Plotly Frames for time slider
def create_multi_slice_3d_visualization_frames(snapshots, snapshot_times, extents, style_params,
                                                n_slices=5, show_cross_slices=False):
    """
    Builds a Plotly figure with pre-computed frames for each time snapshot.
    This allows the time slider to be entirely within the browser, preserving camera state.
    """
    if not snapshots:
        return go.Figure()
    
    Nx, Ny, Nz = snapshots[0].shape
    ext_x = extents['x']; ext_y = extents['y']; ext_z = extents['z']
    x = np.linspace(ext_x[0], ext_x[1], Nx)
    y = np.linspace(ext_y[0], ext_y[1], Ny)
    z = np.linspace(ext_z[0], ext_z[1], Nz)
    
    cmap_name = style_params.get('cmap', 'hot')
    pl_colorscale = matplotlib_to_plotly(cmap_name, pl_entries=20)
    
    # Determine global cmin/cmax across all snapshots
    all_T = np.concatenate([snap.ravel() for snap in snapshots])
    cmin, cmax = resolve_cbar_range(style_params, all_T)
    if cmin is None:
        cmin = float(np.min(all_T))
        cmax = float(np.max(all_T))
    
    # Build z-slice indices once
    if Nz > 2:
        z_slice_indices = np.linspace(1, Nz-2, n_slices, dtype=int)
    else:
        z_slice_indices = np.array([Nz//2])
    z_slice_indices = np.unique(z_slice_indices)
    n_actual_slices = len(z_slice_indices)
    
    # Prepare colorbar
    colorbar_config = dict(
        title=dict(text='Temperature (K)', side='right', font=dict(size=style_params.get('colorbar_fontsize', 12))),
        thickness=int(style_params.get('colorbar_width', 0.6)*25),
        len=style_params.get('colorbar_shrink', 0.8),
        outlinewidth=style_params.get('spine_width', 1.0),
        outlinecolor=style_params.get('spine_color', '#000000'),
        tickfont=dict(size=style_params.get('tick_font_size', 12), color=style_params.get('title_color', '#000000')),
        tickformat='.1f'
    )
    
    # Create initial data (first snapshot)
    initial_T = snapshots[0]
    fig = go.Figure()
    
    # Add main slice traces (they will be updated via frames)
    # We add a constant number of traces (n_slices + optional cross slices)
    for idx, kz in enumerate(z_slice_indices):
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z_pos = np.full_like(X, z[kz])
        T_slice = initial_T[:, :, kz]
        opacity = 0.5 + 0.4 * (kz / max(Nz-1, 1))
        is_last = (idx == n_actual_slices - 1)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_pos,
            surfacecolor=T_slice,
            colorscale=pl_colorscale,
            cmin=cmin, cmax=cmax,
            showscale=is_last,
            colorbar=colorbar_config if is_last else None,
            opacity=opacity,
            name=f'Z = {z[kz]*1000:.1f} mm'
        ))
        # Add mesh lines for this slice (optional)
        if style_params.get('show_grid', True):
            mesh_color = style_params.get('spine_color', '#000000')
            mesh_width = max(0.3, style_params.get('line_width', 1.0)*0.3)
            mesh_opacity = style_params.get('grid_alpha', 0.3)
            step_x = max(1, Nx // 10)
            step_y = max(1, Ny // 10)
            for i in range(0, Nx, step_x):
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[i]], y=[ext_y[0], ext_y[1]], z=[z[kz], z[kz]],
                    mode='lines', line=dict(color=mesh_color, width=mesh_width),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
            for j in range(0, Ny, step_y):
                fig.add_trace(go.Scatter3d(
                    x=[ext_x[0], ext_x[1]], y=[y[j], y[j]], z=[z[kz], z[kz]],
                    mode='lines', line=dict(color=mesh_color, width=mesh_width),
                    opacity=mesh_opacity, showlegend=False, hoverinfo='skip'
                ))
    
    # Optional cross-slice traces (constant)
    if show_cross_slices:
        ky = Ny // 2
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y_pos = np.full_like(X, y[ky])
        fig.add_trace(go.Surface(
            x=X, y=Y_pos, z=Z, surfacecolor=initial_T[:, ky, :],
            colorscale=pl_colorscale, showscale=False,
            cmin=cmin, cmax=cmax,
            opacity=0.25, name='Y-center slice'
        ))
        kx = Nx // 2
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X_pos = np.full_like(Y, x[kx])
        fig.add_trace(go.Surface(
            x=X_pos, y=Y, z=Z, surfacecolor=initial_T[kx, :, :],
            colorscale=pl_colorscale, showscale=False,
            cmin=cmin, cmax=cmax,
            opacity=0.25, name='X-center slice'
        ))
    
    # Domain boundary box (constant)
    box_lines = [
        ([ext_x[0], ext_x[1]], [ext_y[0], ext_y[0]], [ext_z[0], ext_z[0]]),
        ([ext_x[0], ext_x[1]], [ext_y[1], ext_y[1]], [ext_z[0], ext_z[0]]),
        ([ext_x[0], ext_x[0]], [ext_y[0], ext_y[1]], [ext_z[0], ext_z[0]]),
        ([ext_x[1], ext_x[1]], [ext_y[0], ext_y[1]], [ext_z[0], ext_z[0]]),
        ([ext_x[0], ext_x[1]], [ext_y[0], ext_y[0]], [ext_z[1], ext_z[1]]),
        ([ext_x[0], ext_x[1]], [ext_y[1], ext_y[1]], [ext_z[1], ext_z[1]]),
        ([ext_x[0], ext_x[0]], [ext_y[0], ext_y[1]], [ext_z[1], ext_z[1]]),
        ([ext_x[1], ext_x[1]], [ext_y[0], ext_y[1]], [ext_z[1], ext_z[1]]),
        ([ext_x[0], ext_x[0]], [ext_y[0], ext_y[0]], [ext_z[0], ext_z[1]]),
        ([ext_x[1], ext_x[1]], [ext_y[0], ext_y[0]], [ext_z[0], ext_z[1]]),
        ([ext_x[0], ext_x[0]], [ext_y[1], ext_y[1]], [ext_z[0], ext_z[1]]),
        ([ext_x[1], ext_x[1]], [ext_y[1], ext_y[1]], [ext_z[0], ext_z[1]]),
    ]
    box_color = style_params.get('spine_color', '#2c3e50')
    box_width = style_params.get('line_width', 2.0)
    for bx, by, bz in box_lines:
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color=box_color, width=box_width),
            name='Domain Boundary', hoverinfo='skip', showlegend=False
        ))
    corners_x = [ext_x[0], ext_x[1], ext_x[0], ext_x[1], ext_x[0], ext_x[1], ext_x[0], ext_x[1]]
    corners_y = [ext_y[0], ext_y[0], ext_y[1], ext_y[1], ext_y[0], ext_y[0], ext_y[1], ext_y[1]]
    corners_z = [ext_z[0], ext_z[0], ext_z[0], ext_z[0], ext_z[1], ext_z[1], ext_z[1], ext_z[1]]
    fig.add_trace(go.Scatter3d(
        x=corners_x, y=corners_y, z=corners_z,
        mode='markers',
        marker=dict(size=5, color=box_color, symbol='diamond'),
        name='Mesh Nodes', showlegend=True
    ))
    
    # Build frames for each snapshot
    frames = []
    for t_idx, T in enumerate(snapshots):
        frame_data = []
        # Update main slice surfaces
        for idx, kz in enumerate(z_slice_indices):
            T_slice = T[:, :, kz]
            # The surface trace index is idx (since we added them in order)
            frame_data.append(go.Surface(
                surfacecolor=T_slice,
                # other properties remain same as original; we only update surfacecolor
            ))
        # If cross-slices exist, they are at indices n_slices and n_slices+1
        if show_cross_slices:
            # Y-center slice
            frame_data.append(go.Surface(surfacecolor=T[:, ky, :]))
            # X-center slice
            frame_data.append(go.Surface(surfacecolor=T[kx, :, :]))
        # We need to match the number of traces. But we only need to specify the changing ones.
        # Plotly frames can update only the traces that change. We can use 'data' list with updates.
        # However, we also have scatter traces (mesh lines, box) that don't change.
        # To simplify, we can include all traces in each frame, but that would be heavy.
        # Better: use 'data' list containing only the updated surface traces, and rely on 'traces' attribute to specify which traces to update.
        # But easier: include all traces in each frame, but duplicate the non-changing ones. That's inefficient but acceptable for moderate snapshots.
        # For performance, we'll create a full set of traces for each frame (including mesh lines etc.).
        # However, mesh lines are many traces, causing bloat. Instead, we can use the 'traces' argument to update only the surface traces.
        # Since Plotly's Python API supports updating specific traces via 'traces' in Frame, we'll do that.
        # We'll create a list of trace indices to update: the main slice surfaces and cross-slice surfaces.
        update_indices = list(range(n_actual_slices))
        if show_cross_slices:
            update_indices.extend([n_actual_slices, n_actual_slices+1])
        # Build a frame with only the updated data for those indices.
        frame = go.Frame(
            data=[go.Surface(surfacecolor=T[:, :, kz]) for kz in z_slice_indices] + 
                 ([go.Surface(surfacecolor=T[:, ky, :]), go.Surface(surfacecolor=T[kx, :, :])] if show_cross_slices else []),
            name=f't={snapshot_times[t_idx]:.1f}s',
            traces=update_indices
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Layout with sliders and updatemenus
    sliders = [{
        'currentvalue': {'prefix': 'Time: ', 'suffix': ' s'},
        'steps': [
            {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
             'label': f"{snapshot_times[i]:.1f}", 'method': 'animate'}
            for i, f in enumerate(frames)
        ],
        'len': 0.9,
        'x': 0.1,
        'y': 0
    }]
    
    updatemenus = [{
        'type': 'buttons',
        'buttons': [
            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]},
            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ],
        'pad': {'r': 10, 't': 10},
        'showactive': False,
        'x': 0.05,
        'y': 0
    }]
    
    # Styling
    T_min = np.min(all_T)
    T_max = np.max(all_T)
    margin = max(ext_x[1]-ext_x[0], ext_y[1]-ext_y[0], ext_z[1]-ext_z[0]) * 0.1
    title_size = style_params.get('title_font_size', 16)
    title_color = style_params.get('title_color', '#000000')
    label_size = style_params.get('label_font_size', 14)
    tick_size = style_params.get('tick_font_size', 12)
    spine_color = style_params.get('spine_color', '#000000')
    spine_width = style_params.get('spine_width', 1.0)
    bg_color = style_params.get('figure_facecolor', '#FFFFFF')
    axis_template = dict(
        tickfont=dict(size=tick_size, color=title_color),
        gridcolor=spine_color if style_params.get('show_grid', True) else 'rgba(0,0,0,0)',
        gridwidth=style_params.get('grid_width', 0.5),
        showgrid=style_params.get('show_grid', True),
        showline=True, linewidth=spine_width, linecolor=spine_color, zeroline=False
    )
    def make_axis(template, title_text, rng):
        d = template.copy()
        d['title'] = dict(text=title_text, font=dict(size=label_size, color=title_color))
        d['range'] = rng
        return d
    
    fig.update_layout(
        scene=dict(
            uirevision='constant_multi',
            xaxis=make_axis(axis_template, 'X (m)', [ext_x[0]-margin, ext_x[1]+margin]),
            yaxis=make_axis(axis_template, 'Y (m)', [ext_y[0]-margin, ext_y[1]+margin]),
            zaxis=make_axis(axis_template, 'Z (m)', [ext_z[0]-margin, ext_z[1]+margin]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            bgcolor=bg_color
        ),
        uirevision='constant_multi',
        title=dict(
            text=f'🔥 Multi-Slice 3D Thermal Field | T: {T_min:.1f} - {T_max:.1f} K | Mesh: {Nx}×{Ny}×{Nz} | {n_actual_slices} Z-slices',
            x=0.5,
            font=dict(size=title_size, color=title_color, family='Arial')
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            yanchor='top', y=0.99, xanchor='left', x=0.01,
            font=dict(size=style_params.get('legend_fontsize', 12)),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=spine_color, borderwidth=spine_width
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        sliders=sliders,
        updatemenus=updatemenus
    )
    return fig

def create_2d_heatmap_with_mesh(T_2d, extents_xy, style_params, 
                                 show_mesh=True, mesh_color='black',
                                 mesh_alpha=0.3, mesh_linewidth=0.5):
    import matplotlib.pyplot as plt
    cmap_name = style_params.get('cmap', 'hot')
    vmin, vmax = resolve_cbar_range(style_params, T_2d)
    fig, ax = plt.subplots(figsize=(10, 8))
    n_x, n_y = T_2d.shape
    x = np.linspace(extents_xy[0], extents_xy[1], n_x)
    y = np.linspace(extents_xy[2], extents_xy[3], n_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    if show_mesh:
        pcm = ax.pcolormesh(
            X, Y, T_2d,
            cmap=cmap_name,
            shading='nearest',
            vmin=vmin, vmax=vmax,
            edgecolors=mesh_color,
            linewidth=mesh_linewidth,
            alpha=1.0 - mesh_alpha
        )
        node_step_x = max(1, n_x // 10)
        node_step_y = max(1, n_y // 10)
        for i in range(0, n_x, node_step_x):
            for j in range(0, n_y, node_step_y):
                ax.plot(x[i], y[j], 'o', color=mesh_color, 
                       markersize=3, alpha=mesh_alpha + 0.2)
    else:
        pcm = ax.pcolormesh(X, Y, T_2d, cmap=cmap_name, shading='nearest',
                             vmin=vmin, vmax=vmax)
    extend = style_params.get('colorbar_extend', 'neither') \
             if style_params.get('use_custom_cbar_range', False) else 'neither'
    cbar = plt.colorbar(pcm, ax=ax, label='Temperature (K)',
                        shrink=0.85, extend=extend)
    cbar.ax.tick_params(labelsize=11)
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'2D Thermal Field (Mesh: {n_x}×{n_y}) | T: {T_2d.min():.1f} - {T_2d.max():.1f} K',
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    textstr = f'Grid: {n_x}×{n_y}\nΔx = {(x[1]-x[0])*1000:.2f} mm\nΔy = {(y[1]-y[0])*1000:.2f} mm'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    fig = EnhancedFigureStyler.apply_publication_styling(fig, ax, style_params)
    return fig

# -----------------------------------------------------------------------------
# 9. Simulation Runner (unchanged)
# -----------------------------------------------------------------------------
def run_simulation(params, progress_callback=None):
    tracemalloc.start()
    start_time = time.time()
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)
        cpu_start = psutil.cpu_percent(interval=None)
    else:
        mem_before = 0.0
        cpu_start = 0.0

    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    rho = params['rho']; Cp = params['Cp']
    kx, ky, kz = params['kx'], params['ky'], params['kz']
    T_amb = params['T_amb']; h_conv = params['h_conv']; eps = params['eps']
    q_normal = params['q_normal']
    reaction_params = params['reaction_params']
    dt_init = params['dt_init']; dt_min = params['dt_min']; dt_max = params['dt_max']
    t_max = params['t_max']; sample_interval = params['sample_interval']
    trigger_temp = params['trigger_temp']; trigger_radius = params['trigger_radius']
    R = 8.314; sigma = 5.67e-8

    sim_time = params.get('sim_time', t_max)
    snapshot_interval = params.get('snapshot_interval', 30.0)

    q_heater = params.get('q_heater', 0.0)
    heater_cutoff_temp = params.get('heater_cutoff_temp', 500.0)
    heater_face = params.get('heater_face', 0)
    use_heater = params.get('use_heater', False)

    local_heater = params.get('local_heater_enabled', False)
    q_loc = params.get('q_loc', 0.0)
    loc_radius = params.get('loc_radius', 3)
    loc_cutoff = params.get('loc_cutoff_temp', 520.0)

    cfl_factor = params.get('cfl_factor', 0.4)
    adapt_dt_thresh = params.get('adapt_dt_thresh', 600.0)
    adapt_dt_factor = params.get('adapt_dt_factor', 0.8)
    safe_T_limit = params.get('safe_T_limit', 1500.0)
    T_cap = params.get('T_cap', 1200.0)
    max_steps = params.get('max_steps', 2_000_000)
    wall_limit_s = params.get('wall_limit_s', 300.0)

    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1); dz = Lz / (Nz - 1)
    extents = {'x': (0, Lx), 'y': (0, Ly), 'z': (0, Lz)}

    if local_heater:
        T = np.full((Nx, Ny, Nz), T_amb, dtype=np.float64)
    else:
        T = initialize_temperature_field(
            Nx, Ny, Nz,
            T_amb=T_amb,
            trigger_temp=trigger_temp,
            trigger_radius=trigger_radius
        )

    alphas = initialize_reaction_degrees(
        Nx, Ny, Nz,
        trigger_radius=trigger_radius
    )

    loc_mask = np.zeros((Nx, Ny, Nz), dtype=np.bool_)
    if local_heater and loc_radius > 0:
        ii = np.arange(Nx)[:,None,None]
        jj = np.arange(Ny)[None,:,None]
        kk = np.arange(Nz)[None,None,:]
        dist = np.sqrt((ii - Nx//2)**2 + (jj - Ny//2)**2 + (kk - Nz//2)**2)
        loc_mask = dist <= loc_radius

    alpha_x = kx / (rho * Cp)
    alpha_y = ky / (rho * Cp)
    alpha_z = kz / (rho * Cp)
    dt_cfl = cfl_factor / (alpha_x/dx**2 + alpha_y/dy**2 + alpha_z/dz**2)
    dt = min(dt_init, dt_cfl, dt_max)

    t = 0.0; step = 0
    times = []; T_max_history = []
    T_mid_history = []; alpha_mid_history = []
    sample_next = 0.0
    mid_z = Nz // 2

    snapshots_3d = []
    snapshot_times = []
    next_snapshot_time = 0.0
    last_ui = time.time()

    while t < sim_time:
        T_max = np.max(T)
        if not np.isfinite(T_max) or T_max > safe_T_limit:
            break

        fuel_left = 1.0 - np.mean(alphas)
        if fuel_left < 0.02:
            break

        if T_max > adapt_dt_thresh and fuel_left > 0.02:
            dt = max(dt_min, dt * adapt_dt_factor)
        else:
            dt = min(dt_cfl, dt_max)

        heater_active = use_heater and (T_max < heater_cutoff_temp)

        if local_heater and np.any(loc_mask):
            isc_active = np.max(T[loc_mask]) < loc_cutoff
        else:
            isc_active = False

        T, alphas = step_3d(T, alphas, dt,
                           rho, Cp, kx, ky, kz, dx, dy, dz,
                           q_normal, reaction_params, T_amb, h_conv, eps, sigma, R,
                           q_heater, heater_active, heater_face,
                           q_loc, isc_active, loc_mask, T_cap)
        t += dt; step += 1

        if t >= sample_next:
            times.append(t)
            T_max_history.append(T_max)
            T_mid_history.append(T[:, :, mid_z].copy())
            alpha_mid_history.append(alphas[0, :, :, mid_z].copy())
            sample_next += sample_interval

        if t >= next_snapshot_time:
            snapshots_3d.append(T.astype(np.float32))
            snapshot_times.append(t)
            next_snapshot_time += snapshot_interval

        if step > max_steps or (time.time() - start_time) > wall_limit_s:
            break

        if progress_callback is not None and (time.time() - last_ui) > 0.25:
            progress_callback(min(t / sim_time, 1.0))
            last_ui = time.time()

    if len(snapshots_3d) == 0 or snapshot_times[-1] < t - dt:
        snapshots_3d.append(T.astype(np.float32))
        snapshot_times.append(t)

    history = []
    for idx in range(len(times)):
        history.append({
            'time': times[idx],
            'T_max': T_max_history[idx],
            'T_mid': T_mid_history[idx],
            'alpha_mid': alpha_mid_history[idx]
        })

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    if HAS_PSUTIL:
        mem_after = process.memory_info().rss / (1024**2)
        cpu_end = psutil.cpu_percent(interval=None)
        cpu_avg = (cpu_start + cpu_end) / 2.0
        os_mem_delta = mem_after - mem_before
        process_mem_mb = mem_after
    else:
        os_mem_delta = 0.0
        cpu_avg = None
        process_mem_mb = current_mem / (1024**2)

    array_mem_mb = (Nx * Ny * Nz * 8 * 5) / (1024**2)
    efficiency_stats = {
        'wall_time_s': end_time - start_time,
        'peak_memory_mb': peak_mem / (1024**2),
        'current_memory_mb': current_mem / (1024**2),
        'process_memory_mb': process_mem_mb,
        'array_memory_mb': array_mem_mb,
        'os_memory_delta_mb': os_mem_delta,
        'cpu_avg_percent': cpu_avg,
        'mesh_cells': Nx * Ny * Nz,
        'total_steps': step,
        'n_snapshots': len(snapshots_3d)
    }

    metadata = {
        'simulation_date': datetime.now().isoformat(),
        'final_time': t,
        'total_steps': step,
        'final_T_max': float(np.max(T)),
        'wall_time': end_time - start_time,
        'mesh_shape': (Nx, Ny, Nz),
        'dx_dy_dz': (dx, dy, dz),
        'extents': extents,
        'times': times,
        'T_max_history': T_max_history,
        'efficiency': efficiency_stats,
        'snapshot_interval': snapshot_interval
    }
    final_3D = (T.copy(), alphas.copy())
    return history, metadata, final_3D, snapshots_3d, snapshot_times

# -----------------------------------------------------------------------------
# 10. Enhanced Plotting Functions (unchanged)
# -----------------------------------------------------------------------------
def create_publication_heatmaps(simulations, frames, config, style_params):
    n_sims = len(simulations)
    cols = min(3, n_sims)
    rows = (n_sims + cols - 1) // cols
    styles = JournalTemplates.get_journal_styles()
    journal = style_params.get('journal_style', 'custom')
    fig_width = styles[journal]['figure_width_double'] / 2.54
    fig, axes = plt.subplots(rows, cols,
                             figsize=(fig_width, fig_width*0.8*rows/cols),
                             constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    cmap_name = style_params.get('cmap', 'hot')
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    if cmap_name in enhanced_cmaps:
        cmap = enhanced_cmaps[cmap_name]
    else:
        cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'hot'))

    if style_params.get('use_custom_cbar_range', False):
        vmin = float(style_params['cbar_t_min'])
        vmax = float(style_params['cbar_t_max'])
    else:
        all_T = [sim['history'][fi]['T_mid'] for sim, fi in zip(simulations, frames)]
        vmin = float(min(np.nanmin(t) for t in all_T))
        vmax = float(max(np.nanmax(t) for t in all_T))

    for idx, (sim, frame_idx) in enumerate(zip(simulations, frames)):
        row = idx // cols; col = idx % cols
        ax = axes[row, col]
        T_mid = sim['history'][frame_idx]['T_mid']
        ext = sim['metadata']['extents']
        extent_xy = [ext['x'][0], ext['x'][1], ext['y'][0], ext['y'][1]]
        if style_params.get('apply_smoothing', True):
            T_mid = gaussian_filter(T_mid, sigma=1)
        im = ax.imshow(T_mid, extent=extent_xy,
                       cmap=cmap, origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax)
        PublicationEnhancer.add_scale_bar(ax, 0.01, location='lower right', color='white', label='m')
        ax.set_title(sim['params'].get('label', ''))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, label='Temperature (K)',
                     shrink=style_params.get('colorbar_shrink', 0.8),
                     pad=style_params.get('colorbar_pad', 0.05),
                     extend=style_params.get('colorbar_extend', 'neither')
                            if style_params.get('use_custom_cbar_range', False) else 'neither')
    for idx in range(n_sims, rows*cols):
        row = idx // cols; col = idx % cols
        axes[row, col].axis('off')
    fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
    return fig

def create_enhanced_line_profiles(simulations, frames, config, style_params):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulations)))
    profile_type = config.get('profile_direction', 'z')
    for idx, (sim, frame_idx, color) in enumerate(zip(simulations, frames, colors)):
        T_final = sim['final_3D'][0]
        ext = sim['metadata']['extents']
        center = (ext['x'][1]/2, ext['y'][1]/2, ext['z'][1]/2)
        dist, profile, _ = ThermalLineProfiler3D.extract_profile(
            T_final, profile_type, center, ext, offset_ratio=0.5)
        ax.plot(dist, profile, color=color, linewidth=style_params.get('line_width', 2.0),
                label=f"{sim['params'].get('label', '')}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Thermal Profile along {profile_type.upper()}")
    if style_params.get('show_legend', True):
        ax.legend(loc=style_params.get('legend_location', 'best'), fontsize=style_params.get('legend_fontsize', 12))
    fig = EnhancedFigureStyler.apply_publication_styling(fig, ax, style_params)
    return fig

def create_publication_statistics(simulations, frames, config, style_params):
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    ax_box = fig.add_subplot(gs[0, 0])
    ax_violin = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_cdf = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1:3])
    ax_corr = fig.add_subplot(gs[2, 0:2])
    ax_qq = fig.add_subplot(gs[2, 2])

    all_data = []; labels = []
    for sim, frame_idx in zip(simulations, frames):
        T_final = sim['final_3D'][0]
        flat = T_final.flatten()
        flat = flat[np.isfinite(flat)]
        all_data.append(flat)
        labels.append(sim['params'].get('label', ''))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    bp = ax_box.boxplot(all_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax_box.set_title("Temperature Distribution")
    ax_box.set_ylabel("T (K)")

    parts = ax_violin.violinplot(all_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
    ax_violin.set_title("Violin Plot")
    ax_violin.set_xticks(range(1, len(labels)+1))
    ax_violin.set_xticklabels(labels, rotation=45, ha='right')

    for data, color, label in zip(all_data, colors, labels):
        ax_hist.hist(data, bins=30, density=True, alpha=0.5, color=color, label=label)
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 200)
        ax_hist.plot(x_range, kde(x_range), color=color, linewidth=2)
    ax_hist.set_title("Density + KDE")
    ax_hist.legend(fontsize=8)

    for data, color, label in zip(all_data, colors, labels):
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data)+1)/len(sorted_data)
        ax_cdf.plot(sorted_data, y, color=color, label=label)
    ax_cdf.set_title("CDF")
    ax_cdf.set_xlabel("T (K)")
    ax_cdf.set_ylabel("Probability")

    ax_table.axis('off')
    stats_data = []
    for data, label in zip(all_data, labels):
        stats_data.append([
            label,
            len(data),
            f"{np.mean(data):.1f}",
            f"{np.std(data):.1f}",
            f"{np.median(data):.1f}",
            f"{np.max(data):.1f}",
            f"{stats.skew(data):.3f}",
            f"{stats.kurtosis(data):.3f}"
        ])
    table = ax_table.table(cellText=stats_data,
                           colLabels=['Label','N','Mean','Std','Median','Max','Skew','Kurtosis'],
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9)

    if len(simulations) > 0:
        sim0 = simulations[0]
        T_final = sim0['final_3D'][0]
        alpha_final = sim0['final_3D'][1][0]
        flat_T = T_final.flatten()
        flat_alpha = alpha_final.flatten()
        idx = np.random.choice(len(flat_T), min(5000, len(flat_T)), replace=False)
        ax_corr.scatter(flat_T[idx], flat_alpha[idx], s=1, alpha=0.5, c='blue')
        ax_corr.set_xlabel("Temperature (K)")
        ax_corr.set_ylabel("SEI α")
        ax_corr.set_title("T vs α_SEI")

        stats.probplot(flat_T[idx], dist="norm", plot=ax_qq)
        ax_qq.get_lines()[0].set_marker('.')
        ax_qq.get_lines()[0].set_markersize(2)
        ax_qq.get_lines()[0].set_alpha(0.5)
        ax_qq.get_lines()[1].set_color('red')
        ax_qq.set_title("Q-Q Plot (T)")

    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax_box, ax_violin, ax_hist, ax_cdf, ax_corr, ax_qq], style_params)
    return fig

def create_evolution_timeline_plot(simulations, config, style_params):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulations)))
    for sim, color in zip(simulations, colors):
        times = sim['metadata']['times']
        Tmax = sim['metadata']['T_max_history']
        ax.plot(times, Tmax, color=color, linewidth=style_params.get('line_width', 2.0),
                label=sim['params'].get('label', ''))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Maximum Temperature (K)")
    ax.axhline(400, color='orange', linestyle='--', label='SEI onset')
    ax.axhline(462, color='darkred', linestyle='--', label='Runaway threshold')
    if style_params.get('show_legend', True):
        ax.legend(loc=style_params.get('legend_location', 'best'))
    fig = EnhancedFigureStyler.apply_publication_styling(fig, ax, style_params)
    return fig

def create_cross_correlation_plot(simulations, config, style_params):
    param_options = ['h_conv', 'trigger_temp', 'Lx', 'Ly', 'Lz', 'rho', 'Cp', 'kx', 'ky', 'kz']
    x_param = config.get('x_param', 'h_conv')
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[0, 1])
    ax_hist_resid = fig.add_subplot(gs[1, 0])
    ax_qq = fig.add_subplot(gs[1, 1])

    x_vals = []; y_vals = []; labels = []
    for sim in simulations:
        params = sim['params']
        if x_param in params:
            x_vals.append(params[x_param])
            y_vals.append(sim['metadata']['final_T_max'])
            labels.append(params.get('label', ''))
    x_vals = np.array(x_vals); y_vals = np.array(y_vals)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    ax_scatter.scatter(x_vals, y_vals, c='blue', s=50, alpha=0.7)
    for xi, yi, lab in zip(x_vals, y_vals, labels):
        ax_scatter.annotate(lab, (xi, yi), fontsize=8, alpha=0.8)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    y_line = slope * x_line + intercept
    ax_scatter.plot(x_line, y_line, color='red', linestyle='--', label=f'R²={r_value**2:.3f}')
    ax_scatter.set_xlabel(x_param)
    ax_scatter.set_ylabel("Final Tmax (K)")
    ax_scatter.set_title("Parameter vs Final Temperature")
    ax_scatter.legend()

    y_pred = slope * x_vals + intercept
    residuals = y_vals - y_pred
    ax_resid.scatter(y_pred, residuals, color='blue', alpha=0.7)
    ax_resid.axhline(y=0, color='red', linestyle='--')
    ax_resid.set_xlabel("Predicted Tmax")
    ax_resid.set_ylabel("Residuals")
    ax_resid.set_title("Residual Plot")

    ax_hist_resid.hist(residuals, bins=10, density=True, alpha=0.7, color='blue')
    ax_hist_resid.set_xlabel("Residual")
    ax_hist_resid.set_ylabel("Density")
    ax_hist_resid.set_title("Residual Distribution")

    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_marker('.')
    ax_qq.get_lines()[0].set_markersize(5)
    ax_qq.get_lines()[0].set_alpha(0.5)
    ax_qq.get_lines()[1].set_color('red')
    ax_qq.set_title("Q-Q Plot (Residuals)")

    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax_scatter, ax_resid, ax_hist_resid, ax_qq], style_params)
    return fig

def create_publication_correlation(simulations, frames, config, style_params):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulations)))
    for sim, frame_idx, color in zip(simulations, frames, colors):
        T_final = sim['final_3D'][0]
        alpha_final = sim['final_3D'][1][0]
        flat_T = T_final.flatten()
        flat_alpha = alpha_final.flatten()
        idx = np.random.choice(len(flat_T), min(5000, len(flat_T)), replace=False)
        ax.scatter(flat_T[idx], flat_alpha[idx], s=1, alpha=0.5, color=color,
                   label=sim['params'].get('label', ''))
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("SEI Conversion α")
    ax.set_title("T vs α_SEI")
    if style_params.get('show_legend', True):
        ax.legend(loc=style_params.get('legend_location', 'best'))
    fig = EnhancedFigureStyler.apply_publication_styling(fig, ax, style_params)
    return fig

# -----------------------------------------------------------------------------
# 11. Helper function for time‑series plot with marker
# -----------------------------------------------------------------------------
def create_time_series_with_marker(sim_data, current_time, style_params):
    history = sim_data['history']
    if not history:
        return None
    times = [h['time'] for h in history]
    Tmax = [h['T_max'] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=Tmax,
        mode='lines',
        name='T_max',
        line=dict(color='red', width=3)
    ))
    fig.add_vline(x=current_time, line_dash="dash", line_color="green",
                  annotation_text=f"t={current_time:.1f}s")

    fig.add_hline(y=450, line_dash="dot", line_color="orange", annotation_text="SEI onset (450K)")
    fig.add_hline(y=523, line_dash="dot", line_color="red", annotation_text="Electrolyte boil (523K)")
    fig.add_hline(y=773, line_dash="dot", line_color="darkred", annotation_text="Fire (773K)")

    fig.update_layout(
        title="Temperature Evolution",
        xaxis_title="Time (s)",
        yaxis_title="Maximum Temperature (K)",
        height=250,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# 12. Main UI – with Drone Battery Model integration and st.form for structural controls
# -----------------------------------------------------------------------------
advanced_styling = get_styling_controls()

# ---- Preset system: initialise all preset keys with valid defaults ----
preset_defaults = {
    'trigger_temp': 450,
    'trigger_radius': 3,
    'h_conv': 5.0,
    'eps': 0.9,
    'kx': 25.0,
    'ky': 25.0,
    'kz': 1.5,
    'T_cap': 1200,
    'safe_T_limit': 1500,
    'reaction_preset': 'Realistic (Recommended)',
}
for key, default in preset_defaults.items():
    if f'preset_{key}' not in st.session_state:
        st.session_state[f'preset_{key}'] = default

def apply_preset(preset_name):
    presets = {
        "Default (Symmetric)": {
            'trigger_temp': 450,
            'trigger_radius': 3,
            'h_conv': 5.0,
            'eps': 0.9,
            'kx': 25.0,
            'ky': 25.0,
            'kz': 1.5,
            'T_cap': 1200,
            'safe_T_limit': 1500,
            'reaction_preset': 'Realistic (Recommended)',
        },
        "Hotspot High‑Rise": {
            'trigger_temp': 650,
            'trigger_radius': 6,
            'h_conv': 1.0,
            'eps': 0.2,
            'kx': 10.0,
            'ky': 10.0,
            'kz': 0.6,
            'T_cap': 2000,
            'safe_T_limit': 2500,
            'reaction_preset': 'High‑Rise (Hotspot only)',
        }
    }
    if preset_name in presets:
        for key, val in presets[preset_name].items():
            st.session_state[f'preset_{key}'] = val
        st.rerun()

operation_mode = st.sidebar.radio(
    "Operation Mode",
    ["Run New Simulation", "Compare Saved Simulations"],
    index=0
)

if operation_mode == "Run New Simulation":
    st.sidebar.header("🎛️ New Simulation Setup")

    # Drone Model Selector
    selected_model_name = st.sidebar.selectbox(
        "🚁 Select Drone Battery Model",
        list(DRONE_BATTERY_MODELS.keys()),
        index=0,
        help="Selecting a model auto-fills physical dimensions, density, and energy based on drone class."
    )
    model = DRONE_BATTERY_MODELS[selected_model_name]
    
    # Auto-calculate physics based on model
    Lx_m = model['L_mm'] / 1000.0
    Ly_m = model['W_mm'] / 1000.0
    Lz_m = model['H_mm'] / 1000.0
    Vol_m3 = Lx_m * Ly_m * Lz_m
    rho_calc = model['Weight_g'] / 1000.0 / Vol_m3 # kg/m3
    
    E_elec_J = model['V_nom'] * (model['Cap_mAh']/1000.0) * 3600
    E_thermal_J = E_elec_J * 1.5  # Thermal energy is ~1.5x electrical
    Sigma_H = E_thermal_J / Vol_m3 # Volumetric Heat (J/m3)
    
    # Suggested Mesh (Target ~2.0mm per cell to balance speed/accuracy)
    Nx_def = max(20, int(model['L_mm'] / 2.0))
    Ny_def = max(20, int(model['W_mm'] / 2.0))
    Nz_def = max(10, int(model['H_mm'] / 2.0))
    
    st.sidebar.info(f"🔋 **Model Physics:**\n"
                    f"Energy: {E_elec_J/3600:.1f} Wh\n"
                    f"Density: {rho_calc:.0f} kg/m³\n"
                    f"Volumetric Heat (ΣH): {Sigma_H:.2e} J/m³")

    # Quick Preset selector (still available)
    preset_options = ["Default (Symmetric)", "Hotspot High‑Rise"]
    selected_preset = st.sidebar.selectbox(
        "⚡ Quick Preset",
        preset_options,
        index=0,
        help="Automatically sets key parameters for the chosen scenario. "
             "High‑Rise gives ~1100–1400 K peak from hotspot alone."
    )
    if st.sidebar.button("Apply Preset"):
        apply_preset(selected_preset)

    with st.sidebar.expander("⏱️ Time & Mesh", expanded=True):
        sim_time = st.slider("Total Simulation Time (s)", 
                             min_value=10, max_value=600, value=120, step=10,
                             help="Most dynamics finish within 30-60 s; loop exits early when fuel is exhausted.")
        snapshot_interval = st.slider("3D Snapshot Interval (s)",
                                      min_value=1, max_value=120, value=2, step=1,
                                      help="How often to save 3D field for time slider")
        col1, col2, col3 = st.columns(3)
        with col1:
            Nx = st.number_input("Nx", 10, 150, Nx_def, 5)
        with col2:
            Ny = st.number_input("Ny", 10, 150, Ny_def, 5)
        with col3:
            Nz = st.number_input("Nz", 5, 80, Nz_def, 5)
        col1, col2, col3 = st.columns(3)
        with col1:
            Lx = st.number_input("Length (m)", 0.005, 0.300, Lx_m, 0.001, format="%.3f")
        with col2:
            Ly = st.number_input("Width (m)", 0.005, 0.300, Ly_m, 0.001, format="%.3f")
        with col3:
            Lz = st.number_input("Thickness (m)", 0.003, 0.200, Lz_m, 0.001, format="%.3f")

    with st.sidebar.expander("Material & Boundary"):
        rho = st.number_input("Density (kg/m³)", 1000.0, 3000.0, float(rho_calc), 10.0)
        Cp = st.number_input("Cp (J/kg·K)", 500.0, 2000.0, 1100.0, 50.0)
        kx = st.number_input("k_x (W/m·K)", 5.0, 60.0, float(model['kx']), 1.0)
        ky = st.number_input("k_y (W/m·K)", 5.0, 60.0, float(model['ky']), 1.0)
        kz = st.number_input("k_z (W/m·K)", 0.5, 5.0, float(model['kz']), 0.1)
        T_amb = st.number_input("Ambient T (K)", 250, 350, 298, 1)
        h_conv = st.number_input("h_conv (W/m²·K)", 0.0, 50.0, float(model['h_conv']), 1.0,
                                 help="Use ~5 for near-adiabatic (ARC-style), ~15 for natural convection, ~25 for drone prop airflow")
        eps = st.number_input("Emissivity", 0.05, 0.95, 0.9, 0.05)

    # ===== "Heat & Trigger" =====
    with st.sidebar.expander("🔥 Trigger & Heat Sources", expanded=True):
        st.markdown("**Central Hotspot (imposed T)** – use if you want to start with a heated blob.")
        q_normal = st.number_input("Normal Heat (W/m³)", 0.0, 5e5, 0.0, 1e4, format="%.0f",
                                   help="Background heat generation (usually set to 0)")
        trigger_temp = st.number_input("Hotspot T (K)", 250, 2000, 450, 5,
                                       help="Central hotspot temperature. Set >400 K for immediate runaway.")
        trigger_radius = st.slider("Hotspot radius (cells)", 1, 10, 3,
                                   help="Radius of the spherical hotspot in grid cells.")
        st.markdown("---")
        
        st.markdown(f"**Localized ISC Heater (Auto-scaled for {selected_model_name[:8]})**")
        local_heater = st.checkbox("Enable localized ISC heater at defect", value=True,
                                   help="Volumetric Joule heating in a small sphere; fuses open at cutoff.")
        
        # Auto-calculate ISC power based on drone model
        isc_current = model['C_rating'] * (model['Cap_mAh']/1000.0)
        isc_power_W = isc_current * 3.7  # Single cell short voltage
        dx_tmp = Lx / (Nx - 1) if Nx > 1 else 0.001
        isc_vol_m3 = (4/3) * np.pi * (3 * dx_tmp)**3  # Assuming 3 cell radius
        suggested_q_loc = int(isc_power_W / isc_vol_m3) if isc_vol_m3 > 0 else 5e9
        
        st.info(f"💡 **Physics Suggestion:** For a {model['C_rating']}C short ({isc_current:.0f}A = {isc_power_W:.0f}W), "
                f"set `q_loc` ≈ {suggested_q_loc:.2e} W/m³.")
        
        q_loc = st.number_input("ISC power density (W/m³)", 0, int(1e11), suggested_q_loc,
                                format="%.0e", disabled=not local_heater)
        loc_radius = st.slider("ISC radius (cells)", 1, 8, 3, disabled=not local_heater)
        loc_cutoff = st.number_input("ISC cutoff T (K)", 350, 900, 520, 5,
                                     disabled=not local_heater,
                                     help="Short fuses open once local T exceeds this; chemistry then self-propagates")
        st.info("If ISC heater is enabled, the imposed hotspot is ignored and the cell starts uniformly at ambient T.")

    # ===== Sustained Heater (optional) =====
    with st.sidebar.expander("🔥 Sustained Heater (Abuse Protocol)", expanded=True):
        use_heater = st.checkbox("Enable Boundary Heater", value=False,
                                 help="Sustained heat flux into a face (lateral overheating / ARC-style). "
                                      "Turn OFF for symmetric central runaway.")
        q_heater = st.number_input("Heater Flux (W/m²)", 0.0, 5e5, 3e4, 5e3, format="%.0f",
                                   help="Typical range 3e4–1e5 W/m²", disabled=not use_heater)
        heater_cutoff_temp = st.number_input("Heater Cutoff (K)", 300, 1000, 500, 10,
                                             help="Heater turns off when T_max exceeds this", disabled=not use_heater)
        heater_face = st.selectbox("Heater Face", 
                                   ["-X (left)", "+X (right)", "-Y (back)", "+Y (front)", "-Z (bottom)", "+Z (top)"],
                                   index=0, disabled=not use_heater)
        heater_face_map = {"-X (left)":0, "+X (right)":1, "-Y (back)":2, "+Y (front)":3, "-Z (bottom)":4, "+Z (top)":5}
        heater_face_val = heater_face_map[heater_face] if use_heater else 0

    with st.sidebar.expander("Time Stepping"):
        dt_init = st.number_input("dt_init (s)", 0.001, 0.1, 0.01, 0.005, format="%.3f")
        dt_min = st.number_input("dt_min (s)", 1e-7, 1e-4, 1e-6, step=1e-7, format="%.1e")
        dt_max = st.number_input("dt_max (s)", 0.001, 0.1, 0.01, 0.005, format="%.3f")
        sample_interval = st.number_input("Sample interval (s)", 0.1, 10.0, 0.5, 0.1)

    # Advanced Numerics
    with st.sidebar.expander("⚙️ Advanced Numerics", expanded=False):
        cfl_factor = st.slider("CFL Safety Factor", 0.1, 0.45, 0.4, 0.05)
        adapt_dt_thresh = st.slider("Adaptive dt Threshold (K)", 400, 1000, 600, 10)
        adapt_dt_factor = st.slider("dt Shrink Factor", 0.5, 0.95, 0.8, 0.05)
        safe_T_limit = st.slider("Safety Cutoff Temp (K)", 1000, 3000, 1500, 50,
                                 help="Raise to 2500 K for high‑rise presets.")
        T_cap = st.slider("Temperature Cap (K)", 800, 3000, 1200, 50,
                          help="Prevents numerical overshoot; must be above physical peak.")
        max_steps = st.number_input("Max Steps (break)", 100000, 10000000, 2000000, 500000,
                                    help="Safety break if step count exceeds this.")
        wall_limit_s = st.number_input("Wall‑time Limit (s)", 30, 600, 300, 30,
                                       help="Hard wall‑clock cap to prevent endless loops.")

    # Reaction Kinetics Preset – with auto‑scale option
    with st.sidebar.expander("🔬 Reaction Kinetics Preset", expanded=True):
        reaction_preset_options = ['Auto-Scale H for Drone Model (Recommended)', 'Realistic (Base)', 'Aggressive (Faster)', 'Custom']
        reaction_preset = st.radio("Select Reaction Parameters", reaction_preset_options, index=0)
        
        if reaction_preset == 'Auto-Scale H for Drone Model (Recommended)':
            reaction_params = REALISTIC_REACTION_PARAMS.copy()
            # Base Model A Sigma_H is ~1.65e9. Calculate scale factor.
            scale_factor = Sigma_H / 1.65e9
            reaction_params[:, 2] *= scale_factor
            st.success(f"🔋 ΣH auto-scaled by {scale_factor:.2f}x to match {selected_model_name[:15]}... ({Sigma_H:.2e} J/m³)")
        elif reaction_preset == 'Realistic (Base)':
            reaction_params = REALISTIC_REACTION_PARAMS.copy()
            st.info("✅ Should reach 600‑900 K in 5‑15 min with heater enabled")
        elif reaction_preset == 'Aggressive (Faster)':
            reaction_params = AGGRESSIVE_REACTION_PARAMS.copy()
            st.warning("⚡ Faster runaway, may overshoot")
        else:  # Custom
            reaction_params = DEFAULT_REACTION_PARAMS.copy()
            st.info("Adjust each reaction below")
            with st.expander("Custom Reaction Parameters"):
                for i, name in enumerate(['SEI', 'Electrolyte', 'Separator', 'Cathode']):
                    st.write(f"**{name}**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        reaction_params[i,0] = st.number_input(f"Ea{i} (J/mol)", value=float(reaction_params[i,0]), key=f'ea_{i}')
                    with c2:
                        reaction_params[i,1] = st.number_input(f"A{i} (1/s)", value=float(reaction_params[i,1]), format='%e', key=f'a_{i}')
                    with c3:
                        reaction_params[i,2] = st.number_input(f"H{i} (W/m³)", value=float(reaction_params[i,2]), key=f'h_{i}')

    # Label
    label = st.sidebar.text_input("Run Label (optional)", value=f"{selected_model_name[:15]} h={h_conv:.1f} trig={'ISC' if local_heater else f'{trigger_temp:.0f}K'}")

    # Pre‑Simulation Diagnostics
    with st.sidebar.expander("🔍 Pre‑Simulation Diagnostics", expanded=False):
        if local_heater:
            st.write("**Trigger:** Localized ISC heater (power‑controlled)")
            st.write(f"**Power density:** {q_loc:.1e} W/m³")
            st.write(f"**ISC radius:** {loc_radius} cells")
            st.write(f"**Cutoff T:** {loc_cutoff} K")
            if q_loc < 3e9:
                st.warning("⚠️ Power may be too low for ignition; consider increasing.")
        else:
            st.write(f"**Trigger Temperature:** {trigger_temp} K = {trigger_temp-273.15:.0f} °C")
            if trigger_temp < 420 and trigger_radius > 0:
                st.error("⚠️ Trigger too low! Use ≥ 420 K for immediate runaway.")
            elif trigger_temp < 450 and trigger_radius > 0:
                st.warning("⚠️ Lower end – consider 450+ K for robust runaway.")
            else:
                st.success("✅ Realistic central hotspot trigger.")
        st.write(f"**Ambient:** {T_amb} K = {T_amb-273.15:.0f} °C")
        st.write(f"**Mesh cells:** {Nx*Ny*Nz:,}")
        st.write(f"**Snapshots (approx):** {int(sim_time / snapshot_interval)}")
        if use_heater:
            st.info(f"🔥 Heater active on {heater_face} at {q_heater:.0f} W/m², cutoff {heater_cutoff_temp} K")
            st.warning("⚠️ Heater ON → asymmetric heating. For symmetric central runaway, turn OFF the heater.")
        else:
            st.info("ℹ️ Symmetric central runaway mode (Heater OFF).")

    # Domain Sketch
    st.subheader("📐 Initial Domain Sketch (3D Interactive)")
    sketch_params = {
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'T_amb': T_amb,
        'trigger_radius': trigger_radius,
        'local_heater_enabled': local_heater,
        'loc_radius': loc_radius if local_heater else 0,
    }
    fig_3d = plot_3d_domain_sketch(sketch_params)
    st.plotly_chart(fig_3d, use_container_width=True, key='domain_sketch_chart')

    # Efficiency Monitor
    if 'last_efficiency' in st.session_state:
        st.subheader("⚡ Compute Efficiency Monitor (Last Run)")
        eff = st.session_state['last_efficiency']
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Wall Time", f"{eff['wall_time_s']:.2f} s")
        col2.metric("Peak RAM (MB)", f"{eff['peak_memory_mb']:.1f} MB")
        col3.metric("Process RAM (MB)", f"{eff['process_memory_mb']:.1f} MB")
        col4.metric("Grid Arrays (MB)", f"{eff['array_memory_mb']:.1f} MB")
        col5.metric("Total Cells", f"{eff['mesh_cells']:,}")
        if eff.get('cpu_avg_percent') is not None:
            col6, col7 = st.columns(2)
            col6.metric("Avg CPU (%)", f"{eff['cpu_avg_percent']:.1f} %")
            col7.metric("OS RAM Delta (MB)", f"{eff['os_memory_delta_mb']:.1f} MB")
        with st.expander("📊 Detailed Efficiency Metrics & JSON"):
            st.json(eff)

    # Run Button
    if st.sidebar.button("🚀 Run & Save", type="primary"):
        params = {
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
            'rho': rho, 'Cp': Cp,
            'kx': kx, 'ky': ky, 'kz': kz,
            'T_amb': T_amb,
            'h_conv': h_conv,
            'eps': eps,
            'q_normal': q_normal,
            'reaction_params': reaction_params,
            'dt_init': dt_init,
            'dt_min': dt_min,
            'dt_max': dt_max,
            't_max': sim_time,
            'sim_time': sim_time,
            'sample_interval': sample_interval,
            'trigger_temp': trigger_temp,
            'trigger_radius': trigger_radius,
            'label': label,
            'cfl_factor': cfl_factor,
            'adapt_dt_thresh': adapt_dt_thresh,
            'adapt_dt_factor': adapt_dt_factor,
            'safe_T_limit': safe_T_limit,
            'T_cap': T_cap,
            'max_steps': max_steps,
            'wall_limit_s': wall_limit_s,
            'snapshot_interval': snapshot_interval,
            'use_heater': use_heater,
            'q_heater': q_heater if use_heater else 0.0,
            'heater_cutoff_temp': heater_cutoff_temp if use_heater else 0.0,
            'heater_face': heater_face_val if use_heater else 0,
            # ISC heater params
            'local_heater_enabled': local_heater,
            'q_loc': q_loc if local_heater else 0.0,
            'loc_radius': loc_radius if local_heater else 0,
            'loc_cutoff_temp': loc_cutoff if local_heater else 0.0,
        }

        progress_bar = st.progress(0.0)
        live_metrics = st.empty()
        start_time = time.time()

        def update_progress(fraction):
            progress_bar.progress(min(fraction, 1.0), text=f"Running... {fraction*100:.1f}%")
            elapsed = time.time() - start_time
            if fraction > 0:
                eta = elapsed / fraction - elapsed
                live_metrics.info(f"⏱️ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

        with st.spinner("Running 3D thermal runaway simulation..."):
            history, metadata, final_3D, snapshots_3d, snapshot_times = run_simulation(
                params, progress_callback=update_progress
            )
            sim_id = SimulationDB.save_simulation(
                params, history, metadata, final_3D, snapshots_3d, snapshot_times
            )
            st.session_state['last_efficiency'] = metadata['efficiency']

        progress_bar.empty()
        live_metrics.success(f"✅ Done in {metadata['efficiency']['wall_time_s']:.2f}s – {len(snapshots_3d)} snapshots stored")
        time.sleep(0.5)
        st.rerun()

    # Saved Simulations and visualisation
    st.header("📋 Saved Simulations")
    sims = SimulationDB.get_simulation_list()
    if sims:
        df = pd.DataFrame([{'ID': s['id'], 'Name': s['name']} for s in sims])
        st.dataframe(df, width="stretch")
        with st.expander("🗑️ Delete Simulations"):
            to_delete = st.multiselect("Select to delete", [s['name'] for s in sims])
            if st.button("Delete Selected"):
                for name in to_delete:
                    for s in sims:
                        if s['name'] == name:
                            SimulationDB.delete_simulation(s['id'])
                st.rerun()
    else:
        st.info("No simulations saved yet.")

    # “Use Global Min/Max” button logic
    if advanced_styling.get('cbar_auto_from_global', False) and sims:
        latest_id = sims[-1]['id']
        sim_data = SimulationDB.get_all_simulations()[latest_id]
        T_global = sim_data['final_3D'][0]
        advanced_styling['cbar_t_min'] = float(np.min(T_global))
        advanced_styling['cbar_t_max'] = float(np.max(T_global))
        advanced_styling['use_custom_cbar_range'] = True
        st.rerun()

    if sims:
        latest_id = sims[-1]['id']
        sim_data = SimulationDB.get_all_simulations()[latest_id]
        T_final = sim_data['final_3D'][0]
        ext = sim_data['metadata']['extents']
        mid_z = sim_data['metadata']['mesh_shape'][2] // 2
        alphas_final = sim_data['final_3D'][1]
        mesh_shape = sim_data['metadata']['mesh_shape']

        st.subheader("🔬 Advanced 3D Volumetric Studio (with Time Slider)")

        has_snapshots = 'snapshots_3d' in sim_data and len(sim_data['snapshots_3d']) > 0

        with st.expander("⚙️ 3D Visualization Controls", expanded=True):
            tabs = st.tabs([
                "🔄 Multi‑Slice (Time Slider)",
                "📐 Single Slice + Wireframe",
                "🎯 Isosurface (Smooth)",
                "📊 2D Heatmap (Mesh Visible)"
            ])

            with tabs[0]:
                if has_snapshots:
                    st.markdown("**Navigate through time using the Plotly slider below.**")
                    snapshots = sim_data['snapshots_3d']
                    times = sim_data['snapshot_times']
                    
                    # Structural controls in a form to avoid rerun on change
                    with st.form(key='multi_slice_form'):
                        col1, col2 = st.columns(2)
                        with col1:
                            n_slices = st.slider("Z‑slices", 1, min(20, snapshots[0].shape[2]), 5, key='n_slices_ms')
                        with col2:
                            show_cross = st.checkbox("Show X/Y cross‑slices", value=False, key='show_cross_ms')
                        apply_changes = st.form_submit_button("Apply Structural Changes")
                    
                    # Use the latest values from the form
                    if apply_changes or 'multi_slice_applied' not in st.session_state:
                        st.session_state['multi_slice_applied'] = True
                        st.session_state['n_slices_ms'] = n_slices
                        st.session_state['show_cross_ms'] = show_cross
                    else:
                        n_slices = st.session_state.get('n_slices_ms', 5)
                        show_cross = st.session_state.get('show_cross_ms', False)
                    
                    fig_ms = create_multi_slice_3d_visualization_frames(
                        snapshots, times, ext, advanced_styling,
                        n_slices=n_slices,
                        show_cross_slices=show_cross
                    )
                    st.plotly_chart(fig_ms, use_container_width=True, key='multi_slice_chart')
                    
                    # Also show the time-series with marker (optional)
                    current_time = times[-1]  # latest snapshot
                    ts_fig = create_time_series_with_marker(sim_data, current_time, advanced_styling)
                    if ts_fig:
                        st.plotly_chart(ts_fig, use_container_width=True, key='time_series_chart')
                else:
                    st.warning("This simulation has no 3D snapshots. Re‑run with snapshot storage enabled.")
                    fig_ms = create_multi_slice_3d_visualization_frames(
                        [T_final], [sim_data['metadata']['final_time']], ext, advanced_styling,
                        n_slices=5, show_cross_slices=False
                    )
                    st.plotly_chart(fig_ms, use_container_width=True, key='multi_slice_chart')

            with tabs[1]:
                # Structural controls in a form
                with st.form(key='single_slice_form'):
                    col1, col2 = st.columns(2)
                    with col1:
                        slice_axis = st.selectbox("Slice Axis", ['z', 'y', 'x'], key='slice_ax_single')
                    with col2:
                        slice_pos = st.slider("Slice Position", 0.1, 0.9, 0.5, key='slice_pos_single')
                    show_mesh = st.checkbox("Show Mesh Wireframe", value=True, key='show_mesh_single')
                    mesh_opacity = st.slider("Mesh Opacity", 0.1, 0.8, 0.4, key='mesh_op_single') if show_mesh else 0.0
                    apply_single = st.form_submit_button("Apply")
                
                fig_sw = create_mesh_aware_3d_thermal(
                    T_final, ext, advanced_styling,
                    show_mesh=show_mesh,
                    mesh_opacity=mesh_opacity,
                    slice_axis=slice_axis,
                    slice_position=slice_pos
                )
                st.plotly_chart(fig_sw, use_container_width=True, key='single_slice_chart')

            with tabs[2]:
                # Structural controls in a form
                with st.form(key='iso_form'):
                    cmap_3d = st.selectbox("3D Colormap", cmap_list, index=cmap_list.index('inferno'), key='cmap_iso')
                    apply_iso = st.form_submit_button("Apply")
                
                plotly_cmap = matplotlib_to_plotly(cmap_3d)
                if advanced_styling.get('use_custom_cbar_range', False):
                    lo = float(advanced_styling['cbar_t_min'])
                    hi = float(advanced_styling['cbar_t_max'])
                else:
                    lo, hi = float(T_final.min()), float(T_final.max())
                iso_levels = np.linspace(lo, hi, 7)[1:-1].tolist()

                Nx, Ny, Nz = T_final.shape
                X = np.linspace(ext['x'][0], ext['x'][1], Nx)
                Y = np.linspace(ext['y'][0], ext['y'][1], Ny)
                Z = np.linspace(ext['z'][0], ext['z'][1], Nz)
                Xg, Yg, Zg = np.meshgrid(X, Y, Z, indexing='ij')

                fig_iso = go.Figure()
                for lvl in iso_levels:
                    if T_final.max() > lvl:
                        fig_iso.add_trace(go.Isosurface(
                            x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
                            value=T_final.flatten(),
                            isomin=lvl-5, isomax=lvl+5,
                            opacity=0.3, colorscale=plotly_cmap,
                            showscale=False, name=f'T = {lvl:.0f} K'
                        ))
                fig_iso.update_layout(
                    scene=dict(
                        uirevision='constant_iso',
                        xaxis=dict(title='x (m)'),
                        yaxis=dict(title='y (m)'),
                        zaxis=dict(title='z (m)'),
                        aspectmode='data'
                    ),
                    uirevision='constant_iso',
                    title=dict(text=f'🔥 Isosurfaces (Smooth) | Range: {lo:.0f}–{hi:.0f} K', x=0.5),
                    height=700
                )
                st.plotly_chart(fig_iso, use_container_width=True, key='isosurface_chart')

            with tabs[3]:
                # 2D heatmap controls in a form
                with st.form(key='heatmap_form'):
                    T_mid = T_final[:, :, mid_z]
                    extent_xy = [ext['x'][0], ext['x'][1], ext['y'][0], ext['y'][1]]
                    show_mesh_2d = st.checkbox("Show Mesh Edges", value=True, key='show_mesh_2d')
                    mesh_color = st.color_picker("Edge Color", "#000000", key='mesh_color')
                    mesh_alpha = st.slider("Edge Alpha", 0.0, 0.8, 0.3, key='mesh_alpha')
                    mesh_lw = st.slider("Edge Linewidth", 0.1, 2.0, 0.5, key='mesh_lw')
                    apply_hm = st.form_submit_button("Apply")
                
                fig_2d = create_2d_heatmap_with_mesh(
                    T_mid, extent_xy, advanced_styling,
                    show_mesh=show_mesh_2d,
                    mesh_color=mesh_color,
                    mesh_alpha=mesh_alpha,
                    mesh_linewidth=mesh_lw
                )
                st.pyplot(fig_2d)

        # 2D Time Evolution Slider (still uses Streamlit slider because it's a separate figure)
        if len(sim_data['history']) > 1:
            st.subheader("⏳ 2D Time Evolution Slider")
            zmin, zmax = resolve_cbar_range(advanced_styling, sim_data['history'][0]['T_mid'])

            frames = []
            for entry in sim_data['history']:
                T_mid = entry['T_mid']
                frames.append(go.Frame(
                    data=[go.Heatmap(z=T_mid, colorscale='Viridis',
                                     zmin=zmin, zmax=zmax)],
                    name=f"t={entry['time']:.1f}s"
                ))

            fig_slider = go.Figure(
                data=[go.Heatmap(z=sim_data['history'][0]['T_mid'],
                                 colorscale='Viridis',
                                 zmin=zmin, zmax=zmax)],
                frames=frames
            )
            fig_slider.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
                    ]
                }],
                sliders=[{
                    'currentvalue': {'prefix': 'Time: ', 'suffix': ' s'},
                    'steps': [
                        {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                         'label': f"{sim_data['history'][i]['time']:.1f}", 'method': 'animate'}
                        for i, f in enumerate(frames)
                    ]
                }],
                width=800, height=600
            )
            st.plotly_chart(fig_slider, use_container_width=True, key='2d_heatmap_slider')

else:
    st.header("🔬 Multi‑Simulation Comparison")
    sims = SimulationDB.get_simulation_list()
    if not sims:
        st.warning("No simulations available. Run some first!")
    else:
        selected_names = st.sidebar.multiselect(
            "Select simulations to compare",
            [s['name'] for s in sims],
            default=[s['name'] for s in sims][:min(3, len(sims))]
        )
        selected_ids = [s['id'] for s in sims if s['name'] in selected_names]
        if selected_ids:
            all_sims = SimulationDB.get_all_simulations()
            selected_sims = [all_sims[sid] for sid in selected_ids]

            comparison_type = st.sidebar.selectbox(
                "Comparison Type",
                ["Side-by-Side Heatmaps", "Overlay Line Profiles",
                 "Statistical Summary", "Evolution Timeline",
                 "T vs α Correlation", "Parameter Correlation"],
                index=0
            )
            frame_selection = st.sidebar.radio(
                "Frame Selection",
                ["Final Frame", "Same Evolution Time", "Specific Index"],
                index=0
            )
            if frame_selection == "Specific Index":
                frame_idx = st.sidebar.slider("Frame Index", 0, 100, 0)
            else:
                frame_idx = None

            profile_direction = 'z'
            if comparison_type == "Overlay Line Profiles":
                profile_direction = st.sidebar.selectbox(
                    "Profile Direction",
                    ['x', 'y', 'z', 'diag_xy', 'diag_xz', 'diag_yz'],
                    index=2
                )
            x_param = 'h_conv'
            if comparison_type == "Parameter Correlation":
                x_param = st.sidebar.selectbox(
                    "X‑axis parameter",
                    ['h_conv', 'trigger_temp', 'Lx', 'Ly', 'Lz', 'rho', 'Cp', 'kx', 'ky', 'kz'],
                    index=0
                )

            if st.sidebar.button("🔬 Run Comparison", type="primary"):
                frames = []
                for sim in selected_sims:
                    hist_len = len(sim['history'])
                    if frame_selection == "Final Frame":
                        frames.append(hist_len - 1)
                    elif frame_selection == "Same Evolution Time":
                        frames.append(int(hist_len * 0.8))
                    else:
                        frames.append(min(frame_idx, hist_len - 1))

                config = {
                    'type': comparison_type,
                    'profile_direction': profile_direction,
                    'x_param': x_param,
                }
                if comparison_type == "Side-by-Side Heatmaps":
                    fig = create_publication_heatmaps(selected_sims, frames, config, advanced_styling)
                elif comparison_type == "Overlay Line Profiles":
                    fig = create_enhanced_line_profiles(selected_sims, frames, config, advanced_styling)
                elif comparison_type == "Statistical Summary":
                    fig = create_publication_statistics(selected_sims, frames, config, advanced_styling)
                elif comparison_type == "Evolution Timeline":
                    fig = create_evolution_timeline_plot(selected_sims, config, advanced_styling)
                elif comparison_type == "T vs α Correlation":
                    fig = create_publication_correlation(selected_sims, frames, config, advanced_styling)
                elif comparison_type == "Parameter Correlation":
                    fig = create_cross_correlation_plot(selected_sims, config, advanced_styling)
                else:
                    fig = None

                if fig is not None:
                    st.pyplot(fig)

                with st.expander("🔄 Real‑time Post‑Processing", expanded=False):
                    st.subheader("Live Figure Customization")
                    col1, col2 = st.columns(2)
                    with col1:
                        update_fonts = st.checkbox("Update Font Sizes", True)
                        update_lines = st.checkbox("Update Line Styles", True)
                    with col2:
                        update_colors = st.checkbox("Update Colors", True)
                        update_grid = st.checkbox("Update Grid", True)
                    if st.button("🔄 Refresh with New Styling", type="secondary"):
                        st.rerun()

                with st.expander("📊 Simulation Metadata"):
                    df_meta = pd.DataFrame([{
                        'ID': s['id'],
                        'Label': s['params'].get('label', ''),
                        'Final Tmax (K)': s['metadata']['final_T_max'],
                        'Wall time (s)': s['metadata']['wall_time'],
                        'Steps': s['metadata']['total_steps'],
                        'Snapshots': s['metadata'].get('efficiency', {}).get('n_snapshots', 0)
                    } for s in selected_sims])
                    st.dataframe(df_meta)

        else:
            st.info("Select simulations from the sidebar.")

# -----------------------------------------------------------------------------
# 13. Export (unchanged)
# -----------------------------------------------------------------------------
import pickle

def generate_vts_string(T, alphas, extents, time_val):
    Nx, Ny, Nz = T.shape
    dx = (extents['x'][1]-extents['x'][0])/(Nx-1)
    dy = (extents['y'][1]-extents['y'][0])/(Ny-1)
    dz = (extents['z'][1]-extents['z'][0])/(Nz-1)

    T_flat = T.flatten(order='F')
    alpha_flat = alphas[0].flatten(order='F')

    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">',
             f'  <StructuredGrid WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">',
             f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">',
             '      <PointData Scalars="Temperature">',
             '        <DataArray type="Float64" Name="Temperature" format="ascii">',
             ' '.join(map(str, T_flat)),
             '        </DataArray>',
             '        <DataArray type="Float64" Name="SEI_Alpha" format="ascii">',
             ' '.join(map(str, alpha_flat)),
             '        </DataArray>',
             '      </PointData>',
             '      <CellData></CellData>',
             '      <Points>',
             '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">']

    points = []
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                points.append(f"{i*dx} {j*dy} {k*dz}")
    lines.append(' '.join(points))
    lines.extend(['        </DataArray>',
                  '      </Points>',
                  '    </Piece>',
                  '  </StructuredGrid>',
                  '</VTKFile>'])
    return '\n'.join(lines)

st.sidebar.header("💾 Export Options")
export_format = st.sidebar.selectbox(
    "Export Format",
    ["Complete Package (ZIP)", "VTK for ParaView (.vts/.pvd)", "Raw Numpy Arrays (.npy/.pkl)", "Raw Data CSV"]
)
include_styling = st.sidebar.checkbox("Include Styling Parameters", True)

if st.sidebar.button("📦 Generate Export", type="primary"):
    all_sims = SimulationDB.get_all_simulations()
    if not all_sims:
        st.sidebar.warning("No simulations to export.")
    else:
        if export_format == "VTK for ParaView (.vts/.pvd)":
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                pvd_lines = ['<?xml version="1.0"?>',
                             '<VTKFile type="Collection" version="0.1">',
                             '  <Collection>']
                for sim_id, sim_data in all_sims.items():
                    folder = f"sim_{sim_id}"
                    T_final, alpha_final = sim_data['final_3D']
                    ext = sim_data['metadata']['extents']
                    t = sim_data['metadata']['final_time']

                    vts_str = generate_vts_string(T_final, alpha_final, ext, t)
                    vts_filename = f"{folder}/final_state.vts"
                    zf.writestr(vts_filename, vts_str)
                    pvd_lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{vts_filename}"/>')

                pvd_lines.extend(['  </Collection>', '</VTKFile>'])
                zf.writestr("paraview_scene.pvd", '\n'.join(pvd_lines))
            buffer.seek(0)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.sidebar.download_button("📥 Download VTK ZIP", buffer.getvalue(), f"vtk_export_{ts}.zip", "application/zip")
            st.sidebar.success("VTK Export ready! Unzip and open the .pvd file in ParaView.")

        elif export_format == "Raw Numpy Arrays (.npy/.pkl)":
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for sim_id, sim_data in all_sims.items():
                    folder = f"sim_{sim_id}"
                    T_final, alpha_final = sim_data['final_3D']

                    np_bytes = BytesIO()
                    np.save(np_bytes, T_final)
                    zf.writestr(f"{folder}/T_final.npy", np_bytes.getvalue())

                    np_bytes = BytesIO()
                    np.save(np_bytes, alpha_final)
                    zf.writestr(f"{folder}/alpha_final.npy", np_bytes.getvalue())

                    pkl_bytes = BytesIO()
                    pickle.dump(sim_data, pkl_bytes)
                    zf.writestr(f"{folder}/full_data.pkl", pkl_bytes.getvalue())
            buffer.seek(0)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.sidebar.download_button("📥 Download Numpy ZIP", buffer.getvalue(), f"numpy_export_{ts}.zip", "application/zip")
            st.sidebar.success("Numpy/PKL Export ready!")

        elif export_format == "Complete Package (ZIP)":
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for sim_id, sim_data in all_sims.items():
                    folder = f"sim_{sim_id}"
                    zf.writestr(f"{folder}/params.json", json.dumps(sim_data['params'], indent=2, default=str))
                    zf.writestr(f"{folder}/metadata.json", json.dumps(sim_data['metadata'], indent=2))
                    if include_styling:
                        zf.writestr(f"{folder}/styling.json", json.dumps(advanced_styling, indent=2))
                    for i, entry in enumerate(sim_data['history']):
                        df = pd.DataFrame({
                            'T_mid': entry['T_mid'].flatten(),
                            'alpha_mid': entry['alpha_mid'].flatten()
                        })
                        zf.writestr(f"{folder}/frame_{i:04d}.csv", df.to_csv(index=False))
                buffer.seek(0)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.sidebar.download_button("📥 Download ZIP", buffer.getvalue(), f"thermal_simulations_{ts}.zip", "application/zip")
            st.sidebar.success("Export ready!")

        elif export_format == "Raw Data CSV":
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for sim_id, sim_data in all_sims.items():
                    folder = f"sim_{sim_id}"
                    for i, entry in enumerate(sim_data['history']):
                        df = pd.DataFrame({
                            'T_mid': entry['T_mid'].flatten(),
                            'alpha_mid': entry['alpha_mid'].flatten()
                        })
                        zf.writestr(f"{folder}/frame_{i:04d}.csv", df.to_csv(index=False))
            buffer.seek(0)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.sidebar.download_button("📥 Download CSV ZIP", buffer.getvalue(), f"csv_export_{ts}.zip", "application/zip")
            st.sidebar.success("CSV Export ready!")

# -----------------------------------------------------------------------------
# 14. Theoretical Documentation (updated)
# -----------------------------------------------------------------------------
with st.expander("🔬 Theoretical Soundness & Advanced Analysis (v3.6.0)", expanded=False):
    st.markdown("""
    **Multi‑Stage Arrhenius Kinetics**
    - **α‑lock fix:** Reaction degrees initialise at 0.0 (unreacted) with small global seeds.
    - **Conservative fuel consumption:** `dalpha` capped by remaining fuel.
    - **Heat cap:** temperature increase per step limited by `T_cap`.
    - **Enthalpy scaling:** H scaled to match commercial cell energy density.
    - **High‑Rise preset:** ΣH ≈ 2.8e9 J/m³ → adiabatic ΔT ≈ 1100 K (peak ~1400 K with low losses).

    **Drone Battery Model Integration**
    - Five representative battery models covering FPV, racing, freestyle, photography, and industrial drones.
    - Automatic calculation of geometry, density, energy, and mesh resolution.
    - ISC power density suggested from the model's C‑rating.
    - Reaction enthalpies auto‑scaled to match the selected model's volumetric energy.

    **Localized ISC Heater**
    - Simulates a real internal short circuit by applying volumetric Joule heating in a small sphere.
    - The heater turns off (fuses open) when local temperature exceeds `loc_cutoff`, allowing chemistry to take over.
    - Critical power threshold: \(q_{\text{loc}} \gtrsim 6k\Delta T / r^2\). With typical values, \(q_{\text{loc}} \approx 3\times10^9\) W/m³.

    **OOM‑Safe Simulation Loop**
    - Exits early when fuel < 2%
    - UI updates throttled to 4 Hz
    - Snapshots as float32
    - `max_steps` and `wall_limit_s` safety nets

    **Custom Colour‑Bar Range**
    - User‑defined T_min/T_max for consistent visual comparison.
    - Auto‑detect or lock global scale.

    **3D Camera Persistence**
    - `uirevision` tags + `st.form` + Plotly Frames + pre‑allocated traces.
    - Rotating, zooming, or panning any 3D chart is preserved across all interactions.
    """)

st.caption("🔥 Multi‑Simulation Thermal Runaway Platform • v3.6.0 • Definitive Camera Persistence • Drone Battery Models • Auto‑scaled physics • Custom color‑bar")
