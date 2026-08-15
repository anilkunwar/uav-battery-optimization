# =============================================================================
# Streamlit App: FPV LiPo 3D Thermal Runaway Multi‑Simulation Platform
# =============================================================================
# Fully upgraded with all advanced features:
#   - PublicationEnhancer (scale bars, custom colormaps, confidence bands)
#   - Deep statistical analysis (violin, KDE, regression diagnostics, correlation matrix)
#   - Extended styling (wspace, hspace, layout_pad, LaTeX, vector, grid z‑order, minor ticks)
#   - Dynamic UI (context‑sensitive controls)
#   - Real‑time post‑processing (styling refresh without re‑running)
#   - Granular export (per‑frame CSV, styling JSON, final 3D)
#   - Compute Efficiency Monitor (tracemalloc + psutil)
#   - NEW: Live Progress Monitor (ETA & Elapsed Time)
#   - NEW: Interactive 3D Domain Sketch (COMSOL-style geometry view)
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
# 2. Journal & Styling Templates
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
# 3. PublicationEnhancer Class (scale bars, custom colormaps, error/confidence)
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
# 4. Advanced Styling Controls (full set with wspace/hspace/layout_pad)
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
    with st.sidebar.expander("🎨 Colorbar", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            style['colorbar_fontsize'] = st.slider("Colorbar Font", 8, 20, 12)
            style['colorbar_width'] = st.slider("Colorbar Width", 0.2, 1.0, 0.6, 0.05)
            style['colorbar_extend'] = st.selectbox("Colorbar Extend", ['neither','both','min','max'], index=0)
        with col2:
            style['colorbar_shrink'] = st.slider("Colorbar Shrink", 0.5, 1.0, 0.8, 0.05)
            style['colorbar_pad'] = st.slider("Colorbar Pad", 0.0, 0.2, 0.05, 0.01)
            style['cmap_normalization'] = st.selectbox("Norm", ['linear','log','power'], index=0)
            if style['cmap_normalization'] == 'power':
                style['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
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
# 5. FigureStyler & EnhancedFigureStyler with new controls
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
# 6. Simulation Database (In‑memory)
# -----------------------------------------------------------------------------
class SimulationDB:
    @staticmethod
    def generate_id(sim_params):
        safe = {k: v for k, v in sim_params.items()
                if k not in ['reaction_params']}
        param_str = json.dumps(safe, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    @staticmethod
    def save_simulation(sim_params, history, metadata, final_3D):
        if 'thermal_simulations' not in st.session_state:
            st.session_state.thermal_simulations = {}
        sim_id = SimulationDB.generate_id(sim_params)
        st.session_state.thermal_simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'history': history,      
            'metadata': metadata,    
            'final_3D': final_3D,    
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
            name = f"{params.get('label', 'LiPo')} h={params['h_conv']:.1f} trig={params['trigger_temp']:.0f}K"
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
# 7. Thermal Line Profiler (3D) with anisotropic distance
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

        else:  # diagonals
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
            else:  # custom angle in XY plane
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
# 8. Numba Kernel – with corrected boundary conditions
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def step_3d(T, alphas, dt,
            rho, Cp, kx, ky, kz, dx, dy, dz,
            q_normal, reaction_params, T_amb, h_conv, eps, sigma, R):
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
                        f_alpha = alpha
                    elif r == 1:
                        f_alpha = alpha * (1.0 - alpha)
                    else:
                        f_alpha = 1.0 - alpha
                    rate = A * np.exp(-Ea / (R * max(T_ijk, 1.0)))
                    q_abuse += H * rate * f_alpha
                    dalpha = rate * f_alpha * dt
                    alphas_new[r,i,j,k] = max(alpha - dalpha, 0.0)
                q_total = q_normal + q_abuse
                T_new[i,j,k] = T_ijk + dt/(rho*Cp) * (
                    kx*d2Tdx2 + ky*d2Tdy2 + kz*d2Tdz2 + q_total
                )
    # Boundary conditions – corrected heat loss
    for j in prange(Ny):
        for k in prange(Nz):
            T_surf = T[0,j,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[0,j,k] = T_new[1,j,k] - (dx/kx) * (q_conv + q_rad)
            T_surf = T[Nx-1,j,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[Nx-1,j,k] = T_new[Nx-2,j,k] - (dx/kx) * (q_conv + q_rad)
    for i in prange(Nx):
        for k in prange(Nz):
            T_surf = T[i,0,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i,0,k] = T_new[i,1,k] - (dy/ky) * (q_conv + q_rad)
            T_surf = T[i,Ny-1,k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i,Ny-1,k] = T_new[i,Ny-2,k] - (dy/ky) * (q_conv + q_rad)
    for i in prange(Nx):
        for j in prange(Ny):
            T_surf = T[i,j,0]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i,j,0] = T_new[i,j,1] - (dz/kz) * (q_conv + q_rad)
            T_surf = T[i,j,Nz-1]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i,j,Nz-1] = T_new[i,j,Nz-2] - (dz/kz) * (q_conv + q_rad)
    return T_new, alphas_new

# -----------------------------------------------------------------------------
# 8.5 Domain Sketch Functions (2D & 3D)
# -----------------------------------------------------------------------------
def plot_initial_domain_sketch(params):
    """Generate a 2D X-Z cross-section sketch of the initial thermal domain."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    
    Lx, Lz = params['Lx'], params['Lz']
    Nx = params['Nx']
    
    # Cell rectangle (LiPo Core)
    cell_rect = plt.Rectangle((0, 0), Lx, Lz, linewidth=2.5, edgecolor='#2c3e50', facecolor='#3498db', alpha=0.25)
    ax.add_patch(cell_rect)
    
    # Air domain (ambient)
    margin = max(Lx, Lz) * 0.25
    air_rect = plt.Rectangle((-margin, -margin), Lx + 2*margin, Lz + 2*margin, 
                             linewidth=1.5, edgecolor='#7f8c8d', facecolor='none', linestyle='--')
    ax.add_patch(air_rect)
    
    # Hotspot trigger
    cx, cz = Lx/2, Lz/2
    dx = Lx / (Nx - 1)
    r_phys = params['trigger_radius'] * dx
    hotspot = plt.Circle((cx, cz), r_phys, color='#e74c3c', alpha=0.8, label=f'Hotspot ({params["trigger_radius"]} cells)')
    ax.add_patch(hotspot)
    
    # Heat flux arrows
    arrow_style = dict(arrowstyle='->', color='#e67e22', lw=2, mutation_scale=20)
    
    # Top convection/radiation
    ax.annotate('', xy=(Lx/2, Lz + margin*0.6), xytext=(Lx/2, Lz), arrowprops=arrow_style)
    ax.text(Lx/2 + 0.002, Lz + margin*0.6, 'Convection ($h$) + Radiation ($\epsilon$)', fontsize=11, color='#d35400', ha='left', fontweight='bold')
    
    # Bottom
    ax.annotate('', xy=(Lx/2, -margin*0.6), xytext=(Lx/2, 0), arrowprops=arrow_style)
    ax.text(Lx/2 + 0.002, -margin*0.6, 'Convection ($h$) + Radiation ($\epsilon$)', fontsize=11, color='#d35400', ha='left', fontweight='bold')
    
    # Left & Right
    ax.annotate('', xy=(-margin*0.6, Lz/2), xytext=(0, Lz/2), arrowprops=arrow_style)
    ax.annotate('', xy=(Lx + margin*0.6, Lz/2), xytext=(Lx, Lz/2), arrowprops=arrow_style)
    
    # Text labels
    ax.text(Lx/2, Lz/2, 'LiPo Cell Core\n(Anisotropic $k$)', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#2c3e50')
    ax.text(Lx/2, -margin*0.9, f'Ambient Air Domain ($T_\\infty = {params["T_amb"]}$ K)', 
            ha='center', va='center', fontsize=13, color='#7f8c8d', fontstyle='italic')
    
    # Anisotropic conductivity indicators
    # X/Y (High)
    ax.annotate('', xy=(Lx*0.85, Lz/2), xytext=(Lx*0.65, Lz/2), 
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.5))
    ax.text(Lx*0.75, Lz/2 + 0.003, '$k_x, k_y$ (High)', ha='center', fontsize=11, color='#2980b9', fontweight='bold')
    
    # Z (Low)
    ax.annotate('', xy=(Lx/2, Lz*0.85), xytext=(Lx/2, Lz*0.65), 
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax.text(Lx/2 + 0.003, Lz*0.75, '$k_z$ (Low)', ha='left', fontsize=11, color='#27ae60', fontweight='bold')
    
    # Axes setup
    ax.set_xlim(-margin*1.2, Lx + margin*1.2)
    ax.set_ylim(-margin*1.2, Lz + margin*1.2)
    ax.set_xlabel('x (m) - Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('z (m) - Thickness', fontsize=12, fontweight='bold')
    ax.set_title('2D Cross-Section (X-Z Plane) of Initial Thermal Domain', fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    
    return fig

def _add_physical_arrow(fig, base, axis, sign, length, color,
                        head_frac=0.35, head_rad_frac=0.15):
    """Axis-aligned arrow with TRUE data-unit size (no go.Cone autoscaling)."""
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

    # Parametric cone head (base circle at shaft_end, apex at b + vec*length)
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
    """Interactive 3D geometry sketch with physically-proportioned arrows."""
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    Nx = params['Nx']
    r_phys = params['trigger_radius'] * (Lx / (Nx - 1))

    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 30)
    x_s = Lx/2 + r_phys * np.outer(np.cos(u), np.sin(v))
    y_s = Ly/2 + r_phys * np.outer(np.sin(u), np.sin(v))
    z_s = Lz/2 + r_phys * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(  # wireframe cell
        x=[0, Lx, Lx, 0, 0, 0, Lx, Lx, 0, 0, Lx, Lx, Lx, Lx, 0, 0],
        y=[0, 0, Ly, Ly, 0, 0, 0, Ly, Ly, 0, 0, 0, Ly, Ly, Ly, Ly],
        z=[0, 0, 0, 0, 0, Lz, Lz, Lz, Lz, Lz, Lz, 0, 0, Lz, Lz, 0],
        mode='lines', line=dict(color='#2c3e50', width=3),
        name='LiPo Cell Boundary', hoverinfo='skip'))
    fig.add_trace(go.Mesh3d(  # transparent core
        x=[0, Lx, Lx, 0, 0, Lx, Lx, 0], y=[0, 0, Ly, Ly, 0, 0, Ly, Ly],
        z=[0, 0, 0, 0, Lz, Lz, Lz, Lz],
        i=[0,0,4,4,2,2,5,5,1,1,3,3], j=[1,2,5,6,3,0,1,4,5,4,7,6],
        k=[2,3,6,7,0,1,4,0,4,5,6,2],
        color='#3498db', opacity=0.15, flatshading=True,
        name='Cell Core', showscale=False))
    fig.add_trace(go.Surface(  # hotspot
        x=x_s, y=y_s, z=z_s,
        colorscale=[[0, '#e74c3c'], [1, '#e74c3c']],
        showscale=False, opacity=0.6, name='Hotspot Trigger'))

    # Physically-sized boundary-condition arrows (pointing OUTWARD = heat loss)
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

    m = arrow_len * 1.3  # margin so arrows fit inside the axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x (m)', range=[-m, Lx + m], backgroundcolor='white', gridcolor='#eeeeee'),
            yaxis=dict(title='y (m)', range=[-m, Ly + m], backgroundcolor='white', gridcolor='#eeeeee'),
            zaxis=dict(title='z (m)', range=[-m, Lz + m], backgroundcolor='white', gridcolor='#eeeeee'),
            aspectmode='data'),
        title=dict(text='🔥 3D LiPo Cell Geometry & Boundary Conditions', x=0.5),
        height=700, margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
    return fig
# -----------------------------------------------------------------------------
# 9. Simulation Runner – with progress bar and efficiency tracking (NEW)
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
    safe_T_limit = 1500.0

    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1); dz = Lz / (Nz - 1)
    extents = {'x': (0, Lx), 'y': (0, Ly), 'z': (0, Lz)}

    T = np.ones((Nx, Ny, Nz), dtype=np.float64) * T_amb
    alphas = np.ones((4, Nx, Ny, Nz), dtype=np.float64)

    # Hotspot trigger
    cx, cy, cz = Nx//2, Ny//2, Nz//2
    r = trigger_radius
    for i in range(max(0, cx-r), min(Nx, cx+r+1)):
        for j in range(max(0, cy-r), min(Ny, cy+r+1)):
            for k in range(max(0, cz-r), min(Nz, cz+r+1)):
                if (i-cx)**2 + (j-cy)**2 + (k-cz)**2 <= r**2:
                    T[i,j,k] = trigger_temp

    t = 0.0; dt = dt_init; step = 0
    times = []; T_max_history = []
    T_mid_history = []; alpha_mid_history = []
    sample_next = 0.0
    mid_z = Nz // 2

    while t < t_max:
        T, alphas = step_3d(T, alphas, dt,
                           rho, Cp, kx, ky, kz, dx, dy, dz,
                           q_normal, reaction_params, T_amb, h_conv, eps, sigma, R)
        t += dt; step += 1
        T_max = np.max(T)
        if T_max > 400.0:
            dt = max(dt_min, dt * 0.5)
        else:
            dt = min(dt_max, dt * 1.02)
        if t >= sample_next:
            times.append(t)
            T_max_history.append(T_max)
            T_mid_history.append(T[:, :, mid_z].copy())
            alpha_mid_history.append(alphas[0, :, :, mid_z].copy())
            sample_next += sample_interval
        if T_max > safe_T_limit or dt < dt_min * 0.5:
            break
        if progress_callback is not None:
            progress_callback(min(t / t_max, 1.0))

    history = []
    for idx in range(len(times)):
        history.append({
            'time': times[idx],
            'T_max': T_max_history[idx],
            'T_mid': T_mid_history[idx],
            'alpha_mid': alpha_mid_history[idx]
        })

    # Compute efficiency metrics
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

    # Calculate theoretical array memory
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
        'total_steps': step
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
        'efficiency': efficiency_stats
    }
    final_3D = (T.copy(), alphas.copy())
    return history, metadata, final_3D

# -----------------------------------------------------------------------------
# 10. Enhanced Plotting Functions (publication‑ready)
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

    for idx, (sim, frame_idx) in enumerate(zip(simulations, frames)):
        row = idx // cols; col = idx % cols
        ax = axes[row, col]
        T_mid = sim['history'][frame_idx]['T_mid']
        ext = sim['metadata']['extents']
        extent_xy = [ext['x'][0], ext['x'][1], ext['y'][0], ext['y'][1]]
        if style_params.get('apply_smoothing', True):
            T_mid = gaussian_filter(T_mid, sigma=1)
        im = ax.imshow(T_mid, extent=extent_xy,
                       cmap=cmap, origin='lower', aspect='equal')
        PublicationEnhancer.add_scale_bar(ax, 0.01, location='lower right', color='white', label='m')
        ax.set_title(sim['params'].get('label', ''))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, label='Temperature (K)',
                     shrink=style_params.get('colorbar_shrink', 0.8),
                     pad=style_params.get('colorbar_pad', 0.05),
                     extend=style_params.get('colorbar_extend', 'neither'))
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
        alpha_final = sim0['final_3D'][1][0]  # SEI
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
# 11. Main UI
# -----------------------------------------------------------------------------
advanced_styling = get_styling_controls()

operation_mode = st.sidebar.radio(
    "Operation Mode",
    ["Run New Simulation", "Compare Saved Simulations"],
    index=0
)

if operation_mode == "Run New Simulation":
    st.sidebar.header("🎛️ New Simulation Setup")
    with st.sidebar.expander("Geometry & Mesh", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            Lx = st.number_input("Length (m)", 0.005, 0.100, 0.030, 0.001)
        with col2:
            Ly = st.number_input("Width (m)", 0.005, 0.100, 0.040, 0.001)
        with col3:
            Lz = st.number_input("Thickness (m)", 0.003, 0.050, 0.010, 0.001)
        col1, col2, col3 = st.columns(3)
        with col1:
            Nx = st.number_input("Nx", 10, 80, 30, 5)
        with col2:
            Ny = st.number_input("Ny", 10, 100, 40, 5)
        with col3:
            Nz = st.number_input("Nz", 5, 40, 20, 5)
    with st.sidebar.expander("Material & Boundary"):
        rho = st.number_input("Density (kg/m³)", 1000.0, 3000.0, 2330.0, 10.0)
        Cp = st.number_input("Cp (J/kg·K)", 500.0, 2000.0, 1100.0, 50.0)
        kx = st.number_input("k_x (W/m·K)", 5.0, 60.0, 25.0, 1.0)
        ky = st.number_input("k_y (W/m·K)", 5.0, 60.0, 25.0, 1.0)
        kz = st.number_input("k_z (W/m·K)", 0.5, 5.0, 1.5, 0.1)
        T_amb = st.number_input("Ambient T (K)", 250, 350, 300, 1)
        h_conv = st.number_input("h_conv (W/m²·K)", 0.0, 50.0, 15.0, 1.0)
        eps = st.number_input("Emissivity", 0.05, 0.95, 0.20, 0.05)
    with st.sidebar.expander("Heat & Trigger"):
        q_normal = st.number_input("Normal Heat (W/m³)", 0.0, 5e5, 5e4, 1e4, format="%.0f")
        trigger_temp = st.number_input("Hotspot T (K)", 350, 600, 450, 5)
        trigger_radius = st.slider("Hotspot radius (cells)", 1, 10, 3)
    with st.sidebar.expander("Time Stepping"):
        t_max = st.number_input("Duration (s)", 10, 600, 200, 10)
        dt_init = st.number_input("dt_init (s)", 0.001, 0.1, 0.01, 0.005, format="%.3f")
        dt_min = st.number_input("dt_min (s)", 1e-7, 1e-4, 1e-6, step=1e-7, format="%.1e")
        dt_max = st.number_input("dt_max (s)", 0.001, 0.1, 0.01, 0.005, format="%.3f")
        sample_interval = st.number_input("Sample interval (s)", 0.1, 10.0, 0.5, 0.1)
    label = st.sidebar.text_input("Run Label (optional)", value=f"h={h_conv:.1f} trig={trigger_temp:.0f}K")

    # --- 3D DOMAIN SKETCH (UPGRADED) ---
    st.subheader("📐 Initial Domain Sketch (3D Interactive)")
    sketch_params = {
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'T_amb': T_amb, 'trigger_radius': trigger_radius
    }
    fig_3d = plot_3d_domain_sketch(sketch_params)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # --- EFFICIENCY MONITOR ---
    if 'last_efficiency' in st.session_state:
        st.subheader("⚡ Compute Efficiency Monitor (Last Run)")
        eff = st.session_state['last_efficiency']
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Wall Time", f"{eff['wall_time_s']:.2f} s")
        col2.metric("Peak RAM (MB)", f"{eff['peak_memory_mb']:.1f} MB")
        col3.metric("Process RAM (MB)", f"{eff['process_memory_mb']:.1f} MB")
        col4.metric("Grid Arrays (MB)", f"{eff['array_memory_mb']:.1f} MB")
        col5.metric("Total Cells", f"{eff['mesh_cells']:,}")
        if eff['cpu_avg_percent'] is not None:
            col6, col7 = st.columns(2)
            col6.metric("Avg CPU (%)", f"{eff['cpu_avg_percent']:.1f} %")
            col7.metric("OS RAM Delta (MB)", f"{eff['os_memory_delta_mb']:.1f} MB")
        with st.expander("📊 Detailed Efficiency Metrics & JSON"):
            st.json(eff)

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
            'reaction_params': np.array([
                [1.3508e5, 1.667e15, 2.57e5 * rho],
                [1.5006e5, 2.500e13, 1.714e6 * rho],
                [1.3960e5, 6.667e13, 3.140e5 * rho],
                [2.0000e5, 5.700e15, 1.550e5 * rho]
            ], dtype=np.float64),
            'dt_init': dt_init,
            'dt_min': dt_min,
            'dt_max': dt_max,
            't_max': t_max,
            'sample_interval': sample_interval,
            'trigger_temp': trigger_temp,
            'trigger_radius': trigger_radius,
            'label': label
        }
        
        # --- LIVE STATUS PLACEHOLDERS ---
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)
        live_metrics = st.empty()
        start_time = time.time()
        
        def update_progress(fraction):
            progress_bar.progress(min(fraction, 1.0), text=f"Running... {fraction*100:.1f}%")
            elapsed = time.time() - start_time
            if fraction > 0:
                eta = elapsed / fraction - elapsed
                live_metrics.info(f"⏱️ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        with st.spinner("Running thermal simulation..."):
            history, metadata, final_3D = run_simulation(params, progress_callback=update_progress)
            sim_id = SimulationDB.save_simulation(params, history, metadata, final_3D)
            st.session_state['last_efficiency'] = metadata['efficiency']
        
        progress_bar.empty()
        live_metrics.success(f"✅ Done in {metadata['efficiency']['wall_time_s']:.2f}s")
        time.sleep(0.5)
        st.rerun()

    st.header("📋 Saved Simulations")
    sims = SimulationDB.get_simulation_list()
    if sims:
        df = pd.DataFrame([{'ID': s['id'], 'Name': s['name']} for s in sims])
        st.dataframe(df, use_container_width=True)
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

    if sims:
        latest_id = sims[-1]['id']
        sim_data = SimulationDB.get_all_simulations()[latest_id]
        T_final = sim_data['final_3D'][0]
        ext = sim_data['metadata']['extents']
        mid_z = sim_data['metadata']['mesh_shape'][2] // 2
        st.subheader("Interactive 3D Temperature Surface (mid‑plane)")
        X = np.linspace(ext['x'][0], ext['x'][1], T_final.shape[0])
        Y = np.linspace(ext['y'][0], ext['y'][1], T_final.shape[1])
        Xg, Yg = np.meshgrid(X, Y, indexing='ij')
        T_slice = T_final[:, :, mid_z]
        fig3d = go.Figure(data=[go.Surface(z=T_slice, x=Xg, y=Yg, colorscale='Viridis')])
        fig3d.update_layout(scene=dict(xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='T (K)'),
                            width=800, height=600)
        st.plotly_chart(fig3d, use_container_width=True)

        if len(sim_data['history']) > 1:
            st.subheader("Time Evolution Slider")
            frames = []
            for entry in sim_data['history']:
                T_mid = entry['T_mid']
                frames.append(go.Frame(data=[go.Heatmap(z=T_mid, colorscale='Viridis')],
                                       name=f"t={entry['time']:.1f}s"))
            fig_slider = go.Figure(
                data=[go.Heatmap(z=sim_data['history'][0]['T_mid'], colorscale='Viridis')],
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
            st.plotly_chart(fig_slider, use_container_width=True)

else:  # Compare Saved Simulations
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
                        'Steps': s['metadata']['total_steps']
                    } for s in selected_sims])
                    st.dataframe(df_meta)

        else:
            st.info("Select simulations from the sidebar.")

# -----------------------------------------------------------------------------
# 12. Export (with styling parameters and per‑frame CSVs)
# -----------------------------------------------------------------------------
st.sidebar.header("💾 Export Options")
export_format = st.sidebar.selectbox(
    "Export Format",
    ["Complete Package (JSON + CSV + NPZ)", "JSON Parameters Only", "Raw Data CSV"]
)
include_styling = st.sidebar.checkbox("Include Styling Parameters", True)
if st.sidebar.button("📦 Generate Export", type="primary"):
    all_sims = SimulationDB.get_all_simulations()
    if not all_sims:
        st.sidebar.warning("No simulations to export.")
    else:
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
                T_final, alpha_final = sim_data['final_3D']
                np_bytes = BytesIO()
                np.savez_compressed(np_bytes, T=T_final, alpha=alpha_final)
                zf.writestr(f"{folder}/final_3D.npz", np_bytes.getvalue())
                summary = f"Simulation {sim_id}\n"
                summary += f"Label: {sim_data['params'].get('label', '')}\n"
                summary += f"Frames: {len(sim_data['history'])}\n"
                summary += f"Final Tmax: {sim_data['metadata']['final_T_max']:.2f} K\n"
                zf.writestr(f"{folder}/summary.txt", summary)
        buffer.seek(0)
        st.sidebar.download_button(
            "📥 Download ZIP",
            buffer.getvalue(),
            f"thermal_simulations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip"
        )
        st.sidebar.success("Export ready!")

# -----------------------------------------------------------------------------
# 13. Theoretical Documentation
# -----------------------------------------------------------------------------
with st.expander("🔬 Theoretical Soundness & Advanced Analysis", expanded=False):
    st.markdown("""
    **Multi‑Stage Arrhenius Kinetics**
    - **SEI decomposition** (α): first‑order
    - **Anode reaction**: autocatalytic α(1‑α)
    - **Cathode & Electrolyte**: (1‑α) decay
    Each stage has its own activation energy, pre‑exponential, and enthalpy.

    **Anisotropic Heat Conduction**
    - In‑plane conductivity (k_x, k_y) ≈ 25 W/m·K
    - Through‑thickness (k_z) ≈ 1.5 W/m·K → thermal bottleneck in Z direction

    **Boundary Conditions**
    - Convective + radiative losses from all faces
    - Variable h_conv simulates natural convection (h≈15) or forced/liquid cooling (h≫15)

    **Multi‑Simulation Value**
    - Compare different cooling strategies
    - Study hotspot location/severity
    - Quantify thermal runaway onset time

    **New Line Profiling**
    - Extract 1D temperature gradients along any axis
    - Reveal anisotropic heat propagation
    - Correlate with α conversion

    **Parameter Correlation**
    - Scatter plots of any input parameter vs final Tmax
    - Identify key drivers of thermal runaway
    """)

st.caption("🔥 Multi‑Simulation Thermal Runaway Platform • 2026 • Fully upgraded")
