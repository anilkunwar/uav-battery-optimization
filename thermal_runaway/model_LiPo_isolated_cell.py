# =============================================================================
# Streamlit App: FPV LiPo 3D Thermal Runaway Simulator
# =============================================================================
# This app runs the Numba-accelerated 3D anisotropic heat equation with
# multi-stage Arrhenius kinetics for an isolated pouch cell. It offers
# full control over geometry, materials, boundary conditions, and mesh
# resolution. Results are visualised interactively and exported.
# =============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import json
import os
import zipfile
from io import BytesIO
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Numba-accelerated time-step kernel (copied from the standalone script)
# -----------------------------------------------------------------------------
# The kernel uses global constants; we'll keep it as a pure function.
# We will re-define it here exactly as before, but we'll also pass all
# parameters as arguments (no global dependencies).

@njit(parallel=True, fastmath=True, cache=True)
def step_3d(T, alphas, dt,
            rho, Cp, kx, ky, kz, dx, dy, dz,
            q_normal, reaction_params, T_amb, h_conv, eps, sigma, R):
    """
    One explicit time step (Forward Euler) on the 3D grid.
    """
    Nx, Ny, Nz = T.shape
    T_new = T.copy()
    alphas_new = alphas.copy()
    
    # Internal nodes (parallelised)
    for i in prange(1, Nx - 1):
        for j in prange(1, Ny - 1):
            for k in prange(1, Nz - 1):
                # Laplacian
                d2Tdx2 = (T[i+1, j, k] - 2.0*T[i, j, k] + T[i-1, j, k]) / (dx*dx)
                d2Tdy2 = (T[i, j+1, k] - 2.0*T[i, j, k] + T[i, j-1, k]) / (dy*dy)
                d2Tdz2 = (T[i, j, k+1] - 2.0*T[i, j, k] + T[i, j, k-1]) / (dz*dz)
                
                T_ijk = T[i, j, k]
                q_abuse = 0.0
                for r in range(4):
                    Ea = reaction_params[r, 0]
                    A  = reaction_params[r, 1]
                    H  = reaction_params[r, 2]
                    alpha = alphas[r, i, j, k]
                    
                    if r == 0:          # SEI: α
                        f_alpha = alpha
                    elif r == 1:        # Anode: α(1‑α)
                        f_alpha = alpha * (1.0 - alpha)
                    else:               # Cathode & Electrolyte: (1‑α)
                        f_alpha = 1.0 - alpha
                    
                    rate = A * np.exp(-Ea / (R * max(T_ijk, 1.0)))
                    q_abuse += H * rate * f_alpha
                    
                    dalpha = rate * f_alpha * dt
                    alphas_new[r, i, j, k] = max(alpha - dalpha, 0.0)
                
                q_total = q_normal + q_abuse
                T_new[i, j, k] = T_ijk + dt / (rho * Cp) * (
                    kx * d2Tdx2 + ky * d2Tdy2 + kz * d2Tdz2 + q_total
                )
    
    # Boundary conditions (all faces)
    # X=0
    for j in prange(Ny):
        for k in prange(Nz):
            T_surf = T[0, j, k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[0, j, k] = T_new[1, j, k] + (dx / kx) * (q_conv + q_rad)
    # X=Lx
    for j in prange(Ny):
        for k in prange(Nz):
            T_surf = T[Nx-1, j, k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[Nx-1, j, k] = T_new[Nx-2, j, k] + (dx / kx) * (q_conv + q_rad)
    # Y=0
    for i in prange(Nx):
        for k in prange(Nz):
            T_surf = T[i, 0, k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i, 0, k] = T_new[i, 1, k] + (dy / ky) * (q_conv + q_rad)
    # Y=Ly
    for i in prange(Nx):
        for k in prange(Nz):
            T_surf = T[i, Ny-1, k]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i, Ny-1, k] = T_new[i, Ny-2, k] + (dy / ky) * (q_conv + q_rad)
    # Z=0
    for i in prange(Nx):
        for j in prange(Ny):
            T_surf = T[i, j, 0]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i, j, 0] = T_new[i, j, 1] + (dz / kz) * (q_conv + q_rad)
    # Z=Lz
    for i in prange(Nx):
        for j in prange(Ny):
            T_surf = T[i, j, Nz-1]
            q_conv = h_conv * (T_surf - T_amb)
            q_rad = eps * sigma * (T_surf**4 - T_amb**4)
            T_new[i, j, Nz-1] = T_new[i, j, Nz-2] + (dz / kz) * (q_conv + q_rad)
    
    return T_new, alphas_new


# -----------------------------------------------------------------------------
# 2. Simulation runner function (called from Streamlit with caching)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(params):
    """
    Run the thermal runaway simulation with the given parameters.
    Returns a dictionary with all results and metadata.
    """
    # Unpack parameters
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    rho = params['rho']
    Cp = params['Cp']
    kx, ky, kz = params['kx'], params['ky'], params['kz']
    T_amb = params['T_amb']
    h_conv = params['h_conv']
    eps = params['eps']
    q_normal = params['q_normal']
    reaction_params = params['reaction_params']  # (4,3) array
    dt_init = params['dt_init']
    dt_min = params['dt_min']
    dt_max = params['dt_max']
    t_max = params['t_max']
    sample_interval = params['sample_interval']
    trigger_temp = params['trigger_temp']
    trigger_radius = params['trigger_radius']  # in grid cells (approx)
    R = 8.314
    sigma = 5.67e-8
    safe_T_limit = 1500.0
    
    # Compute grid spacing
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)
    
    # Mesh coordinates
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialise fields
    T = np.ones((Nx, Ny, Nz), dtype=np.float64) * T_amb
    alphas = np.ones((4, Nx, Ny, Nz), dtype=np.float64)
    
    # Trigger: local hot spot
    cx, cy, cz = Nx//2, Ny//2, Nz//2
    r = trigger_radius
    # We'll set a spherical region (approx)
    for i in range(max(0, cx-r), min(Nx, cx+r+1)):
        for j in range(max(0, cy-r), min(Ny, cy+r+1)):
            for k in range(max(0, cz-r), min(Nz, cz+r+1)):
                if (i-cx)**2 + (j-cy)**2 + (k-cz)**2 <= r**2:
                    T[i, j, k] = trigger_temp
    
    # Time stepping
    t = 0.0
    dt = dt_init
    step = 0
    
    # Storage for sampling
    times = []
    T_max_history = []
    T_fields = []
    alphas_fields = []
    sample_next = 0.0
    
    # Progress tracking (for Streamlit, we'll use a callback but since we're
    # inside cached function, we cannot update st.progress directly; we'll
    # return intermediate results after the fact. We'll just run the loop
    # and report progress via a separate mechanism? Actually, st.cache_data
    # does not allow side effects. We'll use a workaround: we'll not cache
    # the entire simulation but rather cache the results based on params.
    # Since the simulation may take a while, we'll use st.cache_data with
    # a progress bar that we update from outside? Not straightforward.
    # Simpler: we'll run the simulation inside the main app with a spinner
    # and show a progress bar manually by splitting the simulation into chunks.
    # For now, we'll keep the full simulation inside cached function and
    # accept that the spinner will spin without detailed progress.
    
    # Start simulation
    while t < t_max:
        T, alphas = step_3d(T, alphas, dt,
                           rho, Cp, kx, ky, kz, dx, dy, dz,
                           q_normal, reaction_params, T_amb, h_conv, eps, sigma, R)
        t += dt
        step += 1
        
        T_max = np.max(T)
        if T_max > 400.0:
            dt = max(dt_min, dt * 0.5)
        else:
            dt = min(dt_max, dt * 1.02)
        
        if t >= sample_next:
            times.append(t)
            T_max_history.append(T_max)
            T_fields.append(T.copy())
            alphas_fields.append(alphas.copy())
            sample_next += sample_interval
        
        if T_max > safe_T_limit or dt < dt_min * 0.5:
            break
    
    # Convert lists to arrays
    times = np.array(times)
    T_max_history = np.array(T_max_history)
    T_fields = np.array(T_fields)
    alphas_fields = np.array(alphas_fields)
    
    # Prepare output dictionary
    results = {
        'T_final': T,
        'alphas_final': alphas,
        'times': times,
        'T_max_history': T_max_history,
        'T_fields': T_fields,
        'alphas_fields': alphas_fields,
        'X': X,
        'Y': Y,
        'Z': Z,
        'params': params,
        'metadata': {
            'simulation_date': datetime.now().isoformat(),
            'final_time': t,
            'total_steps': step,
            'final_T_max': float(np.max(T)),
            'wall_time': 0.0,  # will be filled later
            'mesh_shape': (Nx, Ny, Nz),
            'dx_dy_dz': (dx, dy, dz)
        }
    }
    return results


# -----------------------------------------------------------------------------
# 3. Streamlit App
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FPV LiPo Thermal Runaway Simulator", layout="wide")
st.title("🔥 FPV LiPo 3D Thermal Runaway Simulator")
st.markdown("""
This app simulates the thermal behaviour of a single **Lithium Polymer (LiPo) pouch cell** 
under high-current discharge and potential internal short. The model uses **3D anisotropic 
heat conduction** with multi‑stage Arrhenius kinetics (SEI, anode, cathode, electrolyte) 
and convective+radiative boundary conditions. The solver is accelerated with **Numba JIT**.
""")

# -----------------------------------------------------------------------------
# Sidebar – User Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    
    st.subheader("Geometry")
    col1, col2, col3 = st.columns(3)
    with col1:
        Lx = st.number_input("Length (m)", value=0.030, min_value=0.005, max_value=0.100, step=0.001, format="%.3f")
    with col2:
        Ly = st.number_input("Width (m)", value=0.040, min_value=0.005, max_value=0.100, step=0.001, format="%.3f")
    with col3:
        Lz = st.number_input("Thickness (m)", value=0.010, min_value=0.003, max_value=0.050, step=0.001, format="%.3f")
    
    st.subheader("Mesh Resolution")
    col1, col2, col3 = st.columns(3)
    with col1:
        Nx = st.number_input("Nx", value=30, min_value=10, max_value=80, step=5)
    with col2:
        Ny = st.number_input("Ny", value=40, min_value=10, max_value=100, step=5)
    with col3:
        Nz = st.number_input("Nz", value=20, min_value=5, max_value=40, step=5)
    st.caption("Higher resolution = more accurate but slower. For quick testing, use 20x30x15.")
    
    st.subheader("Material Properties")
    rho = st.number_input("Density (kg/m³)", value=2330.0, min_value=1000.0, max_value=3000.0, step=10.0)
    Cp = st.number_input("Specific Heat (J/kg·K)", value=1100.0, min_value=500.0, max_value=2000.0, step=50.0)
    kx = st.number_input("k_x (in‑plane, W/m·K)", value=25.0, min_value=5.0, max_value=60.0, step=1.0)
    ky = st.number_input("k_y (in‑plane, W/m·K)", value=25.0, min_value=5.0, max_value=60.0, step=1.0)
    kz = st.number_input("k_z (through‑thickness, W/m·K)", value=1.5, min_value=0.5, max_value=5.0, step=0.1)
    
    st.subheader("Boundary Conditions")
    T_amb = st.number_input("Ambient Temperature (K)", value=300.0, min_value=250.0, max_value=350.0, step=1.0)
    h_conv = st.number_input("Convective Coefficient (W/m²·K)", value=15.0, min_value=0.0, max_value=50.0, step=1.0)
    eps = st.number_input("Emissivity", value=0.20, min_value=0.05, max_value=0.95, step=0.05)
    
    st.subheader("Normal & Abuse Heat")
    q_normal = st.number_input("Normal Operation Heat (W/m³)", value=5e4, min_value=0.0, max_value=5e5, step=1e4, format="%.0f")
    # Reaction parameters are fixed (we could allow tuning but keep them as defaults)
    
    st.subheader("Trigger (Internal Short)")
    trigger_temp = st.number_input("Hotspot Temperature (K)", value=450.0, min_value=350.0, max_value=600.0, step=5.0)
    trigger_radius = st.slider("Hotspot Radius (grid cells)", min_value=1, max_value=10, value=3)
    
    st.subheader("Time Stepping")
    t_max = st.number_input("Simulation Duration (s)", value=200.0, min_value=10.0, max_value=600.0, step=10.0)
    dt_init = st.number_input("Initial dt (s)", value=0.01, min_value=0.001, max_value=0.1, step=0.005, format="%.3f")
    dt_min = st.number_input("Minimum dt (s)", value=1e-6, min_value=1e-7, max_value=1e-4, step=1e-7, format="%.1e")
    dt_max = st.number_input("Maximum dt (s)", value=0.01, min_value=0.001, max_value=0.1, step=0.005, format="%.3f")
    sample_interval = st.number_input("Sampling Interval (s)", value=0.5, min_value=0.1, max_value=10.0, step=0.1)
    
    st.subheader("Export Options")
    export_vtu = st.checkbox("Export VTU (ParaView)", value=True)
    export_npy = st.checkbox("Export NPY arrays", value=True)
    export_zip = st.checkbox("Create ZIP package", value=True)
    
    # Button to run simulation
    run_button = st.button("🚀 Run Simulation", type="primary")

# -----------------------------------------------------------------------------
# Main area – Run simulation and display results
# -----------------------------------------------------------------------------
if run_button:
    # Prepare parameters dict
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
    }
    
    # Run simulation with spinner and progress bar
    with st.spinner("Running simulation... (may take a few minutes)"):
        start_time = time.time()
        results = run_simulation(params)
        wall_time = time.time() - start_time
        results['metadata']['wall_time'] = wall_time
    
    st.success(f"✅ Simulation completed in {wall_time:.1f} seconds.")
    
    # Unpack results
    T_final = results['T_final']
    alphas_final = results['alphas_final']
    times = results['times']
    T_max_history = results['T_max_history']
    T_fields = results['T_fields']
    alphas_fields = results['alphas_fields']
    X = results['X']; Y = results['Y']; Z = results['Z']
    metadata = results['metadata']
    
    # Display metadata
    st.subheader("📊 Simulation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Time", f"{metadata['final_time']:.1f} s")
    col2.metric("Final T_max", f"{metadata['final_T_max']:.1f} K")
    col3.metric("Total Steps", metadata['total_steps'])
    col4.metric("Wall Time", f"{metadata['wall_time']:.1f} s")
    
    # -------------------------------------------------------------------------
    # Tabs for visualisation
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🌡️ 3D Surface", "🎞️ Interactive Slider", "📤 Export"])
    
    with tab1:
        st.subheader("Maximum Temperature vs. Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, T_max_history, 'r-', linewidth=2)
        ax.axhline(400, color='orange', linestyle='--', label='SEI onset (127 °C)')
        ax.axhline(462, color='darkred', linestyle='--', label='Runaway threshold (189 °C)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Maximum Temperature (K)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        # Also show a table of last few samples
        st.subheader("Recent Data")
        if len(times) > 10:
            df = pd.DataFrame({
                'Time (s)': times[-10:],
                'T_max (K)': T_max_history[-10:]
            })
            st.dataframe(df)
        else:
            df = pd.DataFrame({'Time (s)': times, 'T_max (K)': T_max_history})
            st.dataframe(df)
    
    with tab2:
        st.subheader("3D Surface of Final Temperature (mid‑plane)")
        mid_z = Nz // 2
        T_slice = T_final[:, :, mid_z]
        X_slice = X[:, :, mid_z]
        Y_slice = Y[:, :, mid_z]
        
        # Plotly 3D surface
        import plotly.graph_objects as go
        fig3d = go.Figure(data=[
            go.Surface(
                x=X_slice, y=Y_slice, z=T_slice,
                colorscale='Viridis',
                colorbar=dict(title='Temperature (K)')
            )
        ])
        fig3d.update_layout(
            title=f"Mid‑plane (z={Lz/2*1000:.1f} mm) at t={metadata['final_time']:.1f} s",
            scene=dict(
                xaxis_title='x (m)',
                yaxis_title='y (m)',
                zaxis_title='Temperature (K)'
            ),
            width=800, height=600
        )
        st.plotly_chart(fig3d, use_container_width=True)
        
        # Also show a 2D contour
        st.subheader("2D Contour Plot (mid‑plane)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.contourf(X_slice, Y_slice, T_slice, levels=50, cmap='viridis')
        plt.colorbar(im, ax=ax2, label='Temperature (K)')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_title(f"Temperature at t={metadata['final_time']:.1f} s")
        st.pyplot(fig2)
        plt.close(fig2)
    
    with tab3:
        st.subheader("Interactive Time Slider (mid‑plane)")
        # We'll create a plotly figure with a slider using the stored T_fields
        if len(T_fields) > 1:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Prepare frames
            frames = []
            for idx, T_sample in enumerate(T_fields):
                T_slice_sample = T_sample[:, :, mid_z]
                frame = go.Frame(
                    data=[go.Heatmap(z=T_slice_sample, colorscale='Viridis')],
                    name=f't={times[idx]:.1f}s'
                )
                frames.append(frame)
            
            # Initial frame
            initial_T = T_fields[0][:, :, mid_z]
            fig_slider = go.Figure(
                data=[go.Heatmap(z=initial_T, colorscale='Viridis')],
                frames=frames
            )
            fig_slider.update_layout(
                title='Temperature evolution (mid‑plane)',
                xaxis_title='x index',
                yaxis_title='y index',
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
                         'label': f'{times[i]:.1f}', 'method': 'animate'}
                        for i, f in enumerate(frames)
                    ]
                }],
                width=800, height=600
            )
            st.plotly_chart(fig_slider, use_container_width=True)
        else:
            st.info("Not enough time samples for animation.")
    
    with tab4:
        st.subheader("📤 Export Results")
        
        # Prepare a ZIP file if requested
        if export_zip or export_vtu or export_npy:
            # We'll generate files on the fly and put them in a BytesIO zip
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. Metadata JSON
                meta_json = json.dumps(metadata, indent=2)
                zf.writestr('metadata.json', meta_json)
                
                # 2. Time series data (CSV)
                import pandas as pd
                df_ts = pd.DataFrame({'time': times, 'T_max': T_max_history})
                zf.writestr('T_max_history.csv', df_ts.to_csv(index=False))
                
                # 3. NPY arrays
                if export_npy:
                    # Save each array as .npy in memory
                    np_array_dict = {
                        'T_final.npy': T_final,
                        'alphas_final.npy': alphas_final,
                        'times.npy': times,
                        'T_max_history.npy': T_max_history,
                        'T_fields.npy': T_fields,
                        'alphas_fields.npy': alphas_fields,
                        'X.npy': X,
                        'Y.npy': Y,
                        'Z.npy': Z,
                    }
                    for fname, arr in np_array_dict.items():
                        # Convert to bytes
                        arr_bytes = BytesIO()
                        np.save(arr_bytes, arr)
                        zf.writestr(fname, arr_bytes.getvalue())
                
                # 4. VTU file
                if export_vtu:
                    try:
                        import meshio
                        # Flatten points
                        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
                        # Hexahedral cells
                        cells = []
                        for i in range(Nx - 1):
                            for j in range(Ny - 1):
                                for k in range(Nz - 1):
                                    idx = i + j*Nx + k*Nx*Ny
                                    hex_conn = [
                                        idx, idx + 1, idx + 1 + Nx, idx + Nx,
                                        idx + Nx*Ny, idx + 1 + Nx*Ny,
                                        idx + 1 + Nx + Nx*Ny, idx + Nx + Nx*Ny
                                    ]
                                    cells.append(hex_conn)
                        cells = np.array(cells)
                        point_data = {
                            'Temperature': T_final.ravel().astype(np.float64),
                            'alpha_SEI': alphas_final[0].ravel().astype(np.float64),
                            'alpha_Anode': alphas_final[1].ravel().astype(np.float64),
                            'alpha_Cathode': alphas_final[2].ravel().astype(np.float64),
                            'alpha_Electrolyte': alphas_final[3].ravel().astype(np.float64),
                        }
                        mesh = meshio.Mesh(points, [("hexahedron", cells)], point_data=point_data)
                        # Write to BytesIO
                        vtu_bytes = BytesIO()
                        meshio.write(vtu_bytes, mesh, file_format='vtu', binary=True)
                        zf.writestr('simulation.vtu', vtu_bytes.getvalue())
                        st.success("VTU file added to ZIP.")
                    except ImportError:
                        st.warning("meshio not installed; VTU export skipped.")
                
                # 5. Add plots (PNG) – we can generate them again
                # Time series plot
                fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
                ax_ts.plot(times, T_max_history, 'r-', linewidth=2)
                ax_ts.axhline(400, color='orange', linestyle='--')
                ax_ts.axhline(462, color='darkred', linestyle='--')
                ax_ts.set_xlabel('Time (s)'); ax_ts.set_ylabel('T_max (K)')
                ax_ts.grid(True)
                fig_ts.savefig('T_max_history.png', dpi=150)
                zf.writestr('plots/T_max_history.png', open('T_max_history.png', 'rb').read())
                plt.close(fig_ts)
                os.remove('T_max_history.png')
                
                # Slice plot
                fig_slice, ax_slice = plt.subplots(figsize=(8, 6))
                im = ax_slice.contourf(X_slice, Y_slice, T_slice, levels=50, cmap='viridis')
                plt.colorbar(im, ax=ax_slice, label='Temperature (K)')
                ax_slice.set_xlabel('x (m)'); ax_slice.set_ylabel('y (m)')
                fig_slice.savefig('T_slice.png', dpi=150)
                zf.writestr('plots/T_slice.png', open('T_slice.png', 'rb').read())
                plt.close(fig_slice)
                os.remove('T_slice.png')
                
                # 3D surface plot (HTML)
                if 'fig3d' in locals():
                    fig3d.write_html('T_surface_3D.html')
                    zf.writestr('plots/T_surface_3D.html', open('T_surface_3D.html', 'rb').read())
                    os.remove('T_surface_3D.html')
            
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Download ZIP Package",
                data=zip_buffer,
                file_name=f"fpv_runaway_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        else:
            st.info("Enable export options above to generate downloadable files.")
        
        # Individual downloads (optional)
        st.subheader("Individual File Downloads")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            # CSV of T_max history
            csv_data = pd.DataFrame({'time': times, 'T_max': T_max_history}).to_csv(index=False)
            st.download_button("Download T_max.csv", csv_data, file_name="T_max.csv", mime="text/csv")
        with col_dl2:
            # JSON metadata
            st.download_button("Download metadata.json", json.dumps(metadata, indent=2), file_name="metadata.json", mime="application/json")
        with col_dl3:
            # NPY of final T
            arr_bytes = BytesIO()
            np.save(arr_bytes, T_final)
            st.download_button("Download T_final.npy", arr_bytes.getvalue(), file_name="T_final.npy", mime="application/octet-stream")

else:
    st.info("👈 Adjust the parameters in the sidebar and click 'Run Simulation' to start.")

# Add a footer with note about performance
st.sidebar.markdown("---")
st.sidebar.caption("⚡ Powered by Numba JIT – simulation time depends on mesh resolution. For quick tests, use coarse mesh.")
