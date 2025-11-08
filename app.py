"""
Ultimate Streamlit Scientific Workstation
- Handles multiple CIF files (batch)
- Global element/pair controls + per-pair bond min/max (fully customizable)
- Full scientist-grade control board (advanced options, caches, presets)
- QSPR binding energy with editable coefficients (manual entry)
- SOAP + MBTR optional (cached)
- py3Dmol viewer embedded into Streamlit

Author: Adapted for user (2025)
"""

import os
import io
import json
import math
import tempfile
import zipfile
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from ase import Atoms
from dscribe.descriptors import SOAP, MBTR
import py3Dmol

# ---------------- page config ----------------
st.set_page_config(page_title="Scientific Workstation — Ultimate", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>footer {visibility: hidden;} .stApp { background-color: #f8fafc; }</style>", unsafe_allow_html=True)

# ---------------- defaults ----------------
BOND_LENGTH_FALLBACK = 3.2
DEFAULT_DBSCAN_EPS = 0.25
DEFAULT_DBSCAN_MIN_SAMPLES = 1
DEFAULT_SOAP_N = 50

DEFAULT_COEFFICIENTS = OrderedDict([
    ('b0', 0.0000),
    ('b_Pt-S', -0.1225),
    ('b_Ru-S', 0.0000),
    ('b_Pt-Ru', 0.0050),
    ('b_Ru-Ru', -0.0160),
    ('b_Ru-Ru-', -0.1020),
    ('b_Pt-Pt', 0.0048),
    ('b_Pt-Pt-', 0.0550),
])

# quick element color map (extendable)
ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050",
    "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Pt": "#D0D0E0", "Ru": "#248F8F",
}

# initialize CrystalNN
cnn = CrystalNN(weighted_cn=False)

# ---------------- util helpers ----------------
def adaptive_bond_cutoff(e1: str, e2: str, scale=1.25, fallback=BOND_LENGTH_FALLBACK) -> float:
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [r1, r2]):
            return fallback
        return scale * (r1 + r2)
    except Exception:
        return fallback


def detect_layers(z_coords: List[float], eps=DEFAULT_DBSCAN_EPS, min_samples=DEFAULT_DBSCAN_MIN_SAMPLES) -> np.ndarray:
    if len(z_coords) == 0:
        return np.array([])
    z = np.array(z_coords).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(z)
    labels = clustering.labels_
    unique = sorted(set(labels))
    mapping = {lab: idx for idx, lab in enumerate(unique)}
    return np.array([mapping[lab] for lab in labels])


def find_neighbors_validated(structure: Structure, idx: int, max_check=128) -> List[int]:
    neighbors_idx = set()
    try:
        for info in cnn.get_nn_info(structure, idx):
            j = info['site_index']
            e1, e2 = str(structure[idx].specie), str(structure[j].specie)
            d = structure.get_distance(idx, j)
            if d <= adaptive_bond_cutoff(e1, e2) * 1.05:
                neighbors_idx.add(j)
    except Exception:
        pass
    if not neighbors_idx:
        for j, site in enumerate(structure.sites):
            if j == idx: continue
            e1, e2 = str(structure[idx].specie), str(site.specie)
            d = structure.get_distance(idx, j)
            if d <= adaptive_bond_cutoff(e1, e2):
                neighbors_idx.add(j)
            if len(neighbors_idx) >= max_check:
                break
    return sorted(neighbors_idx)


def structure_to_ase(structure: Structure) -> Atoms:
    symbols = [str(s.specie) for s in structure]
    positions = np.array([s.coords for s in structure])
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def propagate_uncertainty_linear(coeff_map: Dict[str, float], x_counts: Dict[str, int]) -> float:
    sigma2 = 0.0
    for ck, b in coeff_map.items():
        if ck == 'b0': continue
        xkey = ck.replace('b_', '')
        if xkey in x_counts:
            N = max(x_counts[xkey], 0)
            sigma_x = math.sqrt(N)
            sigma2 += (b * sigma_x)**2
    return math.sqrt(sigma2)


# py3Dmol builder
def make_py3dmol_html(structure: Structure, show_bonds=True, bond_min=0.0, bond_max=3.0,
                      highlight_atom_index=None, show_poly=False, atom_size=0.45) -> str:
    coords = np.array([s.coords for s in structure.sites])
    elements = [str(s.specie) for s in structure.sites]
    view = py3Dmol.view(width=780, height=520)
    for i, (el, pos) in enumerate(zip(elements, coords)):
        color = ELEMENT_COLORS.get(el, ELEMENT_COLORS.get(el.capitalize(), '#AAAAAA'))
        view.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                        'radius': atom_size, 'color': color, 'opacity': 1.0})
    if show_bonds:
        n = len(elements)
        for i in range(n):
            for j in range(i+1, n):
                d = structure.get_distance(i, j)
                if bond_min <= d <= bond_max:
                    view.addCylinder({'start': {'x': float(coords[i,0]), 'y': float(coords[i,1]), 'z': float(coords[i,2])},
                                      'end': {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                                      'radius': atom_size*0.12, 'color': '#BBBBBB', 'opacity': 1.0})
    if highlight_atom_index is not None and 0 <= int(highlight_atom_index) < len(coords):
        pos = coords[int(highlight_atom_index)]
        view.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}, 'radius': atom_size*1.4, 'color': 'red', 'opacity': 1.0})
    view.setBackgroundColor('#ffffff')
    view.zoomTo()
    try:
        html = view._make_html()
    except Exception:
        html = view.show()
    return html


# ---------------- core analysis ----------------
@st.cache_data(show_spinner=False)
def compute_descriptors_for_structure(structure: Structure, filename: str,
                                     per_pair_ranges: Dict[str, Tuple[float,float]],
                                     normalize_by: str = 'bonds',
                                     dbscan_eps: float = DEFAULT_DBSCAN_EPS, dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
                                     compute_soap: bool = True, compute_mbtr: bool = True, soap_first_n: int = DEFAULT_SOAP_N,
                                     coeff_map: Dict[str,float] = None) -> Tuple[Dict, pd.DataFrame]:
    n_atoms = len(structure.sites)
    elements = [str(site.specie) for site in structure.sites]
    coords = np.array([site.coords for site in structure])
    z_coords = coords[:,2] if n_atoms>0 else np.array([])

    if n_atoms > 0:
        layer_indices = detect_layers(z_coords, eps=dbscan_eps, min_samples=dbscan_min_samples)
        layer_means = {l: np.mean(z_coords[layer_indices==l]) for l in np.unique(layer_indices)}
        label_map = {old: new for new, old in enumerate(sorted(layer_means, key=lambda L: layer_means[L]))}
        layer_indices = np.array([label_map[l] for l in layer_indices])
        n_layers = int(layer_indices.max()) + 1
        surface_mask = (layer_indices == layer_indices.min()) | (layer_indices == layer_indices.max())
    else:
        layer_indices = np.array([])
        n_layers = 0
        surface_mask = np.array([])

    atom_rows = []
    for i, c in enumerate(coords):
        atom_rows.append({
            'filename': filename, 'atom_index': i, 'element': elements[i],
            'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2]),
            'layer_index': int(layer_indices[i]) if n_atoms>0 else -1,
            'surface_flag': bool(surface_mask[i]) if n_atoms>0 else False
        })
    atom_df = pd.DataFrame(atom_rows)

    bond_counts_intra = defaultdict(int)
    bond_counts_inter = defaultdict(int)
    coordination = np.zeros(n_atoms)
    for i in range(n_atoms):
        neighs = find_neighbors_validated(structure, i)
        coordination[i] = len(neighs)
        for j in neighs:
            if j <= i: continue
            e1, e2 = elements[i], elements[j]
            pair_key = "-".join(sorted([e1, e2]))
            d = structure.get_distance(i, j)
            if per_pair_ranges and pair_key in per_pair_ranges:
                min_r, max_r = per_pair_ranges[pair_key]
            else:
                min_r, max_r = 0.0, adaptive_bond_cutoff(e1, e2) * 1.12
            if not (min_r <= d <= max_r):
                continue
            if n_atoms>0 and layer_indices[i] == layer_indices[j]:
                bond_counts_intra[pair_key] += 1
            else:
                bond_counts_inter[pair_key] += 1

    surface_counts = defaultdict(int)
    for i, surf in enumerate(surface_mask):
        if surf:
            surface_counts[f"{elements[i]}-S"] += 1

    x = {}
    # generic keys: for all detected pairs produce counts so the app is element-agnostic
    all_pairs = set(list(bond_counts_intra.keys()) + list(bond_counts_inter.keys()))
    for pair in sorted(all_pairs):
        x[pair] = bond_counts_intra.get(pair, 0) + bond_counts_inter.get(pair, 0)

    # Include surface keys for detected surface element types
    for k, v in surface_counts.items():
        x[k] = v

    total_bonds = sum(bond_counts_intra.values()) + sum(bond_counts_inter.values())
    desc = {
        'filename': filename,
        'n_atoms': n_atoms,
        'n_layers': n_layers,
        'avg_coordination': float(np.mean(coordination)) if n_atoms>0 else 0.0,
        'surface_ratio': float(np.sum(surface_mask)/n_atoms) if n_atoms>0 else 0.0,
        'total_bonds': int(total_bonds)
    }
    desc.update(x)

    x_norm = {k: v for k, v in x.items()}
    if normalize_by == 'atoms' and n_atoms>0:
        x_norm = {k: v / n_atoms for k, v in x.items()}
    elif normalize_by == 'bonds' and total_bonds>0:
        x_norm = {k: v / total_bonds for k, v in x.items()}

    if coeff_map is None:
        coeff_map = dict(DEFAULT_COEFFICIENTS)
    Y = coeff_map.get('b0', 0.0) + sum(coeff_map.get('b_' + k, 0.0) * x_norm.get(k, 0.0) for k in x_norm)
    sigmaY = propagate_uncertainty_linear(coeff_map, x)
    desc['Binding_Energy_eV'] = float(Y)
    desc['Binding_Energy_sigma_eV_approx'] = float(sigmaY)

    # SOAP & MBTR (may be slow) - compute summary/cached
    try:
        ase_atoms = structure_to_ase(structure)
        species = list(sorted(set(ase_atoms.get_chemical_symbols())))
    except Exception:
        ase_atoms = None
        species = []

    if compute_soap and ase_atoms is not None and species:
        try:
            soap = SOAP(species=species, rcut=5.0, nmax=6, lmax=4, sigma=0.5, periodic=True)
            soap_vec = soap.create(ase_atoms)
            soap_mean = np.mean(soap_vec, axis=0)
            for i, val in enumerate(soap_mean[:soap_first_n]):
                desc[f"SOAP_{i}"] = float(val)
        except Exception as e:
            st.warning(f"SOAP failed for {filename}: {e}")

    if compute_mbtr and ase_atoms is not None and species:
        try:
            mbtr = MBTR(
                species=species,
                k2={"geometry": {"function": "inverse_distance"}, "grid": {"min": 0, "max": 1, "n": 30}},
                k3={"geometry": {"function": "angle"}, "grid": {"min": 0, "max": 180, "n": 30}},
                periodic=True,
                normalization="l2_each"
            )
            mbtr_vec = mbtr.create(ase_atoms)
            desc['MBTR_mean'] = float(np.mean(mbtr_vec))
            desc['MBTR_std'] = float(np.std(mbtr_vec))
        except Exception as e:
            st.warning(f"MBTR failed for {filename}: {e}")

    return desc, atom_df


# ---------------- session state init ----------------
if 'pair_ranges' not in st.session_state:
    st.session_state['pair_ranges'] = {}
if 'coeff_map' not in st.session_state:
    st.session_state['coeff_map'] = dict(DEFAULT_COEFFICIENTS)
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []


# ---------------- UI layout ----------------
st.title("🔬 Scientific Workstation — Universal QSPR + SOAP + MBTR")

tabs = st.tabs(["Viewer", "Control Board", "Descriptors & Run", "Presets"])

# ---------------- Viewer ----------------
with tabs[0]:
    st.header("3D Viewer — first uploaded structure")
    colv, colc = st.columns([3,1])
    with colv:
        vp = st.empty()
    with colc:
        st.markdown("**Viewer controls**")
        v_show_bonds = st.checkbox("Show bonds", value=True, key='v_show_bonds')
        v_bmin = st.number_input("Min bond (Å)", value=0.0, step=0.01, key='v_bmin')
        v_bmax = st.number_input("Max bond (Å)", value=3.0, step=0.01, key='v_bmax')
        v_atom_idx = st.number_input("Highlight atom index (0-based)", value=0, min_value=0, step=1, key='v_atom_idx')

    uploaded_list = st.session_state.get('uploaded_files', [])
    if uploaded_list:
        f = uploaded_list[0]
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as tmp:
                tmp.write(f.getbuffer()); tmp.flush(); p = tmp.name
            s0 = Structure.from_file(p)
            html = make_py3dmol_html(s0, show_bonds=v_show_bonds, bond_min=v_bmin, bond_max=v_bmax, highlight_atom_index=v_atom_idx)
            vp.components = components
            components.html(html, height=520, scrolling=True)
        except Exception as e:
            vp.error(f"Viewer failed: {e}")
        finally:
            try: os.remove(p)
            except: pass
    else:
        vp.info("Upload CIF(s) in Control Board tab to preview")

# ---------------- Control Board ----------------
with tabs[1]:
    st.header("Control Board — uploads, pair ranges, coefficient editor (scientist-grade)")
    left, right = st.columns([2,1])
    with left:
        uploaded = st.file_uploader("Upload CIF(s)", type=['cif'], accept_multiple_files=True, key='uploader')
        if uploaded:
            st.session_state['uploaded_files'] = uploaded
        st.markdown("**Uploaded files**")
        for i, f in enumerate(st.session_state.get('uploaded_files', [])):
            st.write(f"{i+1}. {f.name} — {round(len(f.getbuffer())/1024,1)} KB")

        st.markdown("---")
        st.subheader("Auto-detect element list & pairs")
        if st.button("Detect elements + initialize pair ranges"):
            uploaded_list = st.session_state.get('uploaded_files', [])
            if not uploaded_list:
                st.warning("Upload CIF files first")
            else:
                all_elements = set()
                for f in uploaded_list:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as tmp:
                        tmp.write(f.getbuffer()); tmp.flush(); p = tmp.name
                    try:
                        s = Structure.from_file(p)
                        all_elements.update([str(s.specie) for s in s.sites])
                    finally:
                        try: os.remove(p)
                        except: pass
                all_elements = sorted(all_elements)
                pairs = []
                for i, e1 in enumerate(all_elements):
                    for e2 in all_elements[i:]:
                        pairs.append("-".join(sorted([e1, e2])))
                for pair in pairs:
                    if pair not in st.session_state['pair_ranges']:
                        a, b = pair.split('-')
                        est = adaptive_bond_cutoff(a, b)
                        st.session_state['pair_ranges'][pair] = (0.0, round(est*1.12,3))
                st.success(f"Initialized {len(pairs)} pairs for elements: {', '.join(all_elements)}")
                # one-time rerun to refresh editors safely
                if 'pairs_initialized' not in st.session_state:
                    st.session_state['pairs_initialized'] = True
                    st.rerun()

    with right:
        st.subheader("Global analysis settings")
        normalize_by = st.selectbox("Normalize descriptors by", options=['bonds','atoms','none'], index=0)
        compute_soap = st.checkbox("Compute SOAP (slow)", value=False)
        compute_mbtr = st.checkbox("Compute MBTR (slow)", value=False)
        soap_n = st.number_input("Keep first N SOAP features", min_value=10, max_value=1000, value=50)
        dbscan_eps = st.number_input("DBSCAN eps (layer detection)", value=0.25, step=0.01)
        dbscan_min = st.number_input("DBSCAN min_samples", value=1, step=1, min_value=1)
        st.markdown("---")
        st.subheader("Per-pair bond length editor")
        pr = st.session_state.get('pair_ranges', {})
        if pr:
            df_pairs = pd.DataFrame([{'pair':k, 'min':float(v[0]), 'max':float(v[1])} for k,v in sorted(pr.items())])
            edited = st.data_editor(df_pairs, num_rows='dynamic')
            new_map = {}
            for r in edited.to_dict(orient='records'):
                try:
                    mn = float(r['min']); mx = float(r['max'])
                except Exception:
                    old = pr.get(r['pair'], (0.0, BOND_LENGTH_FALLBACK))
                    mn, mx = old
                new_map[r['pair']] = (mn, mx)
            st.session_state['pair_ranges'] = new_map
        else:
            st.info("No pairs detected. Use the detector above.")

        st.markdown("---")
        st.subheader("Coefficient editor (manual + load/save)")
        coeff_df = pd.DataFrame([{'name':k, 'value':float(v)} for k,v in st.session_state.get('coeff_map', DEFAULT_COEFFICIENTS).items()])
        edited_coeff = st.data_editor(coeff_df, num_rows='dynamic')
        new_coeff = {}
        for row in edited_coeff.to_dict(orient='records'):
            try:
                new_coeff[str(row['name'])] = float(row['value'])
            except Exception:
                new_coeff[str(row['name'])] = 0.0
        st.session_state['coeff_map'] = new_coeff

# ---------------- Descriptors & Run ----------------
with tabs[2]:
    st.header("Descriptors & Run — batch analyze uploaded CIFs")
    st.markdown("Use the coefficients and per-pair bond ranges from Control Board. Outputs: CSV/Excel/ZIP and basic plots.")
    col1, col2 = st.columns([1,1])
    with col1:
        run = st.button("Run full analysis on uploaded files")
    with col2:
        save_last = st.button("Save current preset as 'last_preset'")
    if save_last:
        name = 'last_preset'
        st.session_state['presets'][name] = {'pair_ranges': st.session_state.get('pair_ranges', {}), 'coeff_map': st.session_state.get('coeff_map', {})}
        st.success("Saved 'last_preset'")

    if run:
        uploaded_list = st.session_state.get('uploaded_files', [])
        if not uploaded_list:
            st.warning("Upload files first in Control Board")
        else:
            progress = st.progress(0)
            results = []
            atoms_all = []
            n = len(uploaded_list)
            for i, f in enumerate(uploaded_list):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as tmp:
                        tmp.write(f.getbuffer()); tmp.flush(); path = tmp.name
                    s = Structure.from_file(path)
                    desc, atom_df = compute_descriptors_for_structure(
                        s, f.name,
                        per_pair_ranges=st.session_state.get('pair_ranges', {}),
                        normalize_by=normalize_by,
                        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min,
                        compute_soap=compute_soap, compute_mbtr=compute_mbtr, soap_first_n=int(soap_n),
                        coeff_map=st.session_state.get('coeff_map', {})
                    )
                    results.append(desc)
                    atoms_all.append(atom_df)
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
                finally:
                    try: os.remove(path)
                    except: pass
                progress.progress(int((i+1)/n*100))

            if results:
                df_desc = pd.DataFrame(results)
                df_atoms = pd.concat(atoms_all, ignore_index=True) if atoms_all else pd.DataFrame()
                st.success("Analysis finished")
                st.subheader("Descriptors (preview)")
                st.dataframe(df_desc.head(30), use_container_width=True)

                cdl, cdr = st.columns(2)
                with cdl:
                    st.download_button("Download descriptors CSV", data=df_desc.to_csv(index=False).encode('utf-8'), file_name='descriptors.csv', mime='text/csv')
                with cdr:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                        df_desc.to_excel(writer, sheet_name='descriptors', index=False)
                        df_atoms.to_excel(writer, sheet_name='per_atom', index=False)
                    buf.seek(0)
                    st.download_button("Download Excel workbook", data=buf, file_name='results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                st.markdown("---")
                st.subheader("Basic plots")
                if 'Binding_Energy_eV' in df_desc.columns:
                    st.bar_chart(df_desc['Binding_Energy_eV'])

                zipbuf = io.BytesIO()
                with zipfile.ZipFile(zipbuf, 'w') as zf:
                    zf.writestr('descriptors.csv', df_desc.to_csv(index=False))
                    zf.writestr('per_atom.csv', df_atoms.to_csv(index=False))
                zipbuf.seek(0)
                st.download_button('Download ZIP (CSV files)', data=zipbuf, file_name='results_csvs.zip', mime='application/zip')

# ---------------- Presets ----------------
with tabs[3]:
    st.header('Presets — save/load/export/import')
    name = st.text_input('Preset name', value='my_preset')
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button('Save preset'):
            st.session_state['presets'][name] = {'pair_ranges': st.session_state.get('pair_ranges', {}), 'coeff_map': st.session_state.get('coeff_map', {})}
            st.success(f"Saved preset '{name}'")
    with c2:
        presets = list(st.session_state.get('presets', {}).keys())
        if presets:
            sel = st.selectbox('Load preset', options=presets)
            if st.button('Load selected preset'):
                p = st.session_state['presets'][sel]
                st.session_state['pair_ranges'] = p.get('pair_ranges', {})
                st.session_state['coeff_map'] = p.get('coeff_map', {})
                st.success(f"Loaded '{sel}'")
        else:
            st.info('No presets saved yet')

    st.markdown('### Export / Import')
    colx, coly = st.columns([1,1])
    with colx:
        if st.button('Export presets JSON'):
            data = json.dumps(st.session_state.get('presets', {}), indent=2)
            st.download_button('Download presets JSON', data=data, file_name='presets.json', mime='application/json')
    with coly:
        up = st.file_uploader('Import presets JSON', type=['json'])
        if up:
            try:
                p = json.loads(up.getvalue().decode())
                st.session_state['presets'].update(p)
                st.success('Imported presets')
            except Exception as e:
                st.error(f'Import failed: {e}')

# ---------------- sidebar info ----------------
st.sidebar.markdown('---')
st.sidebar.header('Workstation')
st.sidebar.write('Multi-file, element-agnostic QSPR + SOAP + MBTR tool')
st.sidebar.write('Tip: detect elements after uploading CIFs, then edit per-pair ranges.')

# ---------------- End ----------------


