"""
Streamlit Scientific Workstation
QSPR + SOAP + MBTR + interactive 3D viewer + bond customization
Author: Adapted for the user (2025)
"""

import os
import io
import math
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from ase import Atoms

# DScribe
from dscribe.descriptors import SOAP, MBTR

# 3D viewer helper: py3Dmol (provides HTML for the viewer)
import py3Dmol

# ------------------ Settings & defaults ------------------
st.set_page_config(page_title="Scientific Workstation — QSPR + SOAP/MBTR", layout="wide")
BOND_LENGTH_FALLBACK = 3.2
DEFAULT_DBSCAN_EPS = 0.25
DEFAULT_DBSCAN_MIN_SAMPLES = 1
DEFAULT_SOAP_N = 50

COEFFICIENTS = {
    'b0': 0.0000,
    'b_Pt-S': -0.1225,
    'b_Ru-S': 0.0000,
    'b_Pt-Ru': 0.0050,
    'b_Ru-Ru': -0.0160,
    'b_Ru-Ru-': -0.1020,
    'b_Pt-Pt': 0.0048,
    'b_Pt-Pt-': 0.0550
}

cnn = CrystalNN(weighted_cn=False)

# ------------------ Helper functions ------------------
def adaptive_bond_cutoff(e1, e2, scale=1.25, fallback=BOND_LENGTH_FALLBACK):
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        if any(math.isnan(x) for x in [r1, r2]):
            return fallback
        return scale * (r1 + r2)
    except Exception:
        return fallback

def find_neighbors_validated(structure, idx, max_check=128):
    neighbors_idx = set()
    try:
        for info in cnn.get_nn_info(structure, idx):
            j = info["site_index"]
            e1, e2 = str(structure[idx].specie), str(structure[j].specie)
            d = structure.get_distance(idx, j)
            if d <= adaptive_bond_cutoff(e1, e2) * 1.05:
                neighbors_idx.add(j)
    except Exception:
        pass

    if not neighbors_idx:
        for j, site in enumerate(structure.sites):
            if j == idx:
                continue
            e1, e2 = str(structure[idx].specie), str(site.specie)
            d = structure.get_distance(idx, j)
            if d <= adaptive_bond_cutoff(e1, e2):
                neighbors_idx.add(j)
            if len(neighbors_idx) >= max_check:
                break
    return sorted(neighbors_idx)

def structure_to_ase(structure):
    symbols = [str(s.specie) for s in structure]
    positions = np.array([s.coords for s in structure])
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def detect_layers(z_coords, eps=DEFAULT_DBSCAN_EPS, min_samples=DEFAULT_DBSCAN_MIN_SAMPLES):
    z = np.array(z_coords).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(z)
    labels = clustering.labels_
    unique = sorted(set(labels))
    mapping = {lab: idx for idx, lab in enumerate(unique)}
    return np.array([mapping[lab] for lab in labels])

def propagate_uncertainty_linear(coeff_map, x_counts):
    sigma2 = 0.0
    for ck, b in coeff_map.items():
        if ck == 'b0':
            continue
        xkey = ck.replace('b_', '')
        if xkey in x_counts:
            N = max(x_counts[xkey], 0)
            sigma_x = math.sqrt(N)
            sigma2 += (b * sigma_x) ** 2
    return math.sqrt(sigma2)

# ------------------ Descriptor/analysis core ------------------
def analyze_structure(structure, filename,
                      normalize_by="bonds", dbscan_eps=DEFAULT_DBSCAN_EPS,
                      dbscan_min_samples=DEFAULT_DBSCAN_MIN_SAMPLES,
                      compute_soap=True, compute_mbtr=True, soap_first_n=DEFAULT_SOAP_N):
    n_atoms = len(structure.sites)
    elements = [str(site.specie) for site in structure.sites]
    coords = np.array([site.coords for site in structure])
    z_coords = coords[:, 2]

    # Layers mapping
    layer_indices = detect_layers(z_coords, eps=dbscan_eps, min_samples=dbscan_min_samples)
    layer_means = {l: np.mean(z_coords[layer_indices == l]) for l in np.unique(layer_indices)}
    label_map = {old: new for new, old in enumerate(sorted(layer_means, key=lambda L: layer_means[L]))}
    layer_indices = np.array([label_map[l] for l in layer_indices])
    n_layers = int(layer_indices.max()) + 1
    surface_mask = (layer_indices == layer_indices.min()) | (layer_indices == layer_indices.max())

    # per-atom rows
    atom_rows = [{
        "filename": filename,
        "atom_index": i,
        "element": elements[i],
        "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
        "layer_index": int(layer_indices[i]),
        "surface_flag": bool(surface_mask[i])
    } for i, c in enumerate(coords)]
    atom_df = pd.DataFrame(atom_rows)

    # bond counts
    bond_counts_intra = defaultdict(int)
    bond_counts_inter = defaultdict(int)
    coordination = np.zeros(n_atoms)
    for i in range(n_atoms):
        neighs = find_neighbors_validated(structure, i)
        coordination[i] = len(neighs)
        for j in neighs:
            if j <= i:
                continue
            e1, e2 = elements[i], elements[j]
            pair = "-".join(sorted([e1, e2]))
            d = structure.get_distance(i, j)
            cutoff = adaptive_bond_cutoff(e1, e2)
            if d > cutoff * 1.05:
                continue
            if layer_indices[i] == layer_indices[j]:
                bond_counts_intra[pair] += 1
            else:
                bond_counts_inter[pair] += 1

    surface_counts = defaultdict(int)
    for i, surf in enumerate(surface_mask):
        if surf:
            surface_counts[f"{elements[i]}-S"] += 1

    x = {
        'Pt-S': surface_counts.get('Pt-S', 0),
        'Ru-S': surface_counts.get('Ru-S', 0),
        'Pt-Ru': bond_counts_intra.get('Pt-Ru', 0) + bond_counts_inter.get('Pt-Ru', 0),
        'Pt-Pt': bond_counts_intra.get('Pt-Pt', 0),
        'Ru-Ru': bond_counts_intra.get('Ru-Ru', 0),
        'Pt-Pt-': bond_counts_inter.get('Pt-Pt', 0),
        'Ru-Ru-': bond_counts_inter.get('Ru-Ru', 0)
    }

    total_bonds = sum(bond_counts_intra.values()) + sum(bond_counts_inter.values())
    desc = {
        "filename": filename,
        "n_atoms": n_atoms,
        "n_layers": n_layers,
        "avg_coordination": float(np.mean(coordination)) if n_atoms > 0 else 0.0,
        "surface_ratio": float(np.sum(surface_mask) / n_atoms) if n_atoms > 0 else 0.0,
        "total_bonds": int(total_bonds)
    }
    desc.update(x)

    # normalization
    x_norm = {k: v for k, v in x.items()}
    if normalize_by == "atoms" and n_atoms > 0:
        x_norm = {k: v / n_atoms for k, v in x.items()}
    elif normalize_by == "bonds" and total_bonds > 0:
        x_norm = {k: v / total_bonds for k, v in x.items()}

    # binding energy prediction
    Y = COEFFICIENTS['b0'] + sum(COEFFICIENTS.get('b_' + k, 0) * x_norm.get(k, 0) for k in x_norm)
    sigmaY = propagate_uncertainty_linear(COEFFICIENTS, x)
    desc['Binding_Energy_eV'] = float(Y)
    desc['Binding_Energy_sigma_eV_approx'] = float(sigmaY)

    # SOAP/MBTR
    ase_atoms = structure_to_ase(structure)
    species = list(set(ase_atoms.get_chemical_symbols()))

    if compute_soap:
        try:
            soap = SOAP(species=species, rcut=5.0, nmax=6, lmax=4, sigma=0.5, periodic=True)
            soap_vec = soap.create(ase_atoms)
            soap_mean = np.mean(soap_vec, axis=0)
            for i, val in enumerate(soap_mean[:soap_first_n]):
                desc[f"SOAP_{i}"] = float(val)
        except Exception as e:
            st.warning(f"SOAP failed: {e}")

    if compute_mbtr:
        try:
            mbtr = MBTR(
                species=species,
                k2={"geometry": {"function": "inverse_distance"}, "grid": {"min": 0, "max": 1, "n": 30}},
                k3={"geometry": {"function": "angle"}, "grid": {"min": 0, "max": 180, "n": 30}},
                periodic=True,
                normalization="l2_each"
            )
            mbtr_vec = mbtr.create(ase_atoms)
            desc["MBTR_mean"] = float(np.mean(mbtr_vec))
            desc["MBTR_std"] = float(np.std(mbtr_vec))
        except Exception as e:
            st.warning(f"MBTR failed: {e}")

    return desc, atom_df

# ------------------ 3D visualization ------------------
def make_py3dmol_view(structure, show_bonds=True, bond_min=0.0, bond_max=3.0,
                      highlight_atom_index=None, show_poly=False, poly_neighbors=None,
                      atom_size=0.7):
    """Return HTML of py3Dmol viewer for an ASE/molecule-like structure (uses pymatgen Structure)"""
    atoms = structure
    # feed as x,y,z + element
    view = py3Dmol.view(width=700, height=500)
    coords = np.array([s.coords for s in atoms.sites])
    elements = [str(s.specie) for s in atoms.sites]

    # Add atoms
    for i, (el, pos) in enumerate(zip(elements, coords)):
        style = {"sphere": {"radius": atom_size}}
        if highlight_atom_index is not None and i == highlight_atom_index:
            view.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                            'radius': atom_size * 1.4, 'color': 'red', 'opacity': 1.0})
        view.addModel({"atoms": [{"elem": el, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])} for _ in [0]]}, "sdf")
        # simpler: add sphere by element color using built-in color scheme
        view.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                        'radius': atom_size, 'color': py3Dmol.elementColors.get(el.capitalize(), '#AAAAAA')})

    # Add bonds (simple geometric)
    if show_bonds:
        n = len(elements)
        for i in range(n):
            for j in range(i+1, n):
                d = structure.get_distance(i, j)
                if bond_min <= d <= bond_max:
                    view.addCylinder({'start': {'x': float(coords[i,0]), 'y': float(coords[i,1]), 'z': float(coords[i,2])},
                                      'end': {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                                      'radius': atom_size*0.25, 'color': 'lightgrey', 'opacity': 1.0})

    # Polyhedron: draw lines from highlighted atom to neighbors
    if show_poly and poly_neighbors is not None and highlight_atom_index is not None:
        center = coords[highlight_atom_index]
        for j in poly_neighbors:
            view.addCylinder({'start': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                              'end': {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                              'radius': atom_size*0.12, 'color': 'orange', 'opacity': 1.0})

    view.setBackgroundColor('#ffffff')
    view.zoomTo()
    html = view.show()
    return html

# ------------------ Streamlit UI layout ------------------
st.markdown("<style> .big-font { font-size:22px; font-weight:600; } .muted { color:#6c757d } </style>", unsafe_allow_html=True)
st.title("🔬 Scientific Workstation — QSPR + SOAP + MBTR")

# Sidebar: upload + settings
with st.sidebar:
    st.header("Input & Viewer Settings")
    uploaded = st.file_uploader("Upload CIF files", type=["cif"], accept_multiple_files=True)
    st.markdown("---")
    st.subheader("Bond search")
    element_filter_toggle = st.checkbox("Limit pairs by element selection", value=False)
    min_bond = st.number_input("Min bond length (Å)", value=0.0, step=0.05, format="%.2f")
    max_bond = st.number_input("Max bond length (Å)", value=3.0, step=0.05, format="%.2f")
    show_bonds = st.checkbox("Show bonds", value=True)
    show_poly = st.checkbox("Show coordination polyhedra", value=False)
    dbscan_eps = st.number_input("DBSCAN eps (layer detect)", value=float(DEFAULT_DBSCAN_EPS), step=0.01, format="%.3f")
    dbscan_min = st.number_input("DBSCAN min_samples", value=int(DEFAULT_DBSCAN_MIN_SAMPLES), min_value=1, step=1)

    st.markdown("---")
    st.subheader("Descriptors")
    normalize_by = st.selectbox("Normalize descriptors by", options=["bonds","atoms","none"], index=0)
    compute_soap = st.checkbox("Compute SOAP", value=True)
    compute_mbtr = st.checkbox("Compute MBTR", value=True)
    soap_n = st.number_input("Keep first N SOAP features", min_value=10, max_value=1000, value=50, step=10)
    run_button = st.button("Run analysis on uploaded files")

# Main columns
col_left, col_right = st.columns([1.2, 1])

if uploaded:
    # Parse first file for preview & element list
    first = uploaded[0]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
            tmp.write(first.getbuffer())
            tmp.flush()
            tmp_path = tmp.name
        struct_preview = Structure.from_file(tmp_path)
        elements_unique = sorted(list(set([str(s.specie) for s in struct_preview.sites])))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # element selectors for restricted bond search
    if element_filter_toggle:
        a1 = st.sidebar.selectbox("Atom A1 (element)", options=elements_unique, index=0)
        a2 = st.sidebar.selectbox("Atom A2 (element)", options=elements_unique, index=0)
    else:
        a1 = a2 = None

    # viewer interactive controls
    with col_left:
        st.subheader("3D Structure Viewer")
        viewer_holder = st.empty()
        try:
            html = make_py3dmol_view(struct_preview, show_bonds=show_bonds, bond_min=min_bond, bond_max=max_bond,
                                     highlight_atom_index=None, show_poly=show_poly, poly_neighbors=None)
            viewer_holder.components.v1.html(html, height=520, scrolling=True)
        except Exception as e:
            st.error(f"Viewer failed: {e}")

    with col_right:
        st.subheader("Structure details")
        st.write(f"File: **{first.name}**")
        st.write(f"Atoms: **{len(struct_preview.sites)}**  — Elements: {', '.join(elements_unique)}")
        st.write("Select an atom index to highlight & inspect neighbors:")
        atom_index = st.number_input("Highlight atom index (0-based)", min_value=0, max_value=max(0,len(struct_preview.sites)-1), value=0, step=1)
        if st.button("Highlight & show neighbors"):
            neighs = find_neighbors_validated(struct_preview, int(atom_index))
            html2 = make_py3dmol_view(struct_preview, show_bonds=show_bonds, bond_min=min_bond, bond_max=max_bond,
                                      highlight_atom_index=int(atom_index), show_poly=show_poly, poly_neighbors=neighs)
            viewer_holder.components.v1.html(html2, height=520, scrolling=True)
            st.write("Neighbors indices:", neighs)

else:
    st.info("Upload one or more CIF files in the sidebar to start. Viewer shows the first file as preview.")

# Run analysis when requested
if run_button:
    if not uploaded:
        st.warning("Please upload CIF files before running.")
    else:
        st.info("Processing files — SOAP/MBTR can be slow for many structures.")
        progress = st.progress(0)
        desc_list = []
        atom_tables = []
        n_files = len(uploaded)
        for i, up in enumerate(uploaded):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                    tmp.write(up.getbuffer())
                    tmp.flush()
                    path = tmp.name
                s = Structure.from_file(path)
                desc, atom_df = analyze_structure(
                    s, up.name,
                    normalize_by=normalize_by,
                    dbscan_eps=dbscan_eps,
                    dbscan_min_samples=dbscan_min,
                    compute_soap=compute_soap,
                    compute_mbtr=compute_mbtr,
                    soap_first_n=int(soap_n)
                )
                desc_list.append(desc)
                atom_tables.append(atom_df)
            except Exception as e:
                st.error(f"Error processing {up.name}: {e}")
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
            progress.progress(int((i+1)/n_files*100))

        if desc_list:
            df_desc = pd.DataFrame(desc_list)
            df_atoms = pd.concat(atom_tables, ignore_index=True)
            st.success("Processing complete — results below.")

            st.subheader("Descriptor table (sample)")
            st.dataframe(df_desc.head(10), use_container_width=True)
            csv1 = df_desc.to_csv(index=False).encode('utf-8')
            st.download_button("Download descriptors CSV", data=csv1, file_name="QSPR_SOAP_MBTR_descriptors.csv", mime="text/csv")

            st.subheader("Per-atom table (sample)")
            st.dataframe(df_atoms.head(12), use_container_width=True)
            csv2 = df_atoms.to_csv(index=False).encode('utf-8')
            st.download_button("Download per-atom CSV", data=csv2, file_name="PerAtom_layers_enhanced.csv", mime="text/csv")

            st.markdown("---")
            st.subheader("Basic plots")
            if "Binding_Energy_eV" in df_desc.columns:
                st.bar_chart(df_desc["Binding_Energy_eV"])

            st.write("Summary statistics:")
            st.write(df_desc.describe(include='all'))

