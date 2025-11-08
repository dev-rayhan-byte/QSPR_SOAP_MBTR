import streamlit as st
import os
import io
import math
import tempfile
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from ase import Atoms
from dscribe.descriptors import SOAP, MBTR

# ---------------- Default settings ----------------
BOND_LENGTH_FALLBACK = 3.2
DEFAULT_DBSCAN_EPS = 0.25
DEFAULT_DBSCAN_MIN_SAMPLES = 1
NORMALIZE_CHOICES = ["bonds", "atoms", "none"]
OUTPUT_DESCRIPTORS = "QSPR_SOAP_MBTR_descriptors.csv"
OUTPUT_ATOMS = "PerAtom_layers_enhanced.csv"

# MLR coefficients (adapted placeholder)
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

# ================= Helper functions (same logic as your script) =================

def adaptive_bond_cutoff(e1, e2, scale=1.25, fallback=BOND_LENGTH_FALLBACK):
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        if any(math.isnan(x) for x in [r1, r2]):
            return fallback
        return scale * (r1 + r2)
    except Exception:
        return fallback


def detect_layers(z_coords, eps=DEFAULT_DBSCAN_EPS, min_samples=DEFAULT_DBSCAN_MIN_SAMPLES):
    z = np.array(z_coords).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(z)
    labels = clustering.labels_
    unique = sorted(set(labels))
    mapping = {lab: idx for idx, lab in enumerate(unique)}
    return np.array([mapping[lab] for lab in labels])


def find_neighbors_validated(structure, idx, max_check=64):
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
        center = structure[idx].coords
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


def structure_to_ase(structure):
    symbols = [str(site.specie) for site in structure]
    positions = np.array([site.coords for site in structure])
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


# ================= Core analysis function =================

def analyze_structure(structure: Structure, filename: str, normalize_by: str = "bonds",
                      dbscan_eps: float = DEFAULT_DBSCAN_EPS, dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
                      compute_soap: bool = True, compute_mbtr: bool = True, soap_first_n: int = 50):
    n_atoms = len(structure.sites)
    elements = [str(site.specie) for site in structure.sites]
    coords = np.array([site.coords for site in structure])
    z_coords = coords[:, 2]

    # Layers
    layer_indices = detect_layers(z_coords, eps=dbscan_eps, min_samples=dbscan_min_samples)
    layer_means = {l: np.mean(z_coords[layer_indices == l]) for l in np.unique(layer_indices)}
    label_map = {old: new for new, old in enumerate(sorted(layer_means, key=lambda L: layer_means[L]))}
    layer_indices = np.array([label_map[l] for l in layer_indices])
    n_layers = int(layer_indices.max()) + 1
    surface_mask = (layer_indices == layer_indices.min()) | (layer_indices == layer_indices.max())

    # Per-atom table
    atom_rows = [{
        "filename": filename,
        "atom_index": i,
        "element": elements[i],
        "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
        "layer_index": int(layer_indices[i]),
        "surface_flag": bool(surface_mask[i])
    } for i, c in enumerate(coords)]
    atom_df = pd.DataFrame(atom_rows)

    # Bond counting
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

    # Descriptor map
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

    # Normalization
    x_norm = {k: v for k, v in x.items()}
    if normalize_by == "atoms" and n_atoms > 0:
        x_norm = {k: v / n_atoms for k, v in x.items()}
    elif normalize_by == "bonds" and total_bonds > 0:
        x_norm = {k: v / total_bonds for k, v in x.items()}

    # Binding energy prediction
    Y = COEFFICIENTS['b0'] + sum(COEFFICIENTS.get('b_' + k, 0) * x_norm.get(k, 0) for k in x_norm)
    sigmaY = propagate_uncertainty_linear(COEFFICIENTS, x)
    desc['Binding_Energy_eV'] = float(Y)
    desc['Binding_Energy_sigma_eV_approx'] = float(sigmaY)

    # SOAP + MBTR
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
            st.warning(f"SOAP failed for {filename}: {e}")

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
            st.warning(f"MBTR failed for {filename}: {e}")

    return desc, atom_df


# ================= Batch processing =================

def process_uploaded_files(uploaded_files, normalize_by, dbscan_eps, dbscan_min_samples, compute_soap, compute_mbtr, soap_first_n, progress_callback=None):
    all_desc = []
    all_atoms = []
    n = len(uploaded_files)
    for idx, uploaded in enumerate(uploaded_files):
        fname = uploaded.name
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                tmp_path = tmp.name
            s = Structure.from_file(tmp_path)
            desc, atom_df = analyze_structure(s, fname, normalize_by=normalize_by,
                                             dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
                                             compute_soap=compute_soap, compute_mbtr=compute_mbtr, soap_first_n=soap_first_n)
            all_desc.append(desc)
            all_atoms.append(atom_df)
        except Exception as e:
            st.error(f"Error processing {fname}: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if progress_callback:
            progress_callback((idx + 1) / n)

    if not all_desc:
        return None, None

    df_desc = pd.DataFrame(all_desc)
    df_atoms = pd.concat(all_atoms, ignore_index=True)
    return df_desc, df_atoms


# ================= Streamlit UI =================

def main():
    st.set_page_config(page_title="QSPR + SOAP + MBTR Descriptor Extractor", layout="wide")
    st.title("🔬 QSPR + SOAP + MBTR Descriptor Extractor — Streamlit")

    with st.sidebar:
        st.header("Upload & Settings")
        uploaded_files = st.file_uploader("Upload CIF files", type=["cif"], accept_multiple_files=True)
        normalize_by = st.selectbox("Normalization", options=NORMALIZE_CHOICES, index=0)
        dbscan_eps = st.number_input("DBSCAN eps (layer detection)", value=float(DEFAULT_DBSCAN_EPS), step=0.01, format="%.3f")
        dbscan_min_samples = st.number_input("DBSCAN min_samples", value=int(DEFAULT_DBSCAN_MIN_SAMPLES), min_value=1, step=1)
        compute_soap = st.checkbox("Compute SOAP descriptors", value=True)
        compute_mbtr = st.checkbox("Compute MBTR descriptors", value=True)
        soap_first_n = st.number_input("Keep first N SOAP features", min_value=10, max_value=1000, value=50, step=10)
        run_button = st.button("Run analysis")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    if run_button:
        if not uploaded_files:
            st.warning("Please upload at least one CIF file.")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress(frac):
            progress_bar.progress(min(1.0, max(0.0, frac)))
            status_text.text(f"Processing... {int(min(1.0, max(0.0, frac))*100)}%")

        with st.spinner("Analyzing files — this may take some time for SOAP/MBTR..."):
            df_desc, df_atoms = process_uploaded_files(uploaded_files, normalize_by, dbscan_eps, dbscan_min_samples,
                                                      compute_soap, compute_mbtr, soap_first_n, progress_callback=_progress)

        progress_bar.empty()
        status_text.empty()

        if df_desc is None:
            st.error("No valid CIFs processed.")
            return

        # Show descriptor table (first rows)
        with col1:
            st.subheader("Descriptor table (sample)")
            st.write(df_desc.head(10))
            st.download_button("Download descriptors CSV", data=df_desc.to_csv(index=False).encode('utf-8'),
                               file_name=OUTPUT_DESCRIPTORS, mime='text/csv')

        # Show per-atom table preview
        with col2:
            st.subheader("Per-atom table (sample)")
            st.write(df_atoms.head(10))
            st.download_button("Download per-atom CSV", data=df_atoms.to_csv(index=False).encode('utf-8'),
                               file_name=OUTPUT_ATOMS, mime='text/csv')

        st.success("Processing complete — files saved to downloads when you click the buttons above.")

        # Offer save to session and display basic plots/stats
        st.markdown("---")
        st.subheader("Summary statistics")
        st.write(df_desc.describe(include='all'))

        if 'Binding_Energy_eV' in df_desc.columns:
            st.bar_chart(df_desc['Binding_Energy_eV'])

    else:
        st.info("Upload CIF files in the sidebar and click 'Run analysis' to start.")
        st.markdown("**Tips:** SOAP/MBTR are compute-heavy. For many structures, consider turning them off or increasing `soap_first_n` reduction.")


if __name__ == '__main__':
    main()
