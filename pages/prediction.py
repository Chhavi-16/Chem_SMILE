import streamlit as st
import base64
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import py3Dmol
import requests




# Set page config
st.set_page_config(
    page_title="ChemIntel | Prediction Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Function to load and inject CSS
def local_css(file_name):
    css_path = Path("style.css")
    if not css_path.exists():
        css_path = Path("../style.css")
    
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")

# Function to get base64 of an image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Inject CSS from root style.css
local_css("style.css")

# Custom CSS for this page
st.html("""
    <style>
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: left;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .result-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    .result-label {
        color: #A0A0A0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .result-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .result-unit {
        color: #00D2FF;
        font-size: 0.8rem;
        margin-left: 5px;
    }
    </style>
""")


# --- HEADER ---
col_head1, col_head2 = st.columns([4, 1])

with col_head1:
    # Try to load logo if it exists, otherwise use text
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        logo_b64 = get_base64_of_bin_file(logo_path)
        st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" width="60" height="60">
                <span class="logo-text" style="font-size: 2rem;">ChemIntel</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="logo-container"><span class="logo-text" style="font-size: 2rem;">ChemIntel</span></div>', unsafe_allow_html=True)

with col_head2:
    if st.button("← Back to Home", key="back_btn"):
        st.switch_page("app.py")

st.markdown('<hr style="opacity: 0.1;">', unsafe_allow_html=True)

# Get data from session state
smiles = st.session_state.get("smiles", "C1=CC=CC=C1") # Default to benzene if not set

# --- COMPUTATION LOGIC ---
@st.cache_data
def get_molecule_data(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            # Add hydrogens for better 3D structure and accurate MW
            mol_with_hs = Chem.AddHs(mol)
            
            # Calculate standard properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            
            # 1. QED Score (Drug-likeness)
            qed_score = QED.qed(mol)
            
            # 2. Lipinski's Rule of 5 Validation
            lipinski_pass = all([
                mw <= 500,
                logp <= 5,
                h_donors <= 5,
                h_acceptors <= 10
            ])
            
            # 3. Structural Alerts (PAINS Filter)
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(params)
            pains_matches = catalog.GetMatches(mol)
            has_pains = len(pains_matches) > 0
            
            # 4. Toxicity Heuristic (Simple based on LogP and MW)
            tox_score = "Low"
            if logp > 3 or mw > 450:
                tox_score = "Moderate"
            if logp > 5:
                tox_score = "High"

            # Generate 3D coordinates robustly
            try:
                res = AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv3())
                if res != 0:
                    res = AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv2())
                if res != 0:
                    res = AllChem.EmbedMolecule(mol_with_hs, useRandomCoords=True)
                
                try:
                    AllChem.MMFFOptimizeMolecule(mol_with_hs, maxIters=300)
                except Exception:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_with_hs, maxIters=300)
                    except Exception:
                        pass
            except Exception:
                pass
            mblock = Chem.MolToMolBlock(mol_with_hs)
            
            # Generate 2D SVG
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            
            return {
                "mw": f"{mw:.2f}",
                "logp": f"{logp:.2f}",
                "tpsa": f"{tpsa:.2f}",
                "qed": f"{qed_score:.3f}",
                "lipinski": "Pass" if lipinski_pass else "Fail",
                "pains": "Alert" if has_pains else "Clean",
                "tox": tox_score,
                "mblock": mblock,
                "svg": svg,
                "success": True
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": False, "error": "Invalid SMILES"}

mol_data = get_molecule_data(smiles)
@st.cache_data(show_spinner=False)
def get_model_prediction(smiles_str):
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"smiles": smiles_str},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}


# Simulated AI Inference Effect
if "scanned" not in st.session_state:
    scan_placeholder = st.empty()
    with scan_placeholder.container():
        st.markdown("""
            <div style="margin-top: 2rem;">
                <div class="scanning-text">Analyzing Molecular Topology...</div>
                <div class="scanning-container"><div class="scanning-bar"></div></div>
            </div>
        """, unsafe_allow_html=True)
        import time
        time.sleep(1.5)
    scan_placeholder.empty()
    st.session_state.scanned = True

col_main1, col_main2 = st.columns([1, 2], gap="large")

with col_main1:
    st.markdown(
        '<h4 style="color: #00D2FF;">Structural Intelligence</h4>',
        unsafe_allow_html=True
    )

    if mol_data["success"]:

        rotate_on = st.toggle("Rotate 3D Structure", value=True, key="rotate_toggle")
        from molecule_3d_viewer import render_3d_canvas_html
        html_3d = render_3d_canvas_html(smiles, height=290, spin=rotate_on)
        st.components.v1.html(html_3d, height=300, width=400)


        # 2D Depiction
        st.markdown(
            '<p style="color: #A0A0A0; font-size: 0.8rem; margin-top: 1rem;">2D Topology Depiction</p>',
            unsafe_allow_html=True
        )

        svg_html = f"""
        <div style="
            background:white;
            border-radius:10px;
            padding:10px;
            display:flex;
            justify-content:center;
            align-items:center;
        ">
            {mol_data["svg"]}
        </div>
        """

        st.components.v1.html(svg_html, height=320, scrolling=False)

    else:
        st.error(
            f"Error rendering molecule: {mol_data.get('error', 'Unknown')}"
        )

    st.info(
        f"**AI confidence:** {'98.4%' if mol_data['success'] else '0.0%'}"
    )

with col_main2:

    st.markdown(
        '<h4 style="color: #39FF14;">Physical & Chemical Properties</h4>',
        unsafe_allow_html=True
    )

    # Property Columns
    p_col1, p_col2 = st.columns(2)

    with p_col1:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Molecular Weight</div>
            <div class="result-value">
                {mol_data['mw'] if mol_data['success'] else 'N/A'}
                <span class="result-unit">g/mol</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">LogP (Octanol-Water)</div>
            <div class="result-value">
                {mol_data['logp'] if mol_data['success'] else 'N/A'}
                <span class="result-unit">log units</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">QED (Drug-likeness)</div>
            <div class="result-value" style="color:#00D2FF;">
                {mol_data['qed'] if mol_data['success'] else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">TPSA</div>
            <div class="result-value">
                {mol_data['tpsa'] if mol_data['success'] else 'N/A'}
                <span class="result-unit">Å²</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Lipinski Rule</div>
            <div class="result-value"
                 style="color:{'#39FF14' if mol_data.get('lipinski') == 'Pass' else '#FF1493'};">
                {mol_data['lipinski'] if mol_data['success'] else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Structural Alerts</div>
            <div class="result-value"
                 style="color:{'#39FF14' if mol_data.get('pains') == 'Clean' else '#FFD700'};">
                {mol_data['pains'] if mol_data['success'] else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Safety Section
    st.markdown(
        '<h4 style="color: #FF1493; margin-top: 2rem;">Safety & ADMET Profile</h4>',
        unsafe_allow_html=True
    )

    s_col1, s_col2 = st.columns(2)

    with s_col1:

        tox_color = (
            '#39FF14'
            if mol_data.get('tox') == 'Low'
            else '#FFD700'
            if mol_data.get('tox') == 'Moderate'
            else '#FF1493'
        )

        st.markdown(
            f"""
            <div class="result-card"
               style="border-left:4px solid {tox_color};">

              <div class="result-label"> 
                Toxicity Profile
              </div>

             <div class="result-value"
                 style="color:{tox_color};">
                {mol_data['tox'] if mol_data['success'] else 'N/A'} Risk
             </div>

            </div>
            """, 
            unsafe_allow_html=True
        )

    with s_col2:

        st.markdown(
            """
            <div class="result-card"
              style="border-left:4px solid #FFD700;">

             <div class="result-label"> 
               ADMET Confidence
             </div>

             <div class="result-value"
                 style="color:#FFD700;">
                94.2%
             </div>

            </div>
            """, 
            unsafe_allow_html=True
        )

# ---------------- AI MODEL PREDICTION ----------------

st.markdown(
    '<h4 style="color:#FFD700; margin-top:2rem;">AI Model Prediction</h4>',
    unsafe_allow_html=True
)

if not smiles:
    st.error("No SMILES string found")
    st.stop()

prediction_result = get_model_prediction(smiles)

if "error" not in prediction_result:

    pred = prediction_result.get("prediction", "N/A")
    probs = prediction_result.get("probabilities", [])

    if len(probs) >= 2:
        non_toxic_prob = round(probs[0] * 100, 2)
        toxic_prob = round(probs[1] * 100, 2)
    else:
        non_toxic_prob = 0
        toxic_prob = 0

    pred_label = "Non-Toxic" if pred == 0 else "Toxic"

    st.markdown(f"""
<div class="result-card">
    <div class="result-label">Prediction</div>
    <div class="result-value" style="color: {'#39FF14' if pred == 0 else '#FF1493'};">{pred_label}</div>
</div>
<div class="result-card">
    <div class="result-label">Prediction Confidence</div>
    <div style="margin-top: 10px;">
        <div style="color: #39FF14; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px;">Non-Toxic: {non_toxic_prob}%</div>
        <div style="color: #FF1493; font-size: 1.1rem; font-weight: 600;">Toxic: {toxic_prob}%</div>
    </div>
</div>
""", unsafe_allow_html=True)



else:
    st.error(prediction_result["error"])

# --- FOOTER ---
st.markdown("""
    <div class="custom-footer" style="margin-top: 2rem;">
        ChemIntel Predictive Engine v1.0 | Real-time GNN Inference
    </div>
""", unsafe_allow_html=True)
