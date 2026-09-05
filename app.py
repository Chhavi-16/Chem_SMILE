import streamlit as st
import requests
import base64
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ChemIntel | Prediction",
    page_icon="🧪",
    layout="wide"
)


# ---------------- HELPERS ----------------
def is_smiles(input_text):
    special_chars = ["=", "#", "(", ")", "[", "]"]
    return any(char in input_text for char in special_chars)


import requests
import urllib.parse

def text_to_smiles(name):
    name = name.strip()
    encoded_name = urllib.parse.quote(name)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        # STEP 1: CID request
        url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        res_cid = requests.get(url_cid, headers=headers, timeout=10)

        if res_cid.status_code != 200:
            return None

        data_cid = res_cid.json()

        if "IdentifierList" not in data_cid:
            return None

        cid = data_cid["IdentifierList"]["CID"][0]

        # STEP 2: SMILES request
        url_smiles = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,ConnectivitySMILES/JSON"
        res_smiles = requests.get(url_smiles, headers=headers, timeout=10)

        if res_smiles.status_code != 200:
            return None

        data_smiles = res_smiles.json()

        props = data_smiles["PropertyTable"]["Properties"][0]

        smiles = (
            props.get("CanonicalSMILES")
            or props.get("ConnectivitySMILES")
        )

        return smiles

    except Exception as e:
        print("ERROR:", e)
        return None

# ---------------- CSS ----------------
def local_css(file_name):
    css_path = Path(file_name)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.html(f"<style>{css_content}</style>")

local_css("style.css")


# ---------------- LOGO ----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

logo_path = Path("assets/logo.png")

if logo_path.exists():
    logo_b64 = get_base64_of_bin_file(logo_path)
    st.markdown(f"""
        <div style="display:flex; align-items:center;">
            <img src="data:image/png;base64,{logo_b64}" width="80">
            <h2 style="margin-left:10px;">ChemIntel</h2>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("🧪 ChemIntel")

# ---------------- TITLE ----------------
st.markdown("## Precision Property Prediction")

# ---------------- INPUT ----------------
user_input = st.text_input(
    "Enter SMILES or Molecule Name",
    placeholder="e.g. C1=CC=CC=C1 OR aspirin"
)

# ---------------- BUTTON ----------------
if st.button("Predict Chemical Properties", type="primary"):

    if not user_input or user_input.strip() == "":
        st.warning("Please enter a valid input")
        st.stop()

    user_input_clean = user_input.strip()

    # ---------------- SMILES INPUT ----------------
    source = None
    smiles = None
    
    if is_smiles(user_input_clean):
        smiles = user_input_clean
        source = "SMILES input"

    # ---------------- NAME INPUT ----------------
    else:
        with st.spinner("Converting name to SMILES..."):
            smiles = text_to_smiles(user_input_clean)

        # ---------------- FALLBACK ----------------
        if not smiles:
            fallback = {
                "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "benzene": "C1=CC=CC=C1",
                "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
                "nicotine": "CN1CCC[C@H]1c2cccnc2",
                "ethanol": "CCO"
            }

            smiles = fallback.get(user_input_clean.lower())

            if smiles:
                source = "Resolved via fallback (offline)"
            else:
                source = None

    # ---------------- FINAL CHECK ----------------
    if not smiles:
        st.error(f"❌ Molecule not found for: {user_input_clean}")
        st.stop()



    # ---------------- STORE ----------------
    st.session_state.smiles = smiles
    st.session_state.original_input = user_input_clean
    st.session_state.source = source

    st.switch_page("pages/prediction.py")

# ---------------- FOOTER ----------------
st.markdown("""
---
© 2026 ChemIntel | AI for Chemistry
""") 