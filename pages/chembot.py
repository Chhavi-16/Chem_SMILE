import streamlit as st
import base64
from pathlib import Path
from rag_engine import RAGEngine
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ChemIntel | ChemBot AI",
    page_icon="🤖",
    layout="wide"
)

# --- INITIALIZE RAG ---
@st.cache_resource
def init_rag():
    engine = RAGEngine()
    try:
        df = pd.read_csv("data.csv")
        docs = []
        metadata_list = []
        for _, row in df.iterrows():
            desc = str(row['text']).strip().title()
            smiles = str(row['smiles']).strip()
            label = "Toxic / Hazardous" if row['label'] == 1 else "Non-Toxic / Bio-safe"
            
            doc_str = (
                f"**Compound**: {desc}\n\n"
                f"**Canonical SMILES**: `{smiles}`\n\n"
                f"**Toxicity Profile**: {label}"
            )
            docs.append(doc_str)
            metadata_list.append({
                "name": desc,
                "smiles": smiles,
                "label": row['label']
            })
            
        engine.build(docs, metadata_list)
    except Exception as e:
        engine.build(["Database loaded with fallback knowledge base."])
        
    return engine

rag_engine = init_rag()

# --- CSS INJECTION ---
def local_css():
    css_path = Path("style.css")
    if not css_path.exists():
        css_path = Path("../style.css")
    
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

local_css()

# Custom Styling for ChemBot
st.markdown("""
    <style>
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .user-msg {
        background: rgba(0, 210, 255, 0.08);
        border-left: 4px solid #00D2FF;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .bot-msg {
        background: rgba(57, 255, 20, 0.06);
        border-left: 4px solid #39FF14;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .prompt-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 14px;
        border-radius: 20px;
        color: #00D2FF;
        font-size: 0.85rem;
        cursor: pointer;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .prompt-chip:hover {
        background: rgba(0, 210, 255, 0.2);
        border-color: #00D2FF;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_head1, col_head2 = st.columns([4, 1])

with col_head1:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        logo_b64 = get_base64_of_bin_file(logo_path)
        st.markdown(f"""
            <div class="logo-container" style="display:flex; align-items:center;">
                <img src="data:image/png;base64,{logo_b64}" width="50" height="50">
                <span class="logo-text" style="font-size: 2rem; margin-left: 12px; font-weight:700;">ChemIntel</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="logo-container"><span class="logo-text" style="font-size: 2rem; font-weight:700;">ChemIntel</span></div>', unsafe_allow_html=True)

with col_head2:
    if st.button("← Home Page", key="back_btn"):
        st.switch_page("app.py")

st.markdown('<hr style="opacity: 0.1;">', unsafe_allow_html=True)

# --- MAIN TITLE & PROMPT SUGGESTIONS ---
st.title("🤖 ChemBot RAG Assistant")
st.markdown('<p class="tagline">Ask any question about chemical compounds, SMILES structures, toxicity, drug-likeness, or safety guidance.</p>', unsafe_allow_html=True)

# Suggested prompts
st.markdown("##### 💡 Suggested Quick Queries")
chip_cols = st.columns([1, 1, 1, 1, 1])

selected_chip = None

with chip_cols[0]:
    if st.button("💊 Aspirin vs Ibuprofen", key="chip1", use_container_width=True):
        selected_chip = "Tell me about Aspirin and how it compares to Ibuprofen"
with chip_cols[1]:
    if st.button("☕ Caffeine Profile", key="chip2", use_container_width=True):
        selected_chip = "What is the SMILES, MW, and toxicity profile of Caffeine?"
with chip_cols[2]:
    if st.button("☣️ Is Benzene Toxic?", key="chip3", use_container_width=True):
        selected_chip = "Is Benzene toxic and carcinogenic?"
with chip_cols[3]:
    if st.button("📐 Lipinski's Rule", key="chip4", use_container_width=True):
        selected_chip = "Explain Lipinski's Rule of 5 in chemistry"
with chip_cols[4]:
    if st.button("🔬 What is LogP?", key="chip5", use_container_width=True):
        selected_chip = "What does LogP mean for chemical compounds?"

st.markdown("<br>", unsafe_allow_html=True)

# --- CHAT CONTAINER & HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am **ChemBot AI**. Ask me anything about chemical compounds, SMILES strings, toxicity predictions, molecular weights, or cheminformatics!",
            "smiles": None
        }
    ]

# Action bar for clearing chat
c_col1, c_col2 = st.columns([6, 1])
with c_col2:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Chat history cleared. How else can I assist your chemical research?",
                "smiles": None
            }
        ]
        st.rerun()

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg"><strong>👤 You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><strong>🤖 ChemBot AI:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        
        # If the bot message has associated SMILES, render 2D depiction & action button
        smiles_val = msg.get("smiles")
        if smiles_val:
            try:
                mol = Chem.MolFromSmiles(smiles_val)
                if mol:
                    st.markdown("##### 🔬 Interactive Compound Visualizer")
                    v_col1, v_col2 = st.columns([1, 1])
                    
                    # 2D SVG
                    with v_col1:
                        drawer = rdMolDraw2D.MolDraw2DSVG(280, 200)
                        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                        drawer.FinishDrawing()
                        svg = drawer.GetDrawingText()
                        st.components.v1.html(
                            f'<div style="background:white; border-radius:8px; padding:6px; display:flex; justify-content:center;">{svg}</div>',
                            height=210
                        )
                    
                    # 3D HTML5 Canvas Viewer
                    with v_col2:
                        from molecule_3d_viewer import render_3d_canvas_html
                        html_3d = render_3d_canvas_html(smiles_val, height=200, spin=True)
                        st.components.v1.html(html_3d, height=210)
                        
                    # Button to transfer to Prediction Dashboard
                    if st.button(f"🔬 Analyze `{smiles_val}` in Prediction Dashboard", key=f"nav_pred_{idx}"):
                        st.session_state.smiles = smiles_val
                        st.session_state.original_input = smiles_val
                        st.switch_page("pages/prediction.py")
            except Exception as err:
                pass

# --- INPUT HANDLING ---
user_prompt = selected_chip or st.chat_input("Type your chemistry question or paste SMILES...")

if user_prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt, "smiles": None})
    
    # Process through RAG Engine
    with st.spinner("Analyzing molecular database & running PubChem / RDKit synthesis..."):
        res = rag_engine.generate_comprehensive_response(user_prompt)
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": res["text"],
        "smiles": res["smiles"]
    })
    
    st.rerun()

# --- FOOTER ---
st.markdown("""
    <hr style="opacity:0.1; margin-top:3rem;">
    <div style="text-align:center; color:#A0A0A0; font-size:0.8rem; padding-bottom:1rem;">
        ChemBot v2.0 | Multi-Modal RAG Engine with FAISS, PubChem & RDKit
    </div>
""", unsafe_allow_html=True)
