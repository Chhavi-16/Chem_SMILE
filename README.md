# 🧪 ChemIntel

ChemIntel is a Precision Property Prediction dashboard for chemical molecules. It leverages the **ChemBERTa** machine learning model for toxicity predictions and **RDKit** for in-depth structural and topological analysis, all presented through a highly interactive **Streamlit** user interface.

### 🌐 Live Demo
*Note: This link is temporarily hosted via LocalTunnel from the developer's machine.*
👉 **[Live Link: https://sixty-spiders-boil.loca.lt](https://sixty-spiders-boil.loca.lt)**

---

## ✨ Key Features
- **SMILES & Name Parsing:** Input raw SMILES strings or search by common chemical names (via PubChem integration).
- **Physical & Chemical Properties:** Real-time computation of Molecular Weight, LogP, TPSA, and QED (Drug-likeness) using RDKit.
- **Safety & Toxicity Profiling:** Validates against Lipinski's Rule of 5 and PAINS structural alerts. Uses a fine-tuned ChemBERTa deep learning model to predict toxicity probabilities.
- **3D Molecular Rendering:** Interactive 3D topology views (ball & stick style) and 2D chemical structure drawings.

## 🛠 Tech Stack
- **Frontend:** Streamlit, py3Dmol
- **Backend API:** Flask
- **Machine Learning:** PyTorch, Transformers (Hugging Face)
- **Cheminformatics:** RDKit

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Chhavi-16/SMILE_chem.git
cd SMILE_chem-main
```

### 2. Install dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r CHEMRAG/requirements.txt
```
*(Note: Ensure you have `rdkit`, `streamlit`, `flask`, `torch`, and `transformers` installed.)*

### 3. Start the Inference API
The ML models run on a separate Flask backend. You must start this first:
```bash
python api.py
```
*(This will start the API on http://127.0.0.1:5000/)*

### 4. Start the Streamlit Frontend
Open a new terminal window and run:
```bash
streamlit run app.py
```

---

### ⚠️ Note on Large Model Files
Due to GitHub's file size constraints, the underlying heavy machine learning model files (`*.pth` and `*.safetensors`) have been excluded from this repository via `.gitignore`. To run this project locally, you will need to place the pre-trained `chemberta_model` folder into the root directory.
