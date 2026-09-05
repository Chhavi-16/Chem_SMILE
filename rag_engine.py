import numpy as np
import requests
import urllib.parse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

RDLogger.DisableLog('rdApp.*')

class RAGEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        self.documents = []
        self.doc_metadata = []
        self.tfidf_matrix = None

    def build(self, documents, metadata=None):
        self.documents = documents
        self.doc_metadata = metadata if metadata else [{}] * len(documents)
        if len(documents) > 0:
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query, top_k=3):
        if self.tfidf_matrix is None or len(self.documents) == 0:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.doc_metadata[idx] if idx < len(self.doc_metadata) else {},
                    "score": float(similarities[idx])
                })
            return results
        except Exception:
            return []

    def extract_smiles_or_entity(self, query):
        query_clean = query.strip()

        # Check if query itself is SMILES
        mol = Chem.MolFromSmiles(query_clean)
        if mol:
            return query_clean, "direct_smiles"

        # Check for inline SMILES inside text, e.g. `C1=CC=CC=C1`
        smiles_match = re.search(r'`([A-Za-z0-9@+\-\[\]\(\)\\\/=#$%]+)`', query)
        if smiles_match:
            candidate = smiles_match.group(1)
            if Chem.MolFromSmiles(candidate):
                return candidate, "extracted_smiles"

        # Look for potential compound name in prompt
        ignore_words = {"what", "is", "the", "chemical", "compound", "formula", "smiles", "of", "tell", "me", 
                        "about", "structure", "properties", "toxicity", "toxic", "safe", "for", "in", "and", "or",
                        "how", "does", "work", "use", "used", "can", "you", "explain", "compare", "a", "an"}
        words = [w.strip("?,.!") for w in query_clean.split() if w.lower().strip("?,.!") not in ignore_words]
        
        if words:
            potential_name = " ".join(words[:3])
            return potential_name, "entity_name"

        return None, None

    def fetch_pubchem_info(self, compound_name):
        try:
            encoded_name = urllib.parse.quote(compound_name)
            headers = {"User-Agent": "Mozilla/5.0"}
            
            # Step 1: Get CID
            url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
            res_cid = requests.get(url_cid, headers=headers, timeout=4)
            if res_cid.status_code != 200:
                return None
            data_cid = res_cid.json()
            cids = data_cid.get("IdentifierList", {}).get("CID", [])
            if not cids:
                return None
            cid = cids[0]

            # Step 2: Get Properties
            url_props = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey/JSON"
            res_props = requests.get(url_props, headers=headers, timeout=4)
            if res_props.status_code != 200:
                return None
            
            props = res_props.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            
            # Step 3: Get Description
            url_desc = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
            res_desc = requests.get(url_desc, headers=headers, timeout=4)
            description = ""
            if res_desc.status_code == 200:
                desc_data = res_desc.json()
                for item in desc_data.get("InformationList", {}).get("Information", []):
                    if "Description" in item:
                        description = item["Description"]
                        break

            return {
                "cid": cid,
                "title": props.get("Title", compound_name.title()),
                "iupac": props.get("IUPACName", "N/A"),
                "formula": props.get("MolecularFormula", "N/A"),
                "mw": props.get("MolecularWeight", "N/A"),
                "smiles": props.get("CanonicalSMILES", None),
                "description": description
            }
        except Exception:
            return None

    def compute_rdkit_details(self, smiles_str):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if not mol:
                return None
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            qed_score = QED.qed(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            lipinski_pass = all([mw <= 500, logp <= 5, h_donors <= 5, h_acceptors <= 10])

            tox_risk = "Low Risk"
            if logp > 3 or mw > 450:
                tox_risk = "Moderate Risk"
            if logp > 5:
                tox_risk = "High Risk / Bioaccumulative"

            return {
                "mw": f"{mw:.2f}",
                "logp": f"{logp:.2f}",
                "tpsa": f"{tpsa:.2f}",
                "qed": f"{qed_score:.3f}",
                "h_donors": h_donors,
                "h_acceptors": h_acceptors,
                "rotatable_bonds": rotatable_bonds,
                "lipinski": "Pass" if lipinski_pass else "Fail",
                "tox_risk": tox_risk
            }
        except Exception:
            return None

    def generate_comprehensive_response(self, query):
        # 1. Vector Search
        search_results = self.search(query, top_k=2)
        primary_match = search_results[0]["document"] if search_results and search_results[0]["score"] > 0.05 else ""

        # 2. Extract entity or SMILES
        entity, entity_type = self.extract_smiles_or_entity(query)
        
        target_smiles = None
        pubchem_info = None

        if entity_type in ("direct_smiles", "extracted_smiles"):
            target_smiles = entity
        elif entity_type == "entity_name":
            if search_results and "smiles" in search_results[0]["metadata"]:
                target_smiles = search_results[0]["metadata"]["smiles"]
            
            # PubChem API fetch
            pubchem_info = self.fetch_pubchem_info(entity)
            if pubchem_info and pubchem_info.get("smiles"):
                target_smiles = pubchem_info["smiles"]

        # Compute live RDKit properties
        rdkit_info = self.compute_rdkit_details(target_smiles) if target_smiles else None

        # Build response string
        response = "### 🧪 Scientific Analysis & Insights\n\n"
        
        if pubchem_info and pubchem_info.get("description"):
            response += f"**Overview**: {pubchem_info['description']}\n\n"
        elif primary_match:
            response += f"**Vector Knowledge Match**:\n{primary_match}\n\n"
        else:
            response += f"Here is the molecular analysis for your query regarding **{query}**.\n\n"

        if target_smiles or pubchem_info:
            response += f"### 🧬 Chemical Identifiers & Structure\n"
            if pubchem_info:
                response += f"- **Compound Name**: `{pubchem_info.get('title', 'N/A')}`\n"
                response += f"- **IUPAC Name**: `{pubchem_info.get('iupac', 'N/A')}`\n"
                response += f"- **Molecular Formula**: `{pubchem_info.get('formula', 'N/A')}`\n"
            if target_smiles:
                response += f"- **Canonical SMILES**: `{target_smiles}`\n"
            response += "\n"

        if rdkit_info:
            response += f"### 📊 ADMET & Physicochemical Profile (RDKit Computed)\n"
            response += f"- **Molecular Weight**: `{rdkit_info['mw']} g/mol`\n"
            response += f"- **LogP (Octanol-Water Partition)**: `{rdkit_info['logp']}`\n"
            response += f"- **Topological Polar Surface Area (TPSA)**: `{rdkit_info['tpsa']} Å²`\n"
            response += f"- **QED Drug-likeness Index**: `{rdkit_info['qed']}`\n"
            response += f"- **Lipinski Rule of 5**: `{rdkit_info['lipinski']}` (H-Donors: {rdkit_info['h_donors']}, H-Acceptors: {rdkit_info['h_acceptors']})\n"
            response += f"- **Estimated Toxicity Risk**: `{rdkit_info['tox_risk']}`\n\n"

        q_lower = query.lower()
        if "logp" in q_lower:
            response += "💡 **Cheminformatics Note on LogP**: LogP measures lipophilicity. Values under 5 indicate good oral absorption.\n\n"
        elif "lipinski" in q_lower or "rule of 5" in q_lower:
            response += "💡 **Cheminformatics Note on Lipinski's Rule**: Evaluates drug-likeness based on MW ≤ 500, LogP ≤ 5, H-donors ≤ 5, H-acceptors ≤ 10.\n\n"
        elif "qed" in q_lower:
            response += "💡 **Cheminformatics Note on QED**: Quantitative Estimate of Drug-likeness score (0 to 1). Scores > 0.6 indicate strong drug potential.\n\n"
        elif "tpsa" in q_lower:
            response += "💡 **Cheminformatics Note on TPSA**: Polar surface area correlates with cell permeability. TPSA < 140 Å² is required for bioavailability.\n\n"

        return {
            "text": response,
            "smiles": target_smiles,
            "rdkit_info": rdkit_info,
            "pubchem_info": pubchem_info
        }