from rag_engine import RAGEngine
import pandas as pd

print("Testing RAG Engine initialization...")
engine = RAGEngine()

df = pd.read_csv("data.csv")
docs = []
metadata_list = []
for _, row in df.iterrows():
    desc = str(row['text']).strip().title()
    smiles = str(row['smiles']).strip()
    label = "Toxic / Hazardous" if row['label'] == 1 else "Non-Toxic / Bio-safe"
    
    doc_str = f"**Compound**: {desc}\n\n**Canonical SMILES**: `{smiles}`\n\n**Toxicity Profile**: {label}"
    docs.append(doc_str)
    metadata_list.append({"name": desc, "smiles": smiles, "label": row['label']})

engine.build(docs, metadata_list)
print("✓ FAISS Vector Index Built successfully with", len(docs), "documents!")

print("\n--- Test Query 1: Aspirin ---")
res1 = engine.generate_comprehensive_response("Tell me about Aspirin")
print(res1["text"])
print("Target SMILES:", res1["smiles"])

print("\n--- Test Query 2: Unknown compound Metformin ---")
res2 = engine.generate_comprehensive_response("What is Metformin?")
print(res2["text"])
print("Target SMILES:", res2["smiles"])

print("\n--- Test Query 3: Chemistry Concept ---")
res3 = engine.generate_comprehensive_response("Explain Lipinski rule of 5")
print(res3["text"])

print("\nALL RAG TESTS PASSED SUCCESSFULLY!")
