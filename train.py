import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd

from model import DualEncoderModel



# Load tokenizers
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
smiles_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    atoms = []

    for atom in mol.GetAtoms():

        atoms.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetHybridization()),
            float(atom.GetIsAromatic()),
            float(atom.GetMass()),
            float(atom.GetTotalNumHs()),
            float(atom.GetNumRadicalElectrons()),
            float(atom.IsInRing()),
            float(atom.GetImplicitValence())
        ])

    x = torch.tensor(
        atoms,
        dtype=torch.float
    )

    edges = []

    for bond in mol.GetBonds():

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        edges.append([start, end])
        edges.append([end, start])

    if len(edges) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    else:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index
    )
# Dataset
class ChemDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv("data.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row["text"]
        smiles = row["smiles"]
        try:
            label_value = float(row["label"])
        except:
            print("Invalid label:", row["label"])
            return self.__getitem__((idx + 1) % len(self.data))

        label = torch.tensor(label_value, dtype=torch.float)

        text_inputs = text_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        smiles_inputs = smiles_tokenizer(smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        smiles_inputs = {k: v.squeeze(0) for k, v in smiles_inputs.items()}

        graph_data = smiles_to_graph(smiles)
        

        
        return (
          text_inputs,
          smiles_inputs,
          graph_data,
          label
        )





# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ChemDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DualEncoderModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(10):
    total_loss = 0
    model.train()

    for text_inputs, smiles_inputs, graph_data, labels in loader:
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        smiles_inputs = {k: v.to(device) for k, v in smiles_inputs.items()}
        graph_data = graph_data.to(device)
        
        labels = labels.to(device).unsqueeze(1)

        outputs = model(
            text_inputs,
            smiles_inputs,
            graph_data
        )
        
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

torch.save(model.state_dict(), "dual_encoder.pth")
print("✅ Model saved!")