import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ChemModel:

    def __init__(self, model_name="distilbert-base-uncased"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def encode(self, text):

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        return outputs.last_hidden_state[:, 0, :]

    def predict(self, text):

        emb = self.encode(text)
        out = self.classifier(emb)

        return out.item()


class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.smiles_encoder = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        
        self.fc = nn.Sequential(
            nn.Linear(768 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, text_inputs, smiles_inputs, graph_data=None):
        text_out = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        smiles_out = self.smiles_encoder(**smiles_inputs).last_hidden_state[:, 0, :]
        combined = torch.cat((text_out, smiles_out), dim=1)
        return self.fc(combined)