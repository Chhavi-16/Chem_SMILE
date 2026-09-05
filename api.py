from flask import Flask, request, jsonify

print("Loading PyTorch and Transformers... This may take a minute.")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load model
model_path = "./chemberta_model"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print(f"Local model load failed ({e}), falling back to Hugging Face seyonec/ChemBERTa-zinc-base-v1...")
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()  # ✅ IMPORTANT

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smile = data.get("smiles")

    if not smile:
        return jsonify({"error": "No SMILES string provided"}), 400

    inputs = tokenizer(smile, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    probs = torch.softmax(logits, dim=1).tolist()[0]  # ✅ clean output

    return jsonify({
        "prediction": prediction,
        "probabilities": probs
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # ✅ IMPORTANT
