from rag_engine import RAGEngine
from model import ChemModel
from fusion import ChemIntelSystem

# Initialize components
rag = RAGEngine()
model = ChemModel()

# Build memory (VERY IMPORTANT)
rag.build([
    "Aspirin is a pain reliever",
    "Ethanol is used as fuel alcohol",
    "Benzene is toxic aromatic compound"
])

# Combine system
system = ChemIntelSystem(rag, model)

# Run test query
result = system.run("pain relief drug")

print(result)