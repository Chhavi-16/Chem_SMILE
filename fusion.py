class ChemIntelSystem:

    def __init__(self, rag_engine, model):

        self.rag = rag_engine
        self.model = model

    def run(self, query):

        # 1. Retrieve knowledge (RAG)
        retrieved_docs = self.rag.search(query)

        # 2. Model prediction
        score = self.model.predict(query)

        # 3. Combine outputs
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "prediction_score": score
        }