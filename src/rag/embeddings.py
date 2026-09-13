# embeddings.py
import os
from openai import OpenAI

class LocalEmbedding:
    def __init__(self, model_name=None, dimensions=None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.dimensions = int(dimensions or os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

    def embed_documents(self, texts):
        all_vectors = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            all_vectors.extend([d.embedding for d in resp.data])
        return all_vectors

    def embed_query(self, text):
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return resp.data[0].embedding