from __future__ import annotations
import numpy as np
import gensim.downloader as api
from pathlib import Path


class Word2VecExtractor:
    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        self.model_name = model_name
        self.model = None

    def fit(self, texts: list[str]) -> "Word2VecExtractor":
        """Load the pretrained model (no fitting needed for pretrained)."""
        if self.model is None:
            print(f"  Loading pretrained model: {self.model_name} …")
            self.model = api.load(self.model_name)
        return self

    def transform(
        self,
        texts: list[str],
        max_len: int = 200
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        embeddings = []
        for text in texts:
            words = text.split()[:max_len]
            vecs = []
            for w in words:
                if w in self.model:
                    vecs.append(self.model[w])
                else:
                    vecs.append(np.zeros(self.model.vector_size))
            while len(vecs) < max_len:
                vecs.append(np.zeros(self.model.vector_size))
            embeddings.append(np.array(vecs))
        return np.array(embeddings)

    def fit_transform(self, texts: list[str], max_len: int = 100) -> np.ndarray:
        return self.fit(texts).transform(texts, max_len=max_len)

    def transform_and_save(self, texts, path, max_len=100, chunk_size=1000):
        if self.model is None:
            raise RuntimeError("Call fit() before transform_and_save().")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            embeddings = []

            for text in chunk:
                words = text.split()[:max_len]
                vecs  = []
                for w in words:
                    if w in self.model:
                        vecs.append(self.model[w].astype(np.float32))
                    else:
                        vecs.append(np.zeros(self.model.vector_size, dtype=np.float32))
                while len(vecs) < max_len:
                    vecs.append(np.zeros(self.model.vector_size, dtype=np.float32))
                embeddings.append(np.array(vecs, dtype=np.float32))

            chunks.append(np.array(embeddings, dtype=np.float32))
            print(f"  Processed {min(i+chunk_size, len(texts))}/{len(texts)}")

        np.save(str(path), np.concatenate(chunks, axis=0))

    def save_embeddings(self, embeddings, path):
        np.save(str(path), embeddings)