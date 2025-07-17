import os, csv
from embedder import EmbeddingExtractor
import numpy as np
import faiss
# for faiss-cpu not to conflict with torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FAISS:
    def __init__(self):
        #configurations
        self.model_path = "./MODEL/sr.model"
        self.embeddings_path = "./EMBEDDINGS/"
        self.meta_file_name = 'metadata.csv'
        self.index_file_name = "faiss.index"
        self.k = 1
        self.embeddings_dimention = 192
        self.treshold = 0.65

        #loading the models
        self.model = EmbeddingExtractor(self.model_path)
        self.index = self._build_index()

        #loading metadata
        self.metadata, self.header = self._load_data()

    def _load_data(self, get_meta=1):
        num_embeddings = len([f for f in os.listdir(self.embeddings_path) if f.endswith(".npy")])
        embeddings = np.zeros((num_embeddings, self.embeddings_dimention), dtype="float32")
        embeddings_name = []
        metadata=[]

        for i, file in enumerate(os.listdir(self.embeddings_path)):
            if file.endswith(".npy"):
                name = os.path.splitext(file)[0]
                emb = np.load(os.path.join(self.embeddings_path, file))
                embeddings[i] = emb
                embeddings_name.append(name)
            elif file==self.meta_file_name:
                header = []
                with open(self.embeddings_path+file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for ind, row in enumerate(reader):
                        if ind==0:
                            header = row
                        elif ind==1:
                            for i in range(len(row)):
                                metadata.append([row[i]])
                        else:
                            for i in range(len(row)):
                                metadata[i].append(row[i])
        if get_meta:
            return metadata, header
        else:
            return embeddings
    
    def _build_index(self):
        index_path = os.path.join(self.embeddings_path, self.index_file_name)

        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            self.embeddings = self._load_data(get_meta=0)
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)

            index = faiss.IndexFlatIP(self.embeddings_dimention)
            index.add(self.embeddings)

            faiss.write_index(index, index_path)

        return index
    
    def search(self, audio):
        emb = self.model.get_embedding(audio).numpy().astype("float32")

        emb = emb / np.linalg.norm(emb)

        D, I = self.index.search(emb.reshape(1, -1), self.k)

        if D[0][0] > self.treshold:
            return f"this is a: {self.metadata[0][I[0][0]]}"
        else:
            return f"This is an unknown person! {D[0][0]}"
        
    def add_new_member(self, name, audio):
        emb = self.model.get_embedding(audio).numpy().astype("float32")
        emb /= np.linalg.norm(emb)
        self.index.add(emb.reshape(1, -1))

        if not self.metadata:
            self.metadata = [[name]]
        else:
            self.metadata[0].append(name)

        self.save()

    def save(self):
        faiss.write_index(self.index, self.embeddings_path + self.index_file_name)
        rows = [self.header]
        for row in zip(*self.metadata):
            rows.append(row)


        with open(self.embeddings_path + self.meta_file_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

if __name__ == "__main__":
    faiss = FAISS()
    print(faiss.search("./samples/a2.wav"))