from model import ECAPA_TDNN
import torchaudio, torch, os
import numpy as np

class EmbeddingExtractor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ECAPA_TDNN(1024).to(self.device)
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.max_len = 16000*8

    def get_waveform(self, path):
        if isinstance(path, np.ndarray):
           arr = path
           if arr.ndim == 1:
               arr = arr[np.newaxis, :]
           waveform = torch.from_numpy(arr.copy()).float()
           sr = 16000
        else:
           waveform, sr = torchaudio.load(path)  

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]

        return waveform.unsqueeze(0).to(self.device)  # shape: [1, 192]

    def get_embedding(self, path):
        x = self.get_waveform(path)

        with torch.no_grad():
            emb = self.model(x, False)
        return emb.squeeze(0).detach().cpu()  # returns shape [192]

#emb1 = extractor.get_embedding("./samples/a1.wav")
#emb2 = extractor.get_embedding("./samples/a2.wav")

# def similarity_cosine(a, b):
#     #should be higher than 0.75
#     return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# def L2(a,b):
#     #should be under 0.75
#     a_norm = a / torch.norm(a)
#     b_norm = b / torch.norm(b)
#     return torch.norm(a_norm - b_norm)

# print("Similarity: ", similarity_cosine(emb1, emb2), "%")
# print("-It is the same person-" if similarity_cosine(emb1, emb2) > 0.75 else "-Diffrent people-")

if __name__ == "__main__":
    extractor = EmbeddingExtractor("./MODEL/sr.model")
    path_to_data = "./DATA/"
    output_dir = "./EMBEDDINGS/"
    os.makedirs(output_dir, exist_ok=True)

    data = os.listdir(path_to_data)

    for name in data:
        file_path = os.path.join(path_to_data, name)
        if not name.lower().endswith((".wav", ".flac", ".mp3")):
            continue

        emb = extractor.get_embedding(file_path)
        emb_np = emb.numpy().astype("float32")

        np.save(os.path.join(output_dir, name + ".npy"), emb_np)
