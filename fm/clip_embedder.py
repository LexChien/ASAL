import numpy as np
import torch

try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False

class VLM:
    """Wrapper for VLM_img / VLM_txt embeddings (CLIP if available; else deterministic random).
    """
    def __init__(self, device=None, seed=0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rng = np.random.RandomState(seed)
        if _HAS_OPENCLIP:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.dim = self.model.text_projection.shape[-1]
        else:
            self.dim = 512

    def img_emb(self, pil_img):
        if _HAS_OPENCLIP:
            x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze(0).cpu().numpy()
        # fallback
        arr = np.asarray(pil_img).astype(np.float32).reshape(-1)
        # proj = self.rng.RandomState(arr.shape[0]).randn(arr.shape[0], self.dim)
        proj = self.rng.randn(arr.shape[0], self.dim)
        v = arr @ (proj / np.sqrt(arr.shape[0]))
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

    def txt_emb(self, text: str):
        if _HAS_OPENCLIP:
            tok = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze(0).cpu().numpy()
        rs = np.random.RandomState(abs(hash(text)) % (2**32))
        v = rs.randn(self.dim)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
