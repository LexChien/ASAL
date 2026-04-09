import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
            self.image_size = getattr(self.model.visual, 'image_size', (224, 224))
            if isinstance(self.image_size, int):
                self.image_size = (self.image_size, self.image_size)
            norm = next(
                (t for t in getattr(self.preprocess, 'transforms', []) if hasattr(t, 'mean') and hasattr(t, 'std')),
                None,
            )
            if norm is not None:
                self.image_mean = torch.tensor(norm.mean, dtype=torch.float32).view(1, 3, 1, 1)
                self.image_std = torch.tensor(norm.std, dtype=torch.float32).view(1, 3, 1, 1)
            else:
                self.image_mean = None
                self.image_std = None
        else:
            self.dim = 512

    def _prepare_tensor_batch(self, images):
        if isinstance(images, torch.Tensor):
            batch = images
        else:
            items = list(images)
            if not items:
                return torch.empty((0, 3, *self.image_size), dtype=torch.float32, device=self.device)
            first = items[0]
            if isinstance(first, Image.Image):
                batch = torch.stack([self.preprocess(img) for img in items], dim=0)
                return batch.to(self.device)
            tensors = []
            for item in items:
                if isinstance(item, Image.Image):
                    tensors.append(self.preprocess(item))
                    continue
                tensor = torch.as_tensor(item)
                if tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1)
                tensors.append(tensor)
            batch = torch.stack(tensors, dim=0)

        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        if batch.shape[-1] in (1, 3) and batch.shape[1] not in (1, 3):
            batch = batch.permute(0, 3, 1, 2)
        batch = batch.to(self.device, dtype=torch.float32)
        if batch.max().item() > 1.0:
            batch = batch / 255.0
        batch = F.interpolate(
            batch,
            size=self.image_size,
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        if self.image_mean is not None and self.image_std is not None:
            mean = self.image_mean.to(batch.device)
            std = self.image_std.to(batch.device)
            batch = (batch - mean) / std
        return batch

    def img_emb_batch(self, images):
        if _HAS_OPENCLIP:
            x = self._prepare_tensor_batch(images)
            with torch.inference_mode():
                feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return [feat.cpu().numpy() for feat in feats]
        embs = []
        for image in images if not isinstance(images, torch.Tensor) or images.ndim == 4 else [images]:
            arr = np.asarray(image).astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            arr = arr.reshape(-1)
            proj = self.rng.randn(arr.shape[0], self.dim)
            v = arr @ (proj / np.sqrt(arr.shape[0]))
            v = v / (np.linalg.norm(v) + 1e-9)
            embs.append(v)
        return embs

    def img_emb(self, image):
        return self.img_emb_batch([image])[0]

    def txt_emb(self, text: str):
        if _HAS_OPENCLIP:
            tok = self.tokenizer([text]).to(self.device)
            with torch.inference_mode():
                feats = self.model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze(0).cpu().numpy()
        rs = np.random.RandomState(abs(hash(text)) % (2**32))
        v = rs.randn(self.dim)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
