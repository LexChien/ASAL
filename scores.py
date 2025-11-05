import numpy as np
from fm.clip_embedder import cos_sim

def supervised_target_score(img_embs, txt_emb):
    """Eq.(2): maximize E_T <VLM_img(R^S_T(θ)), VLM_txt(prompt_T)>
    Here we use the final frame only (single prompt).
    Returns: higher is better.
    """
    return cos_sim(img_embs[-1], txt_emb)

def openended_score(img_embs):
    """Eq.(3): minimize historical nearest-neighbor similarity.
    We implement: score = - mean_T max_{T'<T} <emb_T, emb_T'>.
    Higher is better after negation.
    """
    sims = []
    for t in range(1, len(img_embs)):
        cur = img_embs[t]
        past = img_embs[:t]
        nn = max(float(np.dot(cur, p) / (np.linalg.norm(cur)*np.linalg.norm(p)+1e-9)) for p in past)
        sims.append(nn)
    return -float(np.mean(sims) if sims else 0.0)

def illumination_diversity(img_embs_list):
    """Eq.(4): set-level diversity via nearest-neighbor similarity.
    score = - mean over elements of (max similarity to any other).
    Higher is better after negation.
    """
    if len(img_embs_list) <= 1:
        return 0.0
    arr = np.stack(img_embs_list)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    sims = arr @ arr.T
    np.fill_diagonal(sims, -1.0)
    nn = sims.max(axis=1)
    return -float(nn.mean())
