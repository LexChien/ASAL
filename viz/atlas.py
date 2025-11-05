import argparse, os, glob, numpy as np, umap
from PIL import Image
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.run, 'elite_*.png')))
    if not paths:
        print("No elites found.")
        return
    arr = []
    for p in paths:
        im = Image.open(p).resize((96,96))
        arr.append(np.asarray(im).reshape(-1).astype(np.float32)/255.0)
    X = np.stack(arr)
    Z = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1], s=6)
    plt.title('Simulation Atlas (pixel UMAP, demo)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.run, 'atlas.png'), dpi=160)
    print("Saved", os.path.join(args.run, 'atlas.png'))

if __name__ == '__main__':
    main()
