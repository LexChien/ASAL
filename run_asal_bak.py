import argparse, os, time, json, numpy as np
from fm.clip_embedder import VLM
from substrates.boids import Boids
from scores import supervised_target_score, openended_score, illumination_diversity
from search.optim import evo_search

def rollout_boids(theta, steps=400, size=256, vlm=None, capture_every=20):
    env = Boids()
    env.reset(theta)
    frames = []
    embs = []
    for t in range(steps):
        env.step()
        if (t+1) % capture_every == 0:
            img = env.render(size=size)
            frames.append(img)
            if vlm is not None:
                embs.append(vlm.img_emb(img))
    return frames, embs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['target','openended','illuminate'], required=True)
    ap.add_argument('--prompt', type=str, default="a biological cell")
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--keep', type=int, default=32)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    ts = time.strftime('%Y%m%d-%H%M%S')
    run_dir = args.out or f"runs/{ts}"
    os.makedirs(run_dir, exist_ok=True)
    # symlink 'latest'
    if os.path.islink('runs/latest'): os.unlink('runs/latest')
    os.makedirs('runs', exist_ok=True)
    try:
        os.symlink(run_dir, 'runs/latest')
    except Exception:
        pass

    vlm = VLM()
    txt = vlm.txt_emb(args.prompt)

    # θ bounds for Boids: [align, coh, sep, neigh_r, speed]
    low = np.array([0.0, 0.0, 0.0, 10.0, 1.0])
    high = np.array([2.0, 2.0, 2.0, 150.0, 8.0])
    init = [np.random.uniform(low, high) for _ in range(args.keep)]

    if args.mode == 'target':
        def eval_theta(theta):
            _, embs = rollout_boids(theta, steps=args.steps, vlm=vlm)
            return supervised_target_score(embs, txt)
        pool, scores, best, best_score = evo_search(init, eval_theta, iters=args.iters, pop=args.pop, keep=args.keep, sigma=0.2, bounds=(low, high))
        frames, embs = rollout_boids(best, steps=args.steps, vlm=vlm)
        frames[-1].save(os.path.join(run_dir, 'best.png'))
        json.dump({'mode':'target','best_theta':best.tolist(),'best_score':float(best_score)}, open(os.path.join(run_dir,'summary.json'),'w'), indent=2)

    elif args.mode == 'openended':
        def eval_theta(theta):
            _, embs = rollout_boids(theta, steps=args.steps, vlm=vlm, capture_every=max(1,args.steps//32))
            return openended_score(embs)
        pool, scores, best, best_score = evo_search(init, eval_theta, iters=args.iters, pop=args.pop, keep=args.keep, sigma=0.25, bounds=(low, high))
        frames, embs = rollout_boids(best, steps=args.steps, vlm=vlm)
        frames[-1].save(os.path.join(run_dir, 'best.png'))
        json.dump({'mode':'openended','best_theta':best.tolist(),'best_score':float(best_score)}, open(os.path.join(run_dir,'summary.json'),'w'), indent=2)

    else:  # illuminate
        # greedily optimize diversity; evaluation returns score after hypothetically adding candidate
        embs_final = []
        def eval_theta(theta):
            _, embs = rollout_boids(theta, steps=args.steps, vlm=vlm)
            cand = embs[-1]
            test_set = embs_final + [cand]
            return illumination_diversity(test_set)
        pool, scores, best, best_score = evo_search(init, eval_theta, iters=args.iters, pop=args.pop, keep=args.keep, sigma=0.25, bounds=(low, high))
        # materialize & save elites
        for i, th in enumerate(pool):
            frames, embs = rollout_boids(th, steps=args.steps, vlm=vlm)
            embs_final.append(embs[-1])
            frames[-1].save(os.path.join(run_dir, f"elite_{i:03d}.png"))
        json.dump({'mode':'illuminate','thetas':[t.tolist() for t in pool]}, open(os.path.join(run_dir,'summary.json'),'w'), indent=2)

    print("Done. Artifacts:", run_dir)

if __name__ == '__main__':
    main()
