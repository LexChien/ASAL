import argparse, os, sys, time, json, numpy as np
from PIL import Image

import imageio
try:
    import imageio_ffmpeg  # noqa: F401
    _HAS_FFMPEG = True
except Exception:
    _HAS_FFMPEG = False


def _ensure_project_python():
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, '.venv', 'bin', 'python')
    cusparselt_lib = os.path.join(
        project_root,
        '.venv',
        'lib',
        'python3.10',
        'site-packages',
        'nvidia',
        'cusparselt',
        'lib',
    )
    if os.path.abspath(sys.executable) == os.path.abspath(venv_python):
        return
    if not os.path.exists(venv_python):
        return
    try:
        import torch  # noqa: F401
        return
    except Exception:
        pass
    env = os.environ.copy()
    ld_library_path = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f"{cusparselt_lib}:{ld_library_path}" if ld_library_path else cusparselt_lib
    os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_project_python()

from fm.clip_embedder import VLM
from substrates.boids import Boids
from scores import supervised_target_score, openended_score, illumination_diversity
from search.optim import evo_search

def rollout_boids(theta, steps=400, size=256, vlm=None, capture_every=10, seed=0, frame_mode='captured', embed_mode='captured'):
    """Run a Boids simulation and optionally collect rendered frames / embeddings."""
    env = Boids(seed=seed, device=getattr(vlm, 'device', None))
    env.reset(theta)
    frames, render_batch = [], []
    for t in range(steps):
        env.step()
        is_capture = (t+1) % max(1, capture_every) == 0 or t == steps-1
        need_frame = (frame_mode == 'captured' and is_capture) or (frame_mode == 'last' and t == steps-1)
        need_emb = vlm is not None and (
            (embed_mode == 'captured' and is_capture) or
            (embed_mode == 'last' and t == steps-1)
        )
        if not (need_frame or need_emb):
            continue
        frame_tensor = env.render_tensor(size=size)
        if need_frame:
            frames.append(__frame_to_pil(frame_tensor))
        if need_emb:
            render_batch.append(frame_tensor)
    embs = vlm.img_emb_batch(render_batch) if vlm is not None and render_batch else []
    return frames, embs


def __frame_to_pil(frame_tensor):
    arr = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')

def save_animation(frames, out_dir, stem="best", fps=20):
    arr = [np.array(f) for f in frames]
    gif_path = os.path.join(out_dir, f"{stem}.gif")
    imageio.mimsave(gif_path, arr, duration=1.0/max(1,fps))

    mp4_path = None
    if _HAS_FFMPEG:
        mp4_path = os.path.join(out_dir, f"{stem}.mp4")
        writer = imageio.get_writer(mp4_path, fps=fps, codec='h264', quality=8)
        for a in arr:
            writer.append_data(a)
        writer.close()
    return gif_path, mp4_path

def main():
    run_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['target','openended','illuminate'], required=True)
    ap.add_argument('--prompt', type=str, default='a biological cell')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--keep', type=int, default=32)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0, help='global seed for reproducibility')
    args = ap.parse_args()

    # out dir + latest symlink
    ts = time.strftime('%Y%m%d-%H%M%S')
    run_dir = args.out or f'runs/{ts}'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    try:
        if os.path.islink('runs/latest') or os.path.exists('runs/latest'):
            try: os.unlink('runs/latest')
            except Exception: pass
        os.symlink(os.path.abspath(run_dir), 'runs/latest')
    except Exception:
        pass

    np.random.seed(args.seed)
    vlm = VLM()
    txt = vlm.txt_emb(args.prompt)

    # θ bounds: [align, cohesion, separation, neighbor_radius, speed]
    low  = np.array([0.0, 0.0, 0.0, 10.0, 1.0])
    high = np.array([2.0, 2.0, 2.0, 150.0, 8.0])
    init = [np.random.uniform(low, high) for _ in range(args.keep)]

    print(f"[ASAL] mode={args.mode} prompt=\"{args.prompt}\" steps={args.steps} iters={args.iters} pop={args.pop} keep={args.keep}")
    print(f"[ASAL] VLM device: {'CUDA' if vlm.device=='cuda' else 'CPU'} | seed={args.seed}")

    cap_every = max(1, args.steps // 32)

    if args.mode == 'target':
        def eval_theta(theta):
            _, embs = rollout_boids(
                theta,
                steps=args.steps,
                vlm=vlm,
                capture_every=cap_every,
                seed=args.seed,
                frame_mode='none',
                embed_mode='last',
            )
            return supervised_target_score(embs, txt)

        pool, scores, best, best_score = evo_search(
            init, eval_theta, iters=args.iters, pop=args.pop, keep=args.keep, sigma=0.2, bounds=(low, high)
        )
        print(f"[ASAL] best_score={best_score:.4f}")
        frames, _ = rollout_boids(
            best,
            steps=args.steps,
            vlm=vlm,
            capture_every=cap_every,
            seed=args.seed,
            frame_mode='captured',
            embed_mode='none',
        )
        frames[-1].save(os.path.join(run_dir, 'best.png'))
        gif_path, mp4_path = save_animation(frames, run_dir, 'best', fps=args.fps)
        json.dump({'mode':'target','best_theta':best.tolist(),'best_score':float(best_score),
                   'gif':os.path.basename(gif_path),'mp4':os.path.basename(mp4_path) if mp4_path else None,
                   'run_elapsed_seconds': float(time.time() - run_start)},
                  open(os.path.join(run_dir,'summary.json'),'w'), indent=2)

    elif args.mode == 'openended':
        def eval_theta(theta):
            _, embs = rollout_boids(
                theta,
                steps=args.steps,
                vlm=vlm,
                capture_every=cap_every,
                seed=args.seed,
                frame_mode='none',
                embed_mode='captured',
            )
            return openended_score(embs)

        pool, scores, best, best_score = evo_search(
            init, eval_theta, iters=args.iters, pop=args.pop, keep=args.keep, sigma=0.25, bounds=(low, high)
        )
        print(f"[ASAL] best_score={best_score:.4f}")
        frames, _ = rollout_boids(
            best,
            steps=args.steps,
            vlm=vlm,
            capture_every=cap_every,
            seed=args.seed,
            frame_mode='captured',
            embed_mode='none',
        )
        frames[-1].save(os.path.join(run_dir, 'best.png'))
        gif_path, mp4_path = save_animation(frames, run_dir, 'best', fps=args.fps)
        json.dump({'mode':'openended','best_theta':best.tolist(),'best_score':float(best_score),
                   'gif':os.path.basename(gif_path),'mp4':os.path.basename(mp4_path) if mp4_path else None,
                   'run_elapsed_seconds': float(time.time() - run_start)},
                  open(os.path.join(run_dir,'summary.json'),'w'), indent=2)

    else:  # illuminate
        embs_final = []
        def eval_theta(theta):
            _, embs = rollout_boids(
                theta,
                steps=args.steps,
                vlm=vlm,
                capture_every=cap_every,
                seed=args.seed,
                frame_mode='none',
                embed_mode='captured',
            )
            cand = embs[-1]
            test_set = embs_final + [cand]
            return illumination_diversity(test_set)

        pool, scores, best, best_score = evo_search(
            init, eval_theta, iters=args.iters, pop=args.pop, keep=args.keep, sigma=0.25, bounds=(low, high)
        )
        # persist elites
        for i, th in enumerate(pool):
            frames, embs = rollout_boids(
                th,
                steps=args.steps,
                vlm=vlm,
                capture_every=cap_every,
                seed=args.seed,
                frame_mode='last',
                embed_mode='last',
            )
            embs_final.append(embs[-1])
            frames[-1].save(os.path.join(run_dir, f"elite_{i:03d}.png"))
        json.dump({'mode':'illuminate','thetas':[t.tolist() for t in pool],
                   'run_elapsed_seconds': float(time.time() - run_start)},
                  open(os.path.join(run_dir,'summary.json'),'w'), indent=2)

    print('Done. Artifacts:', run_dir)

if __name__ == '__main__':
    main()
