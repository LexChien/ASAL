import numpy as np
from PIL import Image, ImageDraw

class Boids:
    """Simplified Boids substrate parameterized by θ:
    θ = [align_w, coh_w, sep_w, neigh_r, speed].
    Implements Init_θ, Step_θ, Render_θ.
    """
    def __init__(self, n=128, world=512, seed=0):
        self.n = n
        self.world = world
        self.rng = np.random.RandomState(seed)
        self.pos = None
        self.vel = None

    def reset(self, theta):
        self.theta = np.array(theta, dtype=float)
        self.align_w, self.coh_w, self.sep_w, self.neigh_r, self.speed = self.theta
        self.pos = self.rng.rand(self.n, 2) * self.world
        ang = self.rng.rand(self.n) * 2*np.pi
        self.vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * self.speed

    def step(self):
        pos, vel = self.pos, self.vel
        n = self.n
        diff = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(diff, axis=-1) + 1e-9
        mask = (dist < self.neigh_r) & (dist > 0)
        align = (vel[None, :, :] * mask[..., None]).sum(1)
        neigh_pos = (pos[None, :, :] * mask[..., None]).sum(1)
        cnt = mask.sum(1, keepdims=True) + 1e-9
        align = align / cnt
        target = neigh_pos / cnt
        coh = (target - pos)
        sep = (-diff / dist[..., None]) * mask[..., None]
        sep = sep.sum(1)
        acc = self.align_w * align + self.coh_w * coh + self.sep_w * sep
        vel = vel + 0.05 * acc
        speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
        vel = vel / speed * self.speed
        pos = (pos + vel) % self.world
        self.pos, self.vel = pos, vel

    def render(self, size=256):
        img = Image.new("RGB", (size, size), (0,0,0))
        draw = ImageDraw.Draw(img)
        scale = size / self.world
        for p in self.pos:
            x, y = p * scale
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(200, 255, 180))
        return img
