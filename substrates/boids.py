import numpy as np
import torch
from PIL import Image


class Boids:
    """Simplified Boids substrate parameterized by theta.

    theta = [align_w, coh_w, sep_w, neigh_r, speed]
    """

    def __init__(self, n=128, world=512, seed=0, device=None):
        self.n = n
        self.world = world
        self.seed = seed
        self.device = torch.device(device) if device is not None else None
        self.use_torch = self.device is not None and self.device.type != 'cpu'
        self.rng = np.random.RandomState(seed)
        self.torch_rng = None
        if self.use_torch:
            self.torch_rng = torch.Generator(device=self.device)
            self.torch_rng.manual_seed(seed)
        self.pos = None
        self.vel = None

    def reset(self, theta):
        if self.use_torch:
            theta_t = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
            self.theta = theta_t
            self.align_w, self.coh_w, self.sep_w, self.neigh_r, self.speed = theta_t
            self.pos = torch.rand((self.n, 2), generator=self.torch_rng, device=self.device) * self.world
            ang = torch.rand((self.n,), generator=self.torch_rng, device=self.device) * (2 * np.pi)
            self.vel = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1) * self.speed
            return

        self.theta = np.array(theta, dtype=float)
        self.align_w, self.coh_w, self.sep_w, self.neigh_r, self.speed = self.theta
        self.pos = self.rng.rand(self.n, 2) * self.world
        ang = self.rng.rand(self.n) * 2 * np.pi
        self.vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * self.speed

    def step(self):
        if self.use_torch:
            pos, vel = self.pos, self.vel
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.linalg.vector_norm(diff, dim=-1) + 1e-9
            mask = (dist < self.neigh_r) & (dist > 0)
            mask_f = mask.unsqueeze(-1).to(pos.dtype)
            align = (vel.unsqueeze(0) * mask_f).sum(dim=1)
            neigh_pos = (pos.unsqueeze(0) * mask_f).sum(dim=1)
            cnt = mask.sum(dim=1, keepdim=True).to(pos.dtype) + 1e-9
            align = align / cnt
            target = neigh_pos / cnt
            coh = target - pos
            sep = (-diff / dist.unsqueeze(-1)) * mask_f
            sep = sep.sum(dim=1)
            acc = self.align_w * align + self.coh_w * coh + self.sep_w * sep
            vel = vel + 0.05 * acc
            speed = torch.linalg.vector_norm(vel, dim=1, keepdim=True) + 1e-9
            vel = vel / speed * self.speed
            pos = torch.remainder(pos + vel, self.world)
            self.pos, self.vel = pos, vel
            return

        pos, vel = self.pos, self.vel
        diff = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(diff, axis=-1) + 1e-9
        mask = (dist < self.neigh_r) & (dist > 0)
        align = (vel[None, :, :] * mask[..., None]).sum(1)
        neigh_pos = (pos[None, :, :] * mask[..., None]).sum(1)
        cnt = mask.sum(1, keepdims=True) + 1e-9
        align = align / cnt
        target = neigh_pos / cnt
        coh = target - pos
        sep = (-diff / dist[..., None]) * mask[..., None]
        sep = sep.sum(1)
        acc = self.align_w * align + self.coh_w * coh + self.sep_w * sep
        vel = vel + 0.05 * acc
        speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
        vel = vel / speed * self.speed
        pos = (pos + vel) % self.world
        self.pos, self.vel = pos, vel

    def render_tensor(self, size=256, radius=2):
        if self.use_torch:
            img = torch.zeros((3, size, size), dtype=torch.uint8, device=self.device)
            scale = size / self.world
            coords = torch.round(self.pos * scale).to(torch.long)
            coords = coords.clamp(0, size - 1)
            color = torch.tensor([200, 255, 180], dtype=torch.uint8, device=self.device).view(3, 1)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    xs = (coords[:, 0] + dx).clamp(0, size - 1)
                    ys = (coords[:, 1] + dy).clamp(0, size - 1)
                    img[:, ys, xs] = color
            return img

        img = np.zeros((size, size, 3), dtype=np.uint8)
        scale = size / self.world
        coords = np.rint(self.pos * scale).astype(int)
        coords = np.clip(coords, 0, size - 1)
        color = np.array([200, 255, 180], dtype=np.uint8)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                xs = np.clip(coords[:, 0] + dx, 0, size - 1)
                ys = np.clip(coords[:, 1] + dy, 0, size - 1)
                img[ys, xs] = color
        return torch.from_numpy(img).permute(2, 0, 1)

    def render(self, size=256):
        frame = self.render_tensor(size=size)
        arr = frame.permute(1, 2, 0).detach().cpu().numpy()
        return Image.fromarray(arr, mode="RGB")
