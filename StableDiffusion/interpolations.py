import numpy as np
import torch

def interpolate(emb1, emb2, steps):
    interpolated = []
    for alpha in np.linspace(0, 1, steps):
        interpolated.append((1 - alpha) * emb1 + alpha * emb2)
    return interpolated


def interpolate_slerp(emb1, emb2, steps):
    interpolated = []

    flat_emb1 = emb1.reshape(-1)
    flat_emb2 = emb2.reshape(-1)

    flat_emb1_norm = flat_emb1 / torch.norm(flat_emb1)
    flat_emb2_norm = flat_emb2 / torch.norm(flat_emb2)

    dot_product = torch.dot(flat_emb1_norm, flat_emb2_norm)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)

    if omega.abs() < 1e-6:
        return interpolate(emb1, emb2, steps)

    sin_omega = torch.sin(omega)

    for alpha in np.linspace(0, 1, steps):
        t = torch.tensor(alpha, dtype=emb1.dtype, device=emb1.device)

        coef1 = torch.sin((1.0 - t) * omega) / sin_omega
        coef2 = torch.sin(t * omega) / sin_omega

        interpolated_flat = coef1 * flat_emb1 + coef2 * flat_emb2

        interpolated_emb = interpolated_flat.reshape(emb1.shape)
        interpolated.append(interpolated_emb)

    return interpolated


def interpolate_nao(emb1, emb2, steps, iterations=50, lr=0.01):
    interpolated = []

    path = torch.stack(interpolate_slerp(emb1, emb2, steps))
    path.requires_grad = True

    optimizer = torch.optim.Adam([path], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()

        norms = torch.norm(path.reshape(steps, -1), dim=1)
        expected_norm = torch.norm(emb1.reshape(-1))

        loss = torch.mean((norms - expected_norm) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            path[0] = emb1
            path[-1] = emb2

    return [path[i].detach() for i in range(steps)]


def interpolate_cog(emb1, emb2, steps):
    interpolated = []

    for alpha in np.linspace(0, 1, steps):
        t = alpha
        weight1 = (1 - t) / np.sqrt((1 - t) ** 2 + t ** 2)
        weight2 = t / np.sqrt((1 - t) ** 2 + t ** 2)

        interpolated_emb = weight1 * emb1 + weight2 * emb2
        interpolated.append(interpolated_emb)

    return interpolated


def interpolate_noisediffusion(emb1, emb2, steps, noise_level=0.1):
    slerp_results = interpolate_slerp(emb1, emb2, steps)

    interpolated = []
    for slerp_result in slerp_results:
        noise = torch.randn_like(slerp_result) * noise_level
        corrected = slerp_result + noise
        interpolated.append(corrected)

    return interpolated
