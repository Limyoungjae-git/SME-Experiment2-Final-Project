import numpy as np
import scipy.io as sio


EPS = 1e-9

BS_MEDIAN_BIAS = np.array([
    11.7416294512,
    10.1781651690,
    8.2776546211,
    7.9736333316,
    11.6767276727,
    12.5691952925,
    11.8438272381,
    9.9698411010,
    6.3689016900,
    7.7598806119,
    8.9424216015,
    11.5779724611,
    11.1583258124,
    10.8351322987,
    9.7396700481,
    7.9632185948,
    8.7562389503,
    13.1424176581
], dtype=float)

BIAS_ALPHA = 0.5


def apply_bias_correction(d):
    d = np.asarray(d, dtype=float).reshape(-1)

    if d.shape[0] == BS_MEDIAN_BIAS.shape[0]:
        d_corr = d - BIAS_ALPHA * BS_MEDIAN_BIAS
    else:
        d_corr = d.copy()

    return np.maximum(d_corr, EPS)


def linear_multilateration(d, p_bs, indices):
    indices = np.asarray(indices, dtype=int)

    if len(indices) < 3:
        return np.mean(p_bs[:, indices], axis=1)

    ref = indices[np.argmin(d[indices])]
    b_ref = p_bs[:, ref]
    d_ref = d[ref]

    A = []
    y = []

    for i in indices:
        if i == ref:
            continue

        b_i = p_bs[:, i]
        d_i = d[i]

        A.append(2.0 * (b_i - b_ref))
        y.append(
            d_ref ** 2
            - d_i ** 2
            + np.dot(b_i, b_i)
            - np.dot(b_ref, b_ref)
        )

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)

    try:
        p_est, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        p_est = np.mean(p_bs[:, indices], axis=1)

    return np.asarray(p_est, dtype=float)


def robust_refine(p0, d, p_bs, indices, n_iter=8):
    p = np.asarray(p0, dtype=float).copy()
    indices = np.asarray(indices, dtype=int)

    if len(indices) < 3:
        return p

    for _ in range(n_iter):
        bs = p_bs[:, indices].T
        diff = p.reshape(1, 2) - bs
        dist = np.linalg.norm(diff, axis=1) + EPS

        r = dist - d[indices]
        J = diff / dist.reshape(-1, 1)

        med = np.median(r)
        mad = np.median(np.abs(r - med)) + EPS
        scale = 1.4826 * mad + 0.5

        w = 1.0 / (1.0 + (r / (2.0 * scale)) ** 2)
        sw = np.sqrt(w + EPS)

        try:
            delta, _, _, _ = np.linalg.lstsq(
                J * sw.reshape(-1, 1),
                -r * sw,
                rcond=None
            )
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(delta)):
            break

        p = p + delta

        if not np.all(np.isfinite(p)):
            break

        p = np.clip(p, -100.0, 100.0)

        if np.linalg.norm(delta) < 1e-6:
            break

    return p


def estimate_position(d, p_bs, indices):
    p0 = linear_multilateration(d, p_bs, indices)
    p_est = robust_refine(p0, d, p_bs, indices, n_iter=8)
    return p_est


def your_algorithm(d_one_user, p_bs):
    d_raw = np.asarray(d_one_user, dtype=float).reshape(-1)
    p_bs = np.asarray(p_bs, dtype=float)

    num_bs = p_bs.shape[1]

    finite = np.isfinite(d_raw)
    if np.any(finite):
        fill_value = np.nanmedian(d_raw[finite])
        d_raw = np.where(finite, d_raw, fill_value)
    else:
        d_raw = np.ones(num_bs, dtype=float)

    d_raw = np.maximum(d_raw, EPS)

    d = apply_bias_correction(d_raw)

    K = min(5, num_bs)
    base_indices = np.argsort(d)[:K]

    candidates = []

    p_base = estimate_position(d, p_bs, base_indices)
    candidates.append(p_base)

    for leave_out in base_indices:
        subset = base_indices[base_indices != leave_out]
        if len(subset) >= 3:
            p_candidate = estimate_position(d, p_bs, subset)
            candidates.append(p_candidate)

    candidates = np.asarray(candidates, dtype=float)
    candidates = candidates[np.all(np.isfinite(candidates), axis=1)]

    if candidates.shape[0] == 0:
        return np.mean(p_bs, axis=1)

    p_init = np.median(candidates, axis=0)

    residual = np.abs(np.linalg.norm(p_init.reshape(2, 1) - p_bs, axis=0) - d)

    num_refine_bs = min(max(8, K), num_bs)
    refine_indices = np.argsort(residual)[:num_refine_bs]

    p_final = robust_refine(p_init, d, p_bs, refine_indices, n_iter=8)

    if not np.all(np.isfinite(p_final)):
        p_final = p_init

    return p_final


def main():
    mat_path = 'DH_FR1.mat'

    data = sio.loadmat(mat_path, squeeze_me=False)

    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    elif 'BS_positions' in data:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
    else:
        raise KeyError("Base station position variable not found.")

    d_hat = np.asarray(data['d_hat'], dtype=float)

    if 'p' in data:
        p = np.asarray(data['p'], dtype=float)
    elif 'UE_positions' in data:
        p = np.asarray(data['UE_positions'], dtype=float)
    else:
        p = None

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    return p_hat


if __name__ == "__main__":
    main()
