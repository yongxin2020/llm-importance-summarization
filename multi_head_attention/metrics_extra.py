import math
from typing import Tuple
import numpy as np

EPS = 1e-12


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) / (np.asarray(p, dtype=float).sum() + EPS)
    q = np.asarray(q, dtype=float) / (np.asarray(q, dtype=float).sum() + EPS)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def renyi_divergence(p: np.ndarray, q: np.ndarray, alpha: float = 2.0) -> float:
    assert alpha > 0 and alpha != 1.0
    p = np.asarray(p, dtype=float) / (np.asarray(p, dtype=float).sum() + EPS)
    q = np.asarray(q, dtype=float) / (np.asarray(q, dtype=float).sum() + EPS)
    inner = np.sum((p ** alpha) * (q ** (1.0 - alpha)))
    if inner <= 0:
        return float('inf')
    return float((1.0 / (alpha - 1.0)) * np.log(inner))


def weighted_kl_by_power(p: np.ndarray, q: np.ndarray, gamma: float = 2.0) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_w = p ** gamma
    p_w = p_w / (p_w.sum() + EPS)
    return kl_div(p_w, q)


def kl_vs_uniform_normalized(p: np.ndarray, q_kl_vals: float) -> float:
    """
    Given a reference distribution p and an observed KL value KL(p||q) (q_kl_vals),
    compute normalized similarity vs uniform baseline:

      normalized = 1 - KL(p||q) / KL(p||U)

    Returns nan if KL(p||U) is ~0.
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    # KL(p || uniform) = -H(p) + log(n)  (natural log)
    p_norm = p / (p.sum() + EPS)
    entropy = -np.sum(p_norm * np.log(p_norm + EPS))
    kl_uniform = math.log(n) - entropy
    if kl_uniform < EPS:
        return float('nan')
    return 1.0 - float(q_kl_vals) / float(kl_uniform)


def precision_at_k_from_binary(pred_topk_idx: np.ndarray, ref_binary: np.ndarray) -> float:
    # pred_topk_idx: indices of predicted top-k tokens
    k = len(pred_topk_idx)
    return float(ref_binary[pred_topk_idx].sum()) / float(k) if k > 0 else 0.0


def recall_at_k_from_binary(pred_topk_idx: np.ndarray, ref_binary: np.ndarray) -> float:
    # pred_topk_idx: indices of predicted top-k tokens
    total_relevant = float(ref_binary.sum())
    if total_relevant == 0:
        return 0.0
    return float(ref_binary[pred_topk_idx].sum()) / total_relevant


def ndcg_at_k(pred_scores: np.ndarray, ref_gain: np.ndarray, k: int) -> float:
    order = np.argsort(-pred_scores)[:k]
    gains = ref_gain[order]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** gains - 1) / discounts)
    ideal = np.sort(ref_gain)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def bootstrap_mean_and_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_means.append(np.mean(values[idx]))
    lo = np.percentile(boot_means, (1.0 - ci) / 2.0 * 100)
    hi = np.percentile(boot_means, (1.0 + ci) / 2.0 * 100)
    return float(np.mean(values)), float(lo), float(hi)
