"""
校准性评估指标模块。

提供四个核心指标:
1. Spearman Rank Correlation (ρ)
2. Expected Calibration Error (ECE)
3. Jensen-Shannon Divergence (JS)
4. Weighted Recall@K (WR@K)

以及辅助分析:
- Reliability Diagram 数据
- 失败模式分类
"""

import numpy as np
from scipy import stats


def spearman_correlation(p_human: dict, p_llm: dict) -> float:
    """
    计算单个 prompt 的 Spearman rank correlation。

    Args:
        p_human: {translation_text: weight, ...} 人类翻译分布
        p_llm:   {translation_text: score, ...}  LLM 概率/分数

    Returns:
        Spearman ρ (float), 若翻译数 < 3 则返回 NaN
    """
    # 取两个分布的交集
    common = set(p_human.keys()) & set(p_llm.keys())
    if len(common) < 3:
        return float("nan")

    human_vals = [p_human[t] for t in common]
    llm_vals = [p_llm[t] for t in common]

    rho, _ = stats.spearmanr(human_vals, llm_vals)
    return rho


def expected_calibration_error(
    all_p_human: list[float],
    all_p_llm: list[float],
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    计算 ECE 并返回各 bin 的详情（用于 reliability diagram）。

    Args:
        all_p_human: 所有 (prompt, translation) 对的人类权重
        all_p_llm:   对应的 LLM 概率
        n_bins: bin 的数量

    Returns:
        (ece_value, bin_details)
        bin_details: list of {avg_confidence, avg_frequency, count, bin_lower, bin_upper}
    """
    p_human = np.array(all_p_human)
    p_llm = np.array(all_p_llm)
    n = len(p_human)

    # 等频分 bin（按 LLM 概率排序）
    sorted_idx = np.argsort(p_llm)
    bin_size = n // n_bins

    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        idx = sorted_idx[start:end]

        avg_conf = np.mean(p_llm[idx])
        avg_freq = np.mean(p_human[idx])
        count = len(idx)

        ece += (count / n) * abs(avg_conf - avg_freq)
        bin_details.append({
            "avg_confidence": float(avg_conf),
            "avg_frequency": float(avg_freq),
            "count": int(count),
            "bin_lower": float(p_llm[idx].min()),
            "bin_upper": float(p_llm[idx].max()),
        })

    return float(ece), bin_details


def js_divergence(p_human: dict, p_llm: dict) -> float:
    """
    计算单个 prompt 的 Jensen-Shannon 散度。

    两个分布需定义在同一支撑集上。对于只在一个分布中出现的翻译，
    另一个分布赋予 0 概率，然后加 epsilon 平滑。

    Args:
        p_human: {translation_text: weight}
        p_llm:   {translation_text: probability}

    Returns:
        JS divergence ∈ [0, 1] (使用 log base 2)
    """
    # 合并支撑集
    all_keys = set(p_human.keys()) | set(p_llm.keys())
    if not all_keys:
        return 0.0

    eps = 1e-12

    # 构建对齐的概率向量
    h = np.array([p_human.get(k, 0) for k in all_keys], dtype=np.float64)
    l = np.array([p_llm.get(k, 0) for k in all_keys], dtype=np.float64)

    # 归一化（确保求和为 1）
    h = h / (h.sum() + eps)
    l = l / (l.sum() + eps)

    # M = 0.5 * (P + Q)
    m = 0.5 * (h + l)

    # D_KL(P || M) + D_KL(Q || M)
    def kl(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log2(p[mask] / (q[mask] + eps)))

    return float(0.5 * kl(h, m) + 0.5 * kl(l, m))


def weighted_recall_at_k(p_human: dict, p_llm: dict, k: int) -> float:
    """
    计算 Weighted Recall@K。

    LLM 概率最高的 K 个翻译在人类分布中覆盖了多少权重。

    Args:
        p_human: {translation: weight}
        p_llm:   {translation: probability}
        k: Top-K

    Returns:
        WR@K ∈ [0, 1]
    """
    # LLM 排序取 Top-K
    top_k = sorted(p_llm.keys(), key=lambda t: p_llm[t], reverse=True)[:k]

    # 这些翻译在人类分布中的累计权重
    return sum(p_human.get(t, 0) for t in top_k)


def classify_failure_mode(p_human: dict, p_llm: dict, rho: float) -> str:
    """
    对单个 prompt 的校准失败进行分类。

    失败模式:
    - "well_calibrated": 校准良好
    - "overconfident": LLM 分布过于集中
    - "underconfident": LLM 分布过于分散
    - "rank_misaligned": 集中度相似但排序不一致
    - "phantom": LLM 大量概率在 gold set 之外 (仅 Sampling 模式)

    Args:
        p_human: {translation: weight}
        p_llm:   {translation: probability}
        rho: 已计算的 Spearman ρ

    Returns:
        failure mode string
    """
    eps = 1e-12

    # 计算两个分布的熵
    h_vals = np.array(list(p_human.values()))
    h_vals = h_vals / (h_vals.sum() + eps)
    entropy_human = -np.sum(h_vals * np.log(h_vals + eps))

    l_vals = np.array(list(p_llm.values()))
    l_vals = l_vals / (l_vals.sum() + eps)
    entropy_llm = -np.sum(l_vals * np.log(l_vals + eps))

    entropy_diff = entropy_llm - entropy_human

    # gold set 外的概率质量
    gold_keys = set(p_human.keys())
    phantom_mass = sum(v for k, v in p_llm.items() if k not in gold_keys)

    if phantom_mass > 0.2:
        return "phantom"
    elif entropy_diff < -1.0:
        return "overconfident"
    elif entropy_diff > 1.0:
        return "underconfident"
    elif not np.isnan(rho) and rho < 0.5 and abs(entropy_diff) <= 1.0:
        return "rank_misaligned"
    else:
        return "well_calibrated"


def evaluate_prompt(p_human: dict, p_llm: dict, ks=(1, 5, 10, 20)) -> dict:
    """
    对单个 prompt 计算所有指标。

    Args:
        p_human: {translation: weight}
        p_llm:   {translation: probability}
        ks: WR@K 的 K 值列表

    Returns:
        dict with all metrics
    """
    rho = spearman_correlation(p_human, p_llm)
    js = js_divergence(p_human, p_llm)
    wr = {f"wr@{k}": weighted_recall_at_k(p_human, p_llm, k) for k in ks}
    mode = classify_failure_mode(p_human, p_llm, rho)

    return {
        "spearman_rho": rho,
        "js_divergence": js,
        **wr,
        "failure_mode": mode,
    }


def evaluate_dataset(prompts_human: list[dict], prompts_llm: list[dict]) -> dict:
    """
    对整个数据集（多个 prompt）计算汇总指标。

    Args:
        prompts_human: list of {translation: weight} per prompt
        prompts_llm:   list of {translation: probability} per prompt

    Returns:
        dict with aggregated metrics
    """
    all_results = []
    all_p_human_flat = []
    all_p_llm_flat = []

    for ph, pl in zip(prompts_human, prompts_llm):
        result = evaluate_prompt(ph, pl)
        all_results.append(result)

        # 为 ECE 收集展平的概率对
        common = set(ph.keys()) & set(pl.keys())
        for t in common:
            all_p_human_flat.append(ph[t])
            all_p_llm_flat.append(pl[t])

    # 汇总
    rhos = [r["spearman_rho"] for r in all_results if not np.isnan(r["spearman_rho"])]
    jss = [r["js_divergence"] for r in all_results]

    ece, ece_bins = expected_calibration_error(all_p_human_flat, all_p_llm_flat)

    # 失败模式统计
    modes = [r["failure_mode"] for r in all_results]
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    ks = [1, 5, 10, 20]
    wr_means = {}
    for k in ks:
        key = f"wr@{k}"
        wr_means[key] = float(np.mean([r[key] for r in all_results]))

    return {
        "n_prompts": len(all_results),
        "spearman_rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "spearman_rho_std": float(np.std(rhos)) if rhos else float("nan"),
        "js_divergence_mean": float(np.mean(jss)),
        "js_divergence_std": float(np.std(jss)),
        "ece": ece,
        "ece_bins": ece_bins,
        **wr_means,
        "failure_modes": mode_counts,
        "per_prompt": all_results,
    }
