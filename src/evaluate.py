"""评估脚本: 读取 LLM 结果 + STAPLE gold 数据，计算校准性指标并生成图表

python src/evaluate.py --result results/sampling_gpt-4o_en_pt_dev_t1.0_n100.json --lang en_pt --split dev
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import evaluate_prompt, expected_calibration_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_gold(lang, split):
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        return json.load(f)


def load_result(result_path):
    with open(result_path, "r") as f:
        return json.load(f)


def run_evaluation(result_path, lang, split, label=None):
    """对一组 LLM 结果运行完整评估"""
    gold_data = load_gold(lang, split)
    llm_results = load_result(result_path)

    if label is None:
        label = Path(result_path).stem

    gold_map = {}
    for p in gold_data:
        gold_map[p["prompt_id"]] = {t["text"]: t["weight"] for t in p["translations"]}

    per_prompt_results = []
    all_p_human_flat = []
    all_p_llm_flat = []

    for r in llm_results:
        pid = r["prompt_id"]
        p_llm = r["p_llm"]
        p_human = gold_map.get(pid, {})

        if not p_human or not p_llm:
            continue

        result = evaluate_prompt(p_human, p_llm)
        result["prompt_id"] = pid
        result["english"] = r["english"]
        per_prompt_results.append(result)

        # ECE 用的展平数据
        common = set(p_human.keys()) & set(p_llm.keys())
        for t in common:
            all_p_human_flat.append(p_human[t])
            all_p_llm_flat.append(p_llm[t])

    rhos = [r["spearman_rho"] for r in per_prompt_results if not np.isnan(r["spearman_rho"])]
    jss = [r["js_divergence"] for r in per_prompt_results]
    wr1 = [r["wr@1"] for r in per_prompt_results]
    wr5 = [r["wr@5"] for r in per_prompt_results]
    wr10 = [r["wr@10"] for r in per_prompt_results]
    wr20 = [r["wr@20"] for r in per_prompt_results]

    ece = 0.0
    ece_bins = []
    if all_p_human_flat:
        ece, ece_bins = expected_calibration_error(all_p_human_flat, all_p_llm_flat)

    modes = [r["failure_mode"] for r in per_prompt_results]
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    summary = {
        "label": label,
        "lang": lang,
        "split": split,
        "n_prompts": len(per_prompt_results),
        "spearman_rho_mean": float(np.mean(rhos)) if rhos else None,
        "spearman_rho_std": float(np.std(rhos)) if rhos else None,
        "js_divergence_mean": float(np.mean(jss)),
        "js_divergence_std": float(np.std(jss)),
        "ece": ece,
        "wr@1": float(np.mean(wr1)),
        "wr@5": float(np.mean(wr5)),
        "wr@10": float(np.mean(wr10)),
        "wr@20": float(np.mean(wr20)),
        "failure_modes": mode_counts,
    }

    print()
    print(f"  Evaluation: {label}")
    print(f"  Lang: {lang}, Split: {split}, Prompts: {summary['n_prompts']}")
    print(f"  Spearman ρ:     {summary['spearman_rho_mean']:.4f} ± {summary['spearman_rho_std']:.4f}")
    print(f"  JS Divergence:  {summary['js_divergence_mean']:.4f} ± {summary['js_divergence_std']:.4f}")
    print(f"  ECE:            {summary['ece']:.4f}")
    print(f"  WR@1:           {summary['wr@1']:.4f}")
    print(f"  WR@5:           {summary['wr@5']:.4f}")
    print(f"  WR@10:          {summary['wr@10']:.4f}")
    print(f"  WR@20:          {summary['wr@20']:.4f}")
    print(f"  Failure modes:  {mode_counts}")

    plot_reliability_diagram(ece_bins, label)
    plot_rho_distribution(rhos, label)
    plot_js_distribution(jss, label)
    plot_failure_modes(mode_counts, label)

    eval_output = BASE_DIR / "results" / f"eval_{label}.json"
    with open(eval_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation saved to {eval_output}")

    return summary, per_prompt_results


def plot_reliability_diagram(ece_bins, label):
    """绘制 reliability diagram"""
    if not ece_bins:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    confs = [b["avg_confidence"] for b in ece_bins]
    freqs = [b["avg_frequency"] for b in ece_bins]

    ax.bar(range(len(confs)), freqs, alpha=0.6, color="steelblue", label="Actual frequency")
    ax.plot(range(len(confs)), confs, "r--o", label="Perfect calibration", linewidth=1.5)

    ax.set_xlabel("Bin (sorted by LLM confidence)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Reliability Diagram - {label}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"reliability_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved reliability_{label}.png")


def plot_rho_distribution(rhos, label):
    """绘制 Spearman ρ 分布"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rhos, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(np.mean(rhos), color="red", linestyle="--",
               label=f"Mean = {np.mean(rhos):.3f}")
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Count")
    ax.set_title(f"Spearman ρ Distribution - {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"rho_dist_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved rho_dist_{label}.png")


def plot_js_distribution(jss, label):
    """绘制 JS divergence 分布"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(jss, bins=30, color="coral", edgecolor="white", alpha=0.7)
    ax.axvline(np.mean(jss), color="red", linestyle="--",
               label=f"Mean = {np.mean(jss):.3f}")
    ax.set_xlabel("JS Divergence")
    ax.set_ylabel("Count")
    ax.set_title(f"JS Divergence Distribution - {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"js_dist_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved js_dist_{label}.png")


def plot_failure_modes(mode_counts, label):
    """绘制失败模式饼图"""
    fig, ax = plt.subplots(figsize=(6, 6))

    color_map = {
        "well_calibrated": "#66c2a5",
        "overconfident": "#fc8d62",
        "underconfident": "#8da0cb",
        "rank_misaligned": "#e78ac3",
        "phantom": "#a6d854",
    }

    labels_list = list(mode_counts.keys())
    sizes = list(mode_counts.values())
    colors = [color_map.get(l, "#cccccc") for l in labels_list]

    ax.pie(sizes, labels=labels_list, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Failure Mode Distribution - {label}")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"failure_modes_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved failure_modes_{label}.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM calibration results")
    parser.add_argument("--result", required=True, help="Path to LLM result JSON")
    parser.add_argument("--lang", required=True, choices=["en_pt", "en_hu", "en_ja", "en_ko", "en_vi"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--label", default=None, help="Label for this evaluation")
    args = parser.parse_args()

    run_evaluation(args.result, args.lang, args.split, args.label)


if __name__ == "__main__":
    main()
