"""STAPLE 数据集描述性统计与可视化"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANG_PAIRS = ["en_pt", "en_hu", "en_ja", "en_ko", "en_vi"]
LANG_LABELS = {
    "en_pt": "EN→PT",
    "en_hu": "EN→HU",
    "en_ja": "EN→JA",
    "en_ko": "EN→KO",
    "en_vi": "EN→VI",
}


def load_data(lang, split="test"):
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        return json.load(f)


def fig1_translation_count_distribution():
    """图1: 各语言对的翻译数量分布 (箱线图)"""
    fig, ax = plt.subplots(figsize=(10, 5))

    data_all = []
    labels = []
    for lang in LANG_PAIRS:
        prompts = load_data(lang)
        counts = [p["num_translations"] for p in prompts]
        data_all.append(counts)
        labels.append(LANG_LABELS[lang])

    bp = ax.boxplot(data_all, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))

    colors = sns.color_palette("Set2", len(LANG_PAIRS))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Number of Translations per Prompt")
    ax.set_title("Translation Count Distribution by Language Pair (Test Set)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_translation_counts.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_translation_counts.png")


def fig2_weight_distribution():
    """图2: 翻译权重的长尾分布 (log-scale 直方图)"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for i, lang in enumerate(LANG_PAIRS):
        prompts = load_data(lang)
        all_weights = []
        for p in prompts:
            all_weights.extend([t["weight"] for t in p["translations"]])

        axes[i].hist(all_weights, bins=50, color=sns.color_palette("Set2")[i],
                     edgecolor="white", linewidth=0.3)
        axes[i].set_yscale("log")
        axes[i].set_title(LANG_LABELS[lang])
        axes[i].set_xlabel("Weight")
        if i == 0:
            axes[i].set_ylabel("Count (log scale)")
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Translation Weight Distribution (Test Set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_weight_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved fig2_weight_distribution.png")


def fig3_entropy_distribution():
    """图3: 各语言对的翻译熵分布"""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette("Set2", len(LANG_PAIRS))
    for i, lang in enumerate(LANG_PAIRS):
        prompts = load_data(lang)
        entropies = [p["entropy"] for p in prompts]
        ax.hist(entropies, bins=30, alpha=0.6, label=LANG_LABELS[lang],
                color=colors[i], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Shannon Entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Translation Entropy Distribution by Language Pair (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_entropy_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_entropy_distribution.png")


def fig4_topk_coverage():
    """图4: Top-K 覆盖率曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ks = [1, 2, 3, 5, 10, 20, 50]
    colors = sns.color_palette("Set2", len(LANG_PAIRS))

    for i, lang in enumerate(LANG_PAIRS):
        prompts = load_data(lang)
        coverages = []

        for k in ks:
            covs = []
            for p in prompts:
                sorted_w = sorted([t["weight"] for t in p["translations"]], reverse=True)
                covs.append(sum(sorted_w[:k]))
            coverages.append(np.mean(covs))

        ax.plot(ks, coverages, marker="o", label=LANG_LABELS[lang],
                color=colors[i], linewidth=2)

    ax.set_xlabel("K (Top-K Translations)")
    ax.set_ylabel("Cumulative Weight Coverage")
    ax.set_title("Top-K Weight Coverage by Language Pair (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_topk_coverage.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_topk_coverage.png")


def fig5_sentence_length_vs_entropy():
    """图5: 句子长度 vs 翻译熵"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    colors = sns.color_palette("Set2", len(LANG_PAIRS))

    for i, lang in enumerate(LANG_PAIRS):
        prompts = load_data(lang)
        lengths = [p["sentence_length"] for p in prompts]
        entropies = [p["entropy"] for p in prompts]

        axes[i].scatter(lengths, entropies, alpha=0.3, s=10, color=colors[i])
        z = np.polyfit(lengths, entropies, 1)
        x_line = np.linspace(min(lengths), max(lengths), 100)
        axes[i].plot(x_line, np.polyval(z, x_line), color="red", linewidth=1.5,
                     linestyle="--", label=f"slope={z[0]:.2f}")

        axes[i].set_title(LANG_LABELS[lang])
        axes[i].set_xlabel("Sentence Length (words)")
        if i == 0:
            axes[i].set_ylabel("Entropy (nats)")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    fig.suptitle("Sentence Length vs Translation Entropy (Test Set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_length_vs_entropy.png", dpi=150)
    plt.close(fig)
    print("  Saved fig5_length_vs_entropy.png")


def print_summary_table():
    """打印汇总统计表"""
    print()
    print(f"{'Lang':<8} {'Prompts':>8} {'Total Trans':>12} {'Avg Trans':>10} "
          f"{'Avg Entropy':>12} {'Avg Top1%':>10} {'Avg Top5%':>10} {'Avg Top10%':>11}")

    for lang in LANG_PAIRS:
        prompts = load_data(lang)
        n = len(prompts)
        total = sum(p["num_translations"] for p in prompts)
        avg_trans = total / n
        avg_ent = np.mean([p["entropy"] for p in prompts])
        avg_t1 = np.mean([p["top1_weight"] for p in prompts]) * 100
        avg_t5 = np.mean([p["top5_weight"] for p in prompts]) * 100
        avg_t10 = np.mean([p["top10_weight"] for p in prompts]) * 100

        print(f"{LANG_LABELS[lang]:<8} {n:>8} {total:>12} {avg_trans:>10.1f} "
              f"{avg_ent:>12.2f} {avg_t1:>9.1f}% {avg_t5:>9.1f}% {avg_t10:>10.1f}%")



def main():
    print("Generating EDA visualizations...\n")

    print_summary_table()

    print("\nGenerating figures...")
    fig1_translation_count_distribution()
    fig2_weight_distribution()
    fig3_entropy_distribution()
    fig4_topk_coverage()
    fig5_sentence_length_vs_entropy()

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
