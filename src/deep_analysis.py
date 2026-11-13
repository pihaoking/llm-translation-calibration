"""深度分析脚本：生成论文所需的进阶分析和图表

python src/deep_analysis.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import evaluate_prompt, expected_calibration_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"


def load_gold(lang, split="dev"):
    with open(DATA_DIR / f"{lang}_{split}.json") as f:
        return json.load(f)


def load_result(path):
    with open(path) as f:
        return json.load(f)


def evaluate_all(result_file, gold_data):
    """逐 prompt 评估"""
    gold_map = {p["prompt_id"]: {t["text"]: t["weight"] for t in p["translations"]}
                for p in gold_data}
    per_prompt = []
    for r in load_result(result_file):
        pid = r["prompt_id"]
        p_llm = r.get("p_llm", {})
        p_human = gold_map.get(pid, {})
        if not p_human or not p_llm:
            continue
        result = evaluate_prompt(p_human, p_llm)
        result["prompt_id"] = pid
        result["english"] = r["english"]
        result["n_generated"] = r.get("n_generated", len(p_llm))
        result["n_gold"] = len(p_human)
        result["p_llm"] = p_llm
        result["p_human"] = p_human
        per_prompt.append(result)
    return per_prompt


# analysis 1: Cross-language (Haiku)

def analysis_cross_language():
    """语言复杂度 vs 校准性"""
    print("\n=== Analysis 1: Cross-language calibration (Claude Haiku) ===\n")

    langs = ["en_pt", "en_hu", "en_ko", "en_ja"]
    lang_labels = {
        "en_pt": "Portuguese\n(Romance)",
        "en_hu": "Hungarian\n(Agglutinative)",
        "en_ko": "Korean\n(SOV/Agglut.)",
        "en_ja": "Japanese\n(SOV/Mixed)",
    }

    results = {}
    for lang in langs:
        result_file = RESULTS_DIR / f"multi_claude-haiku_{lang}_dev.json"
        if not result_file.exists():
            print(f"  Skip {lang}: no file")
            continue
        gold = load_gold(lang)
        pp = evaluate_all(result_file, gold)
        rhos = [r["spearman_rho"] for r in pp if not np.isnan(r["spearman_rho"])]
        jss = [r["js_divergence"] for r in pp]
        modes = [r["failure_mode"] for r in pp]
        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        results[lang] = {
            "rhos": rhos,
            "jss": jss,
            "rho_mean": np.mean(rhos),
            "js_mean": np.mean(jss),
            "n": len(pp),
            "mode_counts": mode_counts,
            "pp": pp,
        }
        print(f"  {lang}: ρ={np.mean(rhos):.3f}, JS={np.mean(jss):.3f}, n={len(pp)}")

    if len(results) < 2:
        print("  Not enough languages for cross-language analysis")
        return

    # Figure: 跨语言 violin plot (ρ + JS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#6366f1", "#f59e0b", "#ef4444", "#10b981"]
    for i, lang in enumerate(langs):
        if lang not in results:
            continue
        r = results[lang]

        parts1 = ax1.violinplot([r["rhos"]], positions=[i], showmeans=True, showmedians=True)
        for pc in parts1["bodies"]:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        parts2 = ax2.violinplot([r["jss"]], positions=[i], showmeans=True, showmedians=True)
        for pc in parts2["bodies"]:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

    xlabels = [lang_labels.get(l, l) for l in langs if l in results]
    for ax, title, ylabel in [
        (ax1, "Spearman ρ by Language", "Spearman ρ"),
        (ax2, "JS Divergence by Language", "JS Divergence"),
    ]:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    ax1.axhline(0, color="gray", ls=":", alpha=0.5)

    fig.suptitle("Claude Haiku: Calibration Across Languages", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "analysis_cross_language.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved analysis_cross_language.png")

    # Figure: 失败模式按语言
    fig, ax = plt.subplots(figsize=(10, 5))
    mode_names = ["well_calibrated", "rank_misaligned", "overconfident", "underconfident", "phantom"]
    mode_colors = ["#66c2a5", "#e78ac3", "#fc8d62", "#8da0cb", "#a6d854"]
    bottoms = np.zeros(len(langs))
    avail_langs = [l for l in langs if l in results]

    for j, mode in enumerate(mode_names):
        vals = [results[l]["mode_counts"].get(mode, 0) / results[l]["n"] * 100
                for l in avail_langs]
        ax.bar(range(len(avail_langs)), vals, bottom=bottoms[:len(avail_langs)],
               color=mode_colors[j], label=mode, edgecolor="white")
        bottoms[:len(avail_langs)] += vals

    ax.set_xticks(range(len(avail_langs)))
    ax.set_xticklabels([lang_labels.get(l, l) for l in avail_langs], fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Failure Mode Distribution by Language (Claude Haiku)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "analysis_failure_by_lang.png", dpi=150)
    plt.close(fig)
    print("  Saved analysis_failure_by_lang.png")

    return results


# analysis 2: Prompt difficulty

def analysis_prompt_difficulty():
    """翻译变体数 (prompt 难度) vs 校准性"""
    print("\n=== Analysis 2: Prompt difficulty vs calibration ===\n")

    result_file = RESULTS_DIR / "multi_claude-haiku_en_pt_dev.json"
    gold = load_gold("en_pt")
    pp = evaluate_all(result_file, gold)

    n_golds = [r["n_gold"] for r in pp]
    rhos = [r["spearman_rho"] for r in pp]
    jss = [r["js_divergence"] for r in pp]

    bins = [0, 10, 20, 40, 80, 200, float("inf")]
    labels = ["1-10", "11-20", "21-40", "41-80", "81-200", "200+"]
    groups = {l: {"rhos": [], "jss": []} for l in labels}

    for ng, rho, js in zip(n_golds, rhos, jss):
        for i in range(len(bins)-1):
            if bins[i] < ng <= bins[i+1]:
                if not np.isnan(rho):
                    groups[labels[i]]["rhos"].append(rho)
                groups[labels[i]]["jss"].append(js)
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    valid_labels = [l for l in labels if groups[l]["rhos"]]

    for ax, key, title, ylabel in [
        (ax1, "rhos", "Rank Correlation vs Prompt Difficulty", "Spearman ρ"),
        (ax2, "jss", "JS Divergence vs Prompt Difficulty", "JS Divergence"),
    ]:
        means = [np.mean(groups[l][key]) for l in valid_labels]
        stds = [np.std(groups[l][key]) for l in valid_labels]
        counts = [len(groups[l][key]) for l in valid_labels]

        bars = ax.bar(range(len(valid_labels)), means, yerr=stds,
                      color="#6366f1", alpha=0.7, capsize=3, edgecolor="white")
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels)
        ax.set_xlabel("Number of Gold Translations")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, n in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={n}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("How Prompt Difficulty Affects LLM Calibration (Claude Haiku, en_pt)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "analysis_prompt_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved analysis_prompt_difficulty.png")

    valid = [(ng, rho) for ng, rho in zip(n_golds, rhos) if not np.isnan(rho)]
    if valid:
        ngs, rs = zip(*valid)
        corr, pval = stats.spearmanr(ngs, rs)
        print(f"  Correlation (n_gold vs ρ): r={corr:.3f}, p={pval:.4f}")

    corr_js, pval_js = stats.spearmanr(n_golds, jss)
    print(f"  Correlation (n_gold vs JS): r={corr_js:.3f}, p={pval_js:.4f}")


# analysis 3: Case studies

def analysis_case_studies():
    """最好/最差 prompt 案例"""
    print("\n=== Analysis 3: Case studies (Claude Haiku, en_pt) ===\n")

    result_file = RESULTS_DIR / "multi_claude-haiku_en_pt_dev.json"
    gold = load_gold("en_pt")
    pp = evaluate_all(result_file, gold)

    pp_sorted = sorted(pp, key=lambda r: r["js_divergence"])

    print("--- Best calibrated (lowest JS divergence) ---")
    for r in pp_sorted[:5]:
        n_common = len(set(r["p_llm"].keys()) & set(r["p_human"].keys()))
        print(f"  '{r['english']}'")
        print(f"    JS={r['js_divergence']:.3f}, ρ={r['spearman_rho']:.3f}, "
              f"common={n_common}/{len(r['p_llm'])}, mode={r['failure_mode']}")

    print("\n--- Worst calibrated (highest JS divergence) ---")
    for r in pp_sorted[-5:]:
        n_common = len(set(r["p_llm"].keys()) & set(r["p_human"].keys()))
        print(f"  '{r['english']}'")
        print(f"    JS={r['js_divergence']:.3f}, ρ={r['spearman_rho']:.3f}, "
              f"common={n_common}/{len(r['p_llm'])}, mode={r['failure_mode']}")

    # 统计: 句子长度 vs 校准
    eng_lens = [len(r["english"].split()) for r in pp]
    jss = [r["js_divergence"] for r in pp]
    corr, pval = stats.spearmanr(eng_lens, jss)
    print(f"\n  Sentence length vs JS: r={corr:.3f}, p={pval:.4f}")

    # Plot: 句子长度 vs JS
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(eng_lens, jss, alpha=0.3, s=20, color="#6366f1")
    ax.set_xlabel("English Sentence Length (words)")
    ax.set_ylabel("JS Divergence")
    ax.set_title(f"Sentence Length vs Calibration Error (r={corr:.3f})")
    ax.grid(alpha=0.3)

    z = np.polyfit(eng_lens, jss, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(eng_lens), max(eng_lens), 100)
    ax.plot(xs, p(xs), "r--", alpha=0.7, label=f"Trend (r={corr:.3f})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "analysis_length_vs_js.png", dpi=150)
    plt.close(fig)
    print("  Saved analysis_length_vs_js.png")


# analysis 4: Phantom translation analysis

def analysis_phantom():
    """Phantom 翻译深入分析"""
    print("\n=== Analysis 4: Phantom translation analysis (Claude Haiku, en_pt) ===\n")

    result_file = RESULTS_DIR / "multi_claude-haiku_en_pt_dev.json"
    llm_results = load_result(result_file)
    gold = load_gold("en_pt")
    gold_map = {p["prompt_id"]: {t["text"]: t["weight"] for t in p["translations"]}
                for p in gold}

    phantom_masses = []
    phantom_counts = []
    total_probs_in_gold = []

    for r in llm_results:
        pid = r["prompt_id"]
        p_llm = r.get("p_llm", {})
        p_human = gold_map.get(pid, {})
        if not p_llm:
            continue

        gold_set = set(p_human.keys())
        phantom_mass = sum(p for t, p in p_llm.items() if t not in gold_set)
        gold_mass = sum(p for t, p in p_llm.items() if t in gold_set)
        n_phantom = sum(1 for t in p_llm if t not in gold_set)
        n_total = len(p_llm)

        phantom_masses.append(phantom_mass)
        phantom_counts.append(n_phantom / n_total if n_total > 0 else 0)
        total_probs_in_gold.append(gold_mass)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(phantom_masses, bins=30, color="#ef4444", edgecolor="white", alpha=0.7)
    axes[0].axvline(np.mean(phantom_masses), color="black", ls="--",
                    label=f"Mean={np.mean(phantom_masses):.3f}")
    axes[0].set_xlabel("Phantom Probability Mass")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Phantom Mass")
    axes[0].legend()

    axes[1].hist(phantom_counts, bins=30, color="#f59e0b", edgecolor="white", alpha=0.7)
    axes[1].axvline(np.mean(phantom_counts), color="black", ls="--",
                    label=f"Mean={np.mean(phantom_counts):.1%}")
    axes[1].set_xlabel("Fraction of Phantom Translations")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Fraction of Non-Gold Translations")
    axes[1].legend()

    axes[2].hist(total_probs_in_gold, bins=30, color="#10b981", edgecolor="white", alpha=0.7)
    axes[2].axvline(np.mean(total_probs_in_gold), color="black", ls="--",
                    label=f"Mean={np.mean(total_probs_in_gold):.3f}")
    axes[2].set_xlabel("Probability Mass in Gold Set")
    axes[2].set_ylabel("Count")
    axes[2].set_title("LLM Probability Assigned to Gold Translations")
    axes[2].legend()

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phantom Translation Analysis (Claude Haiku, en_pt)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "analysis_phantom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Mean phantom mass: {np.mean(phantom_masses):.3f}")
    print(f"  Mean phantom fraction: {np.mean(phantom_counts):.1%}")
    print(f"  Mean gold mass: {np.mean(total_probs_in_gold):.3f}")
    print("  Saved analysis_phantom.png")


# analysis 5: Summary statistics table

def analysis_summary_table():
    """生成完整汇总表 (JSON)"""
    print("\n=== Analysis 5: Summary table ===\n")

    all_evals = []

    for lang in ["en_pt", "en_hu", "en_ja", "en_ko"]:
        result_file = RESULTS_DIR / f"multi_claude-haiku_{lang}_dev.json"
        if not result_file.exists():
            continue
        eval_file = RESULTS_DIR / f"eval_haiku_{lang}.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
                data["model"] = "Claude Haiku"
                all_evals.append(data)

    for label, model_name in [("llama32_en_pt", "Llama 3.2 3B"),
                               ("qwen25_en_pt", "Qwen 2.5 7B")]:
        eval_file = RESULTS_DIR / f"eval_{label}.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
                data["model"] = model_name
                all_evals.append(data)

    eval_file = RESULTS_DIR / "eval_phi4_en_pt.json"
    if eval_file.exists():
        with open(eval_file) as f:
            data = json.load(f)
            data["model"] = "Phi-4 14B"
            all_evals.append(data)

    output_file = RESULTS_DIR / "summary_all.json"
    with open(output_file, "w") as f:
        json.dump(all_evals, f, indent=2)
    print(f"  Saved {output_file}")

    print(f"\n  {'Model':<20} {'Lang':>6} {'ρ':>8} {'JS':>8} {'ECE':>8} {'WR@10':>8}")
    for e in all_evals:
        rho = f"{e['spearman_rho_mean']:.3f}" if e.get("spearman_rho_mean") else "N/A"
        js = f"{e['js_divergence_mean']:.3f}" if e.get("js_divergence_mean") else "N/A"
        print(f"  {e.get('model', '?'):<20} {e['lang']:>6} {rho:>8} {js:>8} "
              f"{e['ece']:>8.3f} {e.get('wr@10', 0):>8.3f}")



def main():
    print("  DEEP ANALYSIS: LLM Calibration in Educational Translation")

    analysis_cross_language()
    analysis_prompt_difficulty()
    analysis_case_studies()
    analysis_phantom()
    analysis_summary_table()

    print()
    print("  All analyses complete!")


if __name__ == "__main__":
    main()
