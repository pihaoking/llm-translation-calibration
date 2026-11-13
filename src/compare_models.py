"""跨模型对比分析脚本

python src/compare_models.py --lang en_pt --split dev
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import evaluate_prompt, expected_calibration_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "Claude Haiku": "multi_claude-haiku",
    "Llama 3.2 3B": "multi_llama3.2-latest",
    "Qwen 2.5 7B": "multi_qwen2.5-7b",
    "Phi-4 14B": "multi_phi4-14b",
    "Gemma 2 27B": "multi_gemma2-27b",
}

MODEL_COLORS = {
    "Claude Haiku": "#6366f1",
    "Llama 3.2 3B": "#f59e0b",
    "Qwen 2.5 7B": "#10b981",
    "Phi-4 14B": "#ef4444",
    "Gemma 2 27B": "#8b5cf6",
}

MODEL_PARAMS = {
    "Claude Haiku": "n/a",
    "Llama 3.2 3B": "3.2B",
    "Qwen 2.5 7B": "7.6B",
    "Phi-4 14B": "14B",
    "Gemma 2 27B": "27B",
}


def load_gold(lang, split):
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        return json.load(f)


def evaluate_model(result_file, gold_data):
    """对单个模型结果运行评估，返回逐prompt结果"""
    with open(result_file, "r") as f:
        llm_results = json.load(f)

    gold_map = {p["prompt_id"]: {t["text"]: t["weight"] for t in p["translations"]}
                for p in gold_data}

    per_prompt = []
    p_human_flat, p_llm_flat = [], []

    for r in llm_results:
        pid = r["prompt_id"]
        p_llm = r.get("p_llm", {})
        p_human = gold_map.get(pid, {})
        if not p_human or not p_llm:
            continue

        result = evaluate_prompt(p_human, p_llm)
        result["prompt_id"] = pid
        result["n_generated"] = r.get("n_generated", len(p_llm))
        per_prompt.append(result)

        common = set(p_human.keys()) & set(p_llm.keys())
        for t in common:
            p_human_flat.append(p_human[t])
            p_llm_flat.append(p_llm[t])

    ece = 0.0
    if p_human_flat:
        ece, _ = expected_calibration_error(p_human_flat, p_llm_flat)

    return per_prompt, ece


def compute_summary(per_prompt, ece, model_name):
    """从逐prompt结果汇总"""
    rhos = [r["spearman_rho"] for r in per_prompt if not np.isnan(r["spearman_rho"])]
    jss = [r["js_divergence"] for r in per_prompt]
    modes = [r["failure_mode"] for r in per_prompt]
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    n_gen = [r["n_generated"] for r in per_prompt]

    return {
        "model": model_name,
        "params": MODEL_PARAMS.get(model_name, "?"),
        "n_prompts": len(per_prompt),
        "avg_translations": float(np.mean(n_gen)) if n_gen else 0,
        "spearman_rho": float(np.mean(rhos)) if rhos else None,
        "spearman_rho_std": float(np.std(rhos)) if rhos else None,
        "js_divergence": float(np.mean(jss)) if jss else None,
        "js_divergence_std": float(np.std(jss)) if jss else None,
        "ece": ece,
        "wr@1": float(np.mean([r["wr@1"] for r in per_prompt])),
        "wr@5": float(np.mean([r["wr@5"] for r in per_prompt])),
        "wr@10": float(np.mean([r["wr@10"] for r in per_prompt])),
        "wr@20": float(np.mean([r["wr@20"] for r in per_prompt])),
        "failure_modes": mode_counts,
        "phantom_rate": mode_counts.get("phantom", 0) / len(per_prompt) if per_prompt else 0,
        "well_calibrated_rate": mode_counts.get("well_calibrated", 0) / len(per_prompt) if per_prompt else 0,
        "rhos": rhos,
        "jss": jss,
    }


def plot_metric_comparison(summaries, lang, split):
    """主要指标柱状图对比"""
    models = [s["model"] for s in summaries]
    colors = [MODEL_COLORS.get(m, "#999") for m in models]

    metrics = {
        "Spearman ρ ↑": [s["spearman_rho"] or 0 for s in summaries],
        "1 - JS Div ↑": [1 - (s["js_divergence"] or 0) for s in summaries],
        "1 - ECE ↑": [1 - s["ece"] for s in summaries],
        "WR@10 ↑": [s["wr@10"] for s in summaries],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(range(len(models)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Model Calibration Comparison ({lang}, {split})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"compare_metrics_{lang}_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved compare_metrics_{lang}_{split}.png")


def plot_rho_violin(all_per_prompt, lang, split):
    """Spearman ρ 小提琴图"""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = list(all_per_prompt.keys())
    data = []
    positions = []
    colors = []

    for i, model in enumerate(models):
        rhos = [r["spearman_rho"] for r in all_per_prompt[model]
                if not np.isnan(r["spearman_rho"])]
        if rhos:
            data.append(rhos)
            positions.append(i)
            colors.append(MODEL_COLORS.get(model, "#999"))

    if not data:
        plt.close(fig)
        return

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Distribution of Spearman ρ Across Models ({lang})")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"compare_rho_violin_{lang}_{split}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved compare_rho_violin_{lang}_{split}.png")


def plot_failure_modes_stacked(summaries, lang, split):
    """失败模式堆叠柱状图"""
    models = [s["model"] for s in summaries]
    mode_names = ["well_calibrated", "rank_misaligned", "overconfident", "underconfident", "phantom"]
    mode_colors = {
        "well_calibrated": "#66c2a5",
        "overconfident": "#fc8d62",
        "underconfident": "#8da0cb",
        "rank_misaligned": "#e78ac3",
        "phantom": "#a6d854",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = np.zeros(len(models))
    for mode in mode_names:
        values = [s["failure_modes"].get(mode, 0) / s["n_prompts"] * 100 for s in summaries]
        ax.bar(range(len(models)), values, bottom=bottoms,
               color=mode_colors[mode], label=mode, edgecolor="white", linewidth=0.5)
        bottoms += values

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Failure Mode Distribution ({lang})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"compare_failure_modes_{lang}_{split}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved compare_failure_modes_{lang}_{split}.png")


def plot_params_vs_calibration(summaries, lang, split):
    """模型参数量 vs 校准性散点图"""
    param_values = {"3.2B": 3.2, "7.6B": 7.6, "14B": 14, "27B": 27}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    xs, ys_rho, ys_js, ys_ece, names, colors = [], [], [], [], [], []
    for s in summaries:
        pv = param_values.get(s["params"])
        if pv and s["spearman_rho"] is not None:
            xs.append(pv)
            ys_rho.append(s["spearman_rho"])
            ys_js.append(s["js_divergence"])
            ys_ece.append(s["ece"])
            names.append(s["model"])
            colors.append(MODEL_COLORS.get(s["model"], "#999"))

    for ax, ys, ylabel, title in [
        (axes[0], ys_rho, "Spearman ρ", "Params vs Rank Correlation"),
        (axes[1], ys_js, "JS Divergence", "Params vs JS Divergence"),
        (axes[2], ys_ece, "ECE", "Params vs Calibration Error"),
    ]:
        ax.scatter(xs, ys, c=colors, s=100, zorder=5, edgecolor="white")
        for x, y, name in zip(xs, ys, names):
            ax.annotate(name, (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Parameters (B)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Model Scale vs Calibration ({lang})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"compare_scale_{lang}_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved compare_scale_{lang}_{split}.png")


def print_latex_table(summaries):
    """输出 LaTeX 格式表格"""
    print("\n--- LaTeX Table ---")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\hline")
    print(r"Model & Params & Spearman $\rho$ & JS Div & ECE & WR@1 & WR@10 & Phantom\% & Calibrated\% \\")
    print(r"\hline")
    for s in summaries:
        rho = f"{s['spearman_rho']:.3f}" if s["spearman_rho"] else "N/A"
        js = f"{s['js_divergence']:.3f}" if s["js_divergence"] else "N/A"
        print(f"{s['model']} & {s['params']} & {rho} & {js} & "
              f"{s['ece']:.3f} & {s['wr@1']:.3f} & {s['wr@10']:.3f} & "
              f"{s['phantom_rate']:.1%} & {s['well_calibrated_rate']:.1%} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en_pt", choices=["en_pt", "en_hu", "en_ja", "en_ko", "en_vi"])
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    gold_data = load_gold(args.lang, args.split)

    summaries = []
    all_per_prompt = {}

    for model_name, file_prefix in MODEL_CONFIGS.items():
        result_file = RESULTS_DIR / f"{file_prefix}_{args.lang}_{args.split}.json"
        if not result_file.exists():
            print(f"  Skipping {model_name} (no result file: {result_file.name})")
            continue

        with open(result_file) as f:
            data = json.load(f)
        valid = sum(1 for r in data if r.get("n_generated", 0) > 0)
        if valid == 0:
            print(f"  Skipping {model_name} (0 valid results)")
            continue

        print(f"  Evaluating {model_name} ({valid}/{len(data)} valid)...")
        per_prompt, ece = evaluate_model(result_file, gold_data)
        summary = compute_summary(per_prompt, ece, model_name)
        summaries.append(summary)
        all_per_prompt[model_name] = per_prompt

    if len(summaries) < 2:
        print(f"\nOnly {len(summaries)} model(s) available. Need at least 2 for comparison.")
        if summaries:
            print_latex_table(summaries)
        return

    summaries.sort(key=lambda s: s["spearman_rho"] or 0, reverse=True)

    print()
    print(f"  CROSS-MODEL COMPARISON: {args.lang} ({args.split})")
    print(f"{'Model':<20} {'Params':>8} {'ρ':>8} {'JS':>8} {'ECE':>8} {'WR@10':>8} {'Phantom':>8} {'Good':>8}")
    for s in summaries:
        rho = f"{s['spearman_rho']:.3f}" if s["spearman_rho"] else "N/A"
        js = f"{s['js_divergence']:.3f}" if s["js_divergence"] else "N/A"
        print(f"{s['model']:<20} {s['params']:>8} {rho:>8} {js:>8} "
              f"{s['ece']:>8.3f} {s['wr@10']:>8.3f} "
              f"{s['phantom_rate']:>7.1%} {s['well_calibrated_rate']:>7.1%}")

    print("\nGenerating comparison plots...")
    plot_metric_comparison(summaries, args.lang, args.split)
    plot_rho_violin(all_per_prompt, args.lang, args.split)
    plot_failure_modes_stacked(summaries, args.lang, args.split)
    plot_params_vs_calibration(summaries, args.lang, args.split)

    print_latex_table(summaries)

    output_summaries = [{k: v for k, v in s.items() if k not in ("rhos", "jss")}
                        for s in summaries]
    output_file = RESULTS_DIR / f"compare_{args.lang}_{args.split}.json"
    with open(output_file, "w") as f:
        json.dump(output_summaries, f, indent=2)
    print(f"\nSaved comparison to {output_file}")


if __name__ == "__main__":
    main()
