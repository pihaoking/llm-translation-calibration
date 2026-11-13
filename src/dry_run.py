"""端到端 dry run: 用随机数据模拟 LLM 输出，验证整条管线"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import evaluate_prompt, evaluate_dataset, expected_calibration_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"


def simulate_llm_output(gold_prompts, noise_level=0.3):
    """模拟 LLM 输出: 在人类分布上加噪声"""
    simulated = []

    for prompt in gold_prompts:
        p_human = {t["text"]: t["weight"] for t in prompt["translations"]}

        # 从 gold 翻译中随机取一部分 (模拟 LLM 不一定覆盖所有翻译)
        n_keep = max(3, int(len(p_human) * 0.3))
        # 按权重优先保留
        sorted_trans = sorted(p_human.items(), key=lambda x: -x[1])
        kept = dict(sorted_trans[:n_keep])

        p_llm = {}
        for t, w in kept.items():
            noisy_w = w * (1 - noise_level) + random.random() * noise_level
            p_llm[t] = max(noisy_w, 1e-6)

        total = sum(p_llm.values())
        p_llm = {t: w / total for t, w in p_llm.items()}

        simulated.append({
            "prompt_id": prompt["prompt_id"],
            "english": prompt["english"],
            "p_llm": p_llm,
        })

    return simulated


def main():
    lang = "en_pt"
    split = "dev"

    print(f"Loading gold data: {lang}/{split}")
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        gold_prompts = json.load(f)

    # 只取前 50 个 prompt 做 dry run
    gold_prompts = gold_prompts[:50]
    print(f"Using {len(gold_prompts)} prompts for dry run\n")

    # 模拟 3 种 "LLM": 好/中/差
    for name, noise in [("good_llm", 0.1), ("medium_llm", 0.5), ("bad_llm", 0.9)]:
        print(f"--- Simulating: {name} (noise={noise}) ---")
        simulated = simulate_llm_output(gold_prompts, noise_level=noise)

        # 保存模拟结果
        sim_path = RESULTS_DIR / f"dryrun_{name}_{lang}_{split}.json"
        with open(sim_path, "w", encoding="utf-8") as f:
            json.dump(simulated, f, ensure_ascii=False, indent=2)

        prompts_human = []
        prompts_llm = []
        for gp, sp in zip(gold_prompts, simulated):
            p_human = {t["text"]: t["weight"] for t in gp["translations"]}
            prompts_human.append(p_human)
            prompts_llm.append(sp["p_llm"])

        result = evaluate_dataset(prompts_human, prompts_llm)

        print(f"  Spearman ρ:    {result['spearman_rho_mean']:.4f} ± {result['spearman_rho_std']:.4f}")
        print(f"  JS Divergence: {result['js_divergence_mean']:.4f} ± {result['js_divergence_std']:.4f}")
        print(f"  ECE:           {result['ece']:.4f}")
        print(f"  WR@1:          {result['wr@1']:.4f}")
        print(f"  WR@5:          {result['wr@5']:.4f}")
        print(f"  WR@10:         {result['wr@10']:.4f}")
        print(f"  Failure modes: {result['failure_modes']}")
        print()

    print("Dry run complete! Pipeline is working.")


if __name__ == "__main__":
    main()
