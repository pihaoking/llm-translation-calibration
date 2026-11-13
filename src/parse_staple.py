"""解析 STAPLE gold 文件为结构化 JSON 格式"""

import json
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STAPLE_DIR = BASE_DIR / "2020 STAPLE Shared Task Data" / "staple-2020"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

LANG_PAIRS = ["en_pt", "en_hu", "en_ja", "en_ko", "en_vi"]
SPLITS = ["train", "dev", "test"]


def compute_entropy(weights):
    """计算 Shannon 熵 (以 nat 为单位)"""
    return -sum(w * math.log(w) for w in weights if w > 0)


def parse_gold_file(filepath):
    """解析一个 STAPLE gold 文件"""
    prompts = []
    current_prompt = None
    current_translations = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                # 空行 = block 结束，保存当前 prompt
                if current_prompt is not None:
                    prompts.append(_build_prompt_dict(current_prompt, current_translations))
                    current_prompt = None
                    current_translations = []
                continue

            parts = line.split("|")

            if parts[0].startswith("prompt_"):
                # prompt 行
                if current_prompt is not None:
                    # 保存上一个（文件末尾没有空行的情况）
                    prompts.append(_build_prompt_dict(current_prompt, current_translations))
                    current_translations = []
                current_prompt = {
                    "prompt_id": parts[0],
                    "english": parts[1],
                }
            else:
                # 翻译行: text|weight
                current_translations.append({
                    "text": parts[0],
                    "weight": float(parts[1]),
                })

    if current_prompt is not None:
        prompts.append(_build_prompt_dict(current_prompt, current_translations))

    return prompts


def _build_prompt_dict(prompt_info, translations):
    """构建一个 prompt 的完整字典"""
    weights = [t["weight"] for t in translations]
    sorted_weights = sorted(weights, reverse=True)

    return {
        "prompt_id": prompt_info["prompt_id"],
        "english": prompt_info["english"],
        "translations": translations,
        "num_translations": len(translations),
        "entropy": compute_entropy(weights),
        "top1_weight": sorted_weights[0] if sorted_weights else 0,
        "top5_weight": sum(sorted_weights[:5]),
        "top10_weight": sum(sorted_weights[:10]),
        "sentence_length": len(prompt_info["english"].split()),
    }


def find_gold_file(lang, split):
    """找到对应的 gold 文件路径"""
    lang_dir = STAPLE_DIR / lang
    for f in lang_dir.iterdir():
        if f.name.startswith(f"{split}.") and f.name.endswith(".gold.txt"):
            return f
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang in LANG_PAIRS:
        for split in SPLITS:
            gold_file = find_gold_file(lang, split)
            if gold_file is None:
                print(f"  [SKIP] {lang}/{split}: gold file not found")
                continue

            print(f"  Parsing {lang}/{split} from {gold_file.name} ...")
            prompts = parse_gold_file(gold_file)

            output_file = OUTPUT_DIR / f"{lang}_{split}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)

            n_prompts = len(prompts)
            n_trans = sum(p["num_translations"] for p in prompts)
            avg_trans = n_trans / n_prompts if n_prompts else 0
            avg_entropy = sum(p["entropy"] for p in prompts) / n_prompts if n_prompts else 0

            print(f"    → {n_prompts} prompts, {n_trans} translations (avg {avg_trans:.1f}/prompt)")
            print(f"    → avg entropy: {avg_entropy:.2f}")
            print(f"    → saved to {output_file.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
