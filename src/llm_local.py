"""本地模型多翻译管线 (via Ollama)

python src/llm_local.py --lang en_pt --split dev --model phi4:14b --max_prompts 5
python src/llm_local.py --lang en_pt --split dev --model gemma2:27b --max_prompts 5
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LANG_NAMES = {
    "en_pt": "Portuguese",
    "en_hu": "Hungarian",
    "en_ja": "Japanese",
    "en_ko": "Korean",
    "en_vi": "Vietnamese",
}

OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = """You are a language education expert. When given an English sentence, list ALL reasonable translations in {target_lang}.

Rules:
- Include formal and informal variations
- Include different word choices (synonyms)
- Include variations with/without subject pronouns
- Each translation on a new line, numbered
- Output ONLY the numbered translations, nothing else
- Aim for at least 15-30 translations if possible
- All translations must be lowercase"""

USER_PROMPT = """List all reasonable {target_lang} translations of this English sentence:

"{english}"
"""


def load_prompts(lang, split):
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        return json.load(f)


def parse_translations(response_text: str) -> list[str]:
    """从模型输出中解析翻译列表"""
    translations = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # 跳过明显的非翻译行 (解释、注释等)
        if any(line.lower().startswith(w) for w in ["note:", "here", "these", "the ", "i ", "above"]):
            continue
        cleaned = re.sub(r"^\d+[\.\)\:\-]\s*", "", line)
        # 去掉 markdown bold/italic
        cleaned = re.sub(r"\*+", "", cleaned)
        cleaned = cleaned.strip().lower()
        cleaned = cleaned.strip('"').strip("'").strip('\u201c').strip('\u201d')
        # 去掉括号注释 (informal) 等
        cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 1:
            translations.append(cleaned)
    return translations


def rank_to_probability(n_translations: int, decay: float = 0.7) -> list[float]:
    """排名 → 概率分布 (指数衰减 + 归一化)"""
    weights = [decay ** i for i in range(n_translations)]
    total = sum(weights)
    return [w / total for w in weights]


def generate_translations_ollama(
    english: str,
    lang: str,
    model: str,
    temperature: float = 0.8,
) -> dict:
    """通过 Ollama API 生成多个翻译"""
    target_lang = LANG_NAMES[lang]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(target_lang=target_lang)},
            {"role": "user", "content": USER_PROMPT.format(target_lang=target_lang, english=english)},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2000,
        },
    }

    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    raw_text = data["message"]["content"]
    translations = parse_translations(raw_text)

    seen = set()
    unique = []
    for t in translations:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if unique:
        probs = rank_to_probability(len(unique))
        p_llm = {t: p for t, p in zip(unique, probs)}
    else:
        p_llm = {}

    return {
        "translations": unique,
        "p_llm": p_llm,
        "raw_response": raw_text,
        "eval_count": data.get("eval_count", 0),
        "total_duration_ms": data.get("total_duration", 0) / 1e6,
    }


def run_local(lang, split, model, max_prompts=None):
    """主流程"""
    prompts = load_prompts(lang, split)
    if max_prompts:
        prompts = prompts[:max_prompts]

    # 模型名清理 (用于文件名)
    model_safe = model.replace(":", "-").replace("/", "-")

    print(f"Multi-translate {len(prompts)} prompts (local)")
    print(f"Model: {model}, Lang: {lang}, Split: {split}")
    print()

    results = []
    total_tokens = 0

    for prompt in tqdm(prompts, desc=f"{model_safe}"):
        english = prompt["english"]
        gold_set = {t["text"] for t in prompt["translations"]}

        try:
            result = generate_translations_ollama(english, lang, model)
        except Exception as e:
            print(f"\n  Error on '{english[:40]}': {e}")
            results.append({
                "prompt_id": prompt["prompt_id"],
                "english": english,
                "p_llm": {},
                "translations": [],
                "n_generated": 0,
                "n_in_gold": 0,
                "phantom_mass": 0,
            })
            continue

        total_tokens += result["eval_count"]

        n_in_gold = sum(1 for t in result["translations"] if t in gold_set)
        phantom_mass = sum(
            p for t, p in result["p_llm"].items() if t not in gold_set
        )

        results.append({
            "prompt_id": prompt["prompt_id"],
            "english": english,
            "p_llm": result["p_llm"],
            "translations": result["translations"],
            "n_generated": len(result["translations"]),
            "n_in_gold": n_in_gold,
            "phantom_mass": phantom_mass,
        })

    output_file = RESULTS_DIR / f"multi_{model_safe}_{lang}_{split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_gen = np.mean([r["n_generated"] for r in results])
    avg_gold = np.mean([r["n_in_gold"] for r in results])
    avg_phantom = np.mean([r["phantom_mass"] for r in results])
    gold_rate = np.mean([
        r["n_in_gold"] / r["n_generated"] if r["n_generated"] > 0 else 0
        for r in results
    ])

    print(f"\nResults ({model}):")
    print(f"  Avg translations generated: {avg_gen:.1f}")
    print(f"  Avg in gold set: {avg_gold:.1f}")
    print(f"  Gold hit rate: {gold_rate:.1%}")
    print(f"  Avg phantom mass: {avg_phantom:.3f}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"\nSaved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en_pt", choices=list(LANG_NAMES.keys()))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default="phi4:14b")
    parser.add_argument("--max_prompts", type=int, default=None)
    args = parser.parse_args()

    # 检查 ollama 是否运行
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama models available: {models}")
        if args.model not in models:
            print(f"WARNING: {args.model} not found. Available: {models}")
            print(f"Run: ollama pull {args.model}")
            return
    except Exception:
        print("ERROR: Ollama not running. Start with: ollama serve")
        return

    run_local(args.lang, args.split, args.model, args.max_prompts)


if __name__ == "__main__":
    main()
