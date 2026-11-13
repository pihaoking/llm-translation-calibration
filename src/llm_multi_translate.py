"""LLM 多翻译管线

python src/llm_multi_translate.py --lang en_pt --split dev --model claude-haiku --max_prompts 5
python src/llm_multi_translate.py --lang en_pt --split dev --model gpt-4o-mini --max_prompts 5
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
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

MODEL_MAP = {
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
}

PROVIDERS = {
    "claude-haiku": "anthropic",
    "claude-sonnet": "anthropic",
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
}

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
        # 去掉编号 (1. / 1) / 1: 等格式)
        cleaned = re.sub(r"^\d+[\.\)\:\-]\s*", "", line)
        cleaned = cleaned.strip().lower()
        cleaned = cleaned.strip('"').strip("'").strip('"').strip('"')
        if cleaned:
            translations.append(cleaned)
    return translations


def rank_to_probability(n_translations: int, decay: float = 0.7) -> list[float]:
    """用指数衰减将排名转为概率分布"""
    weights = [decay ** i for i in range(n_translations)]
    total = sum(weights)
    return [w / total for w in weights]


def generate_translations(
    client,
    english: str,
    lang: str,
    model: str,
    provider: str,
) -> dict:
    """一次 API 调用生成多个翻译。支持 Anthropic 和 OpenAI"""
    target_lang = LANG_NAMES[lang]
    api_model = MODEL_MAP.get(model, model)
    sys_msg = SYSTEM_PROMPT.format(target_lang=target_lang)
    usr_msg = USER_PROMPT.format(target_lang=target_lang, english=english)

    if provider == "anthropic":
        response = client.messages.create(
            model=api_model,
            max_tokens=2000,
            temperature=1.0,
            system=sys_msg,
            messages=[{"role": "user", "content": usr_msg}],
        )
        raw_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
    else:  # openai
        response = client.chat.completions.create(
            model=api_model,
            max_tokens=2000,
            temperature=1.0,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ],
        )
        raw_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

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
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run_multi_translate(lang, split, model, max_prompts=None, rate_limit_delay=0.1):
    """主流程"""
    prompts = load_prompts(lang, split)
    if max_prompts:
        prompts = prompts[:max_prompts]

    provider = PROVIDERS.get(model, "openai")
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
    else:
        from openai import OpenAI
        client = OpenAI()

    print(f"Multi-translate {len(prompts)} prompts")
    print(f"Model: {model}, Lang: {lang}, Split: {split}")
    print()

    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for prompt in tqdm(prompts, desc="Translating"):
        english = prompt["english"]
        gold_set = {t["text"] for t in prompt["translations"]}

        try:
            result = generate_translations(client, english, lang, model, provider)
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
            time.sleep(1)
            continue

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        # 统计 gold 匹配
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

        time.sleep(rate_limit_delay)

    output_file = RESULTS_DIR / f"multi_{model}_{lang}_{split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_gen = np.mean([r["n_generated"] for r in results])
    avg_gold = np.mean([r["n_in_gold"] for r in results])
    avg_phantom = np.mean([r["phantom_mass"] for r in results])
    gold_rate = np.mean([
        r["n_in_gold"] / r["n_generated"] if r["n_generated"] > 0 else 0
        for r in results
    ])

    cost_input = total_input_tokens / 1_000_000 * (1.0 if "haiku" in model else 3.0)
    cost_output = total_output_tokens / 1_000_000 * (5.0 if "haiku" in model else 15.0)
    total_cost = cost_input + cost_output

    print(f"\nResults:")
    print(f"  Avg translations generated: {avg_gen:.1f}")
    print(f"  Avg in gold set: {avg_gold:.1f}")
    print(f"  Gold hit rate: {gold_rate:.1%}")
    print(f"  Avg phantom mass: {avg_phantom:.3f}")
    print(f"\nTokens used: {total_input_tokens} input + {total_output_tokens} output")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"\nSaved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en_pt", choices=list(LANG_NAMES.keys()))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default="claude-haiku", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--rate_limit_delay", type=float, default=0.1)
    args = parser.parse_args()

    run_multi_translate(args.lang, args.split, args.model, args.max_prompts, args.rate_limit_delay)


if __name__ == "__main__":
    main()
