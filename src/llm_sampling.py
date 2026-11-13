"""LLM Sampling 管线 (路线 B)

python src/llm_sampling.py --lang en_pt --split dev --model gpt-4o --n_samples 50 --max_prompts 10
python src/llm_sampling.py --lang en_pt --split dev --model claude-sonnet --n_samples 50 --max_prompts 10
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

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

MODEL_PROVIDERS = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "claude-sonnet": "anthropic",
    "claude-haiku": "anthropic",
}


def load_prompts(lang, split):
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        return json.load(f)


def sample_openai(client, english: str, lang: str, model: str, temperature: float) -> str:
    """用 OpenAI API 生成一次翻译"""
    target_lang = LANG_NAMES[lang]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a translator. Output only the translation, nothing else."},
            {"role": "user", "content": f"Translate the following English sentence to {target_lang}.\n\n{english}"},
        ],
        max_tokens=200,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip().lower()


def sample_anthropic(client, english: str, lang: str, model: str, temperature: float) -> str:
    """用 Anthropic API 生成一次翻译"""
    target_lang = LANG_NAMES[lang]

    model_map = {
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-haiku": "claude-haiku-4-5-20251001",
    }
    api_model = model_map.get(model, model)

    response = client.messages.create(
        model=api_model,
        max_tokens=200,
        temperature=temperature,
        system="You are a translator. Output only the translation, nothing else.",
        messages=[
            {"role": "user", "content": f"Translate the following English sentence to {target_lang}.\n\n{english}"},
        ],
    )
    return response.content[0].text.strip().lower()


def sample_translations(
    english: str,
    lang: str,
    model: str,
    n_samples: int,
    temperature: float,
    client_openai=None,
    client_anthropic=None,
    rate_limit_delay: float = 0.05,
) -> dict:
    """对一个 prompt 采样 N 次翻译，返回频率分布"""
    provider = MODEL_PROVIDERS.get(model, "openai")
    samples = []

    for _ in range(n_samples):
        try:
            if provider == "openai":
                t = sample_openai(client_openai, english, lang, model, temperature)
            elif provider == "anthropic":
                t = sample_anthropic(client_anthropic, english, lang, model, temperature)
            else:
                raise ValueError(f"Unknown provider for model {model}")
            samples.append(t)
        except Exception as e:
            print(f"  Sampling error: {e}")
            continue

        time.sleep(rate_limit_delay)

    counter = Counter(samples)
    total = len(samples)
    p_llm = {t: count / total for t, count in counter.items()} if total > 0 else {}

    return {
        "p_llm": p_llm,
        "samples": samples,
        "unique_count": len(counter),
        "total_sampled": total,
    }


def run_sampling(lang, split, model, n_samples, temperature, max_prompts=None, rate_limit_delay=0.05):
    """主流程"""
    prompts = load_prompts(lang, split)
    if max_prompts:
        prompts = prompts[:max_prompts]

    client_openai = None
    client_anthropic = None
    provider = MODEL_PROVIDERS.get(model, "openai")

    if provider == "openai":
        from openai import OpenAI
        client_openai = OpenAI()
    elif provider == "anthropic":
        from anthropic import Anthropic
        client_anthropic = Anthropic()

    print(f"Sampling {len(prompts)} prompts, {n_samples} samples each")
    print(f"Model: {model}, Lang: {lang}, Split: {split}, Temp: {temperature}")
    print()

    results = []

    for prompt in tqdm(prompts, desc="Sampling prompts"):
        english = prompt["english"]
        gold_translations = {t["text"] for t in prompt["translations"]}

        sample_result = sample_translations(
            english, lang, model, n_samples, temperature,
            client_openai=client_openai,
            client_anthropic=client_anthropic,
            rate_limit_delay=rate_limit_delay,
        )

        # 计算 gold set 内外的覆盖
        in_gold = sum(1 for t in sample_result["p_llm"] if t in gold_translations)
        phantom_mass = sum(
            p for t, p in sample_result["p_llm"].items() if t not in gold_translations
        )

        results.append({
            "prompt_id": prompt["prompt_id"],
            "english": english,
            "p_llm": sample_result["p_llm"],
            "unique_count": sample_result["unique_count"],
            "total_sampled": sample_result["total_sampled"],
            "in_gold_count": in_gold,
            "phantom_mass": phantom_mass,
        })

    output_file = RESULTS_DIR / f"sampling_{model}_{lang}_{split}_t{temperature}_n{n_samples}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_unique = sum(r["unique_count"] for r in results) / len(results)
    avg_phantom = sum(r["phantom_mass"] for r in results) / len(results)
    avg_in_gold = sum(r["in_gold_count"] for r in results) / len(results)

    print(f"\nResults summary:")
    print(f"  Avg unique translations per prompt: {avg_unique:.1f}")
    print(f"  Avg in-gold translations: {avg_in_gold:.1f}")
    print(f"  Avg phantom mass (outside gold): {avg_phantom:.3f}")
    print(f"\nSaved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Translation Sampling Pipeline")
    parser.add_argument("--lang", default="en_pt", choices=list(LANG_NAMES.keys()))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default="gpt-4o", choices=list(MODEL_PROVIDERS.keys()))
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.05)
    args = parser.parse_args()

    run_sampling(
        args.lang, args.split, args.model,
        args.n_samples, args.temperature,
        args.max_prompts, args.rate_limit_delay,
    )


if __name__ == "__main__":
    main()
