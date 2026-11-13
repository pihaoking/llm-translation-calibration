"""
LLM Scoring 管线 (路线 A)。

对 STAPLE 中每个 prompt 的所有已知翻译，计算 LLM 的条件 log-probability，
然后做长度归一化 + softmax，得到 P_LLM(t|s)。

支持:
  - OpenAI GPT-4o (via API, logprobs)
  - 本地开源模型 (via HuggingFace transformers) [预留接口]

用法:
  python src/llm_scoring.py --lang en_pt --split dev --model gpt-4o --max_prompts 10
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
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


def load_prompts(lang, split):
    with open(DATA_DIR / f"{lang}_{split}.json", "r") as f:
        return json.load(f)


def build_messages(english: str, lang: str):
    """构建翻译 prompt"""
    target_lang = LANG_NAMES[lang]
    return [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": f"Translate the following English sentence to {target_lang}.\n\n{english}"},
    ]


def score_translations_openai(
    client: OpenAI,
    english: str,
    translations: list[dict],
    lang: str,
    model: str = "gpt-4o",
    batch_size: int = 1,
) -> dict:
    """
    用 OpenAI API 对每条翻译打分。

    通过让模型在给定 prompt 后生成翻译文本，利用 logprobs 获取每个 token 的对数概率。

    Returns:
        {translation_text: normalized_logprob, ...}
    """
    scores = {}
    messages = build_messages(english, lang)

    for t in translations:
        text = t["text"]
        try:
            # 使用 completion API 打分: 给定 prompt，让模型"生成"已知翻译
            # 通过 logprobs 获取每个 token 的概率
            response = client.chat.completions.create(
                model=model,
                messages=messages + [{"role": "assistant", "content": text}],
                max_tokens=1,
                logprobs=True,
                temperature=0,
            )

            # 从 logprobs 中提取 assistant message 的 token 概率
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                # 注意: 这种方式只能拿到续写 token 的 logprob
                # 更好的方式是使用 Completions API (非 Chat)
                # 这里使用一种替代方案: 直接用 echo 模式
                pass

        except Exception as e:
            print(f"  Error scoring '{text[:30]}...': {e}")
            continue

    return scores


def score_translations_openai_batch(
    client: OpenAI,
    english: str,
    translations: list[dict],
    lang: str,
    model: str = "gpt-4o",
) -> dict:
    """
    替代方案: 用 OpenAI 的 completions API 计算翻译的 logprob。

    思路: 将 system + user + assistant(translation) 拼成完整序列,
    然后用 model 的 logprobs 来估计翻译质量。

    实际实现: 我们让模型生成翻译, 同时开启 logprobs,
    通过生成概率间接估计各翻译的可能性。

    更实用的方案: 对每条翻译, 计算 prompt + translation 的 log-likelihood。
    使用 /v1/chat/completions 的 logprobs 功能。
    """
    scores = {}
    target_lang = LANG_NAMES[lang]
    system_msg = "You are a translator."
    user_msg = f"Translate the following English sentence to {target_lang}.\n\n{english}"

    for t in translations:
        text = t["text"]
        try:
            # 使用 predicted_output 来获取完整的 logprobs
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=len(text.split()) * 3 + 20,  # 足够长
                logprobs=True,
                temperature=0,
                # 强制模型输出指定文本
                prediction={"type": "content", "content": text},
            )

            # 提取 logprobs
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                token_logprobs = [
                    lp.logprob
                    for lp in response.choices[0].logprobs.content
                ]
                total_logprob = sum(token_logprobs)
                n_tokens = len(token_logprobs)
                # 长度归一化
                normalized = total_logprob / n_tokens if n_tokens > 0 else -100
                scores[text] = normalized
            else:
                scores[text] = -100

        except Exception as e:
            print(f"  Error scoring '{text[:30]}...': {e}")
            scores[text] = -100

    return scores


def logprob_to_probability(scores: dict) -> dict:
    """将 log-probability scores 转换为概率分布 (softmax)"""
    if not scores:
        return {}

    texts = list(scores.keys())
    logprobs = np.array([scores[t] for t in texts])

    # softmax
    logprobs = logprobs - logprobs.max()  # 数值稳定性
    probs = np.exp(logprobs)
    probs = probs / probs.sum()

    return {t: float(p) for t, p in zip(texts, probs)}


def run_scoring(lang, split, model, max_prompts=None, rate_limit_delay=0.1):
    """主流程: 对指定语言对和数据集跑 scoring"""
    prompts = load_prompts(lang, split)
    if max_prompts:
        prompts = prompts[:max_prompts]

    client = OpenAI()  # 使用环境变量 OPENAI_API_KEY

    results = []
    total_translations = sum(len(p["translations"]) for p in prompts)
    print(f"Scoring {len(prompts)} prompts, {total_translations} translations")
    print(f"Model: {model}, Lang: {lang}, Split: {split}")
    print()

    for i, prompt in enumerate(tqdm(prompts, desc="Scoring prompts")):
        english = prompt["english"]
        translations = prompt["translations"]

        # 打分
        scores = score_translations_openai_batch(
            client, english, translations, lang, model
        )

        # 转为概率
        p_llm = logprob_to_probability(scores)

        # 保存
        results.append({
            "prompt_id": prompt["prompt_id"],
            "english": english,
            "p_llm": p_llm,
            "raw_scores": scores,
            "num_scored": len([s for s in scores.values() if s > -100]),
            "num_total": len(translations),
        })

        # Rate limiting
        time.sleep(rate_limit_delay)

    # 保存结果
    output_file = RESULTS_DIR / f"scoring_{model}_{lang}_{split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Translation Scoring Pipeline")
    parser.add_argument("--lang", default="en_pt", choices=list(LANG_NAMES.keys()))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.1,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    run_scoring(args.lang, args.split, args.model, args.max_prompts, args.rate_limit_delay)


if __name__ == "__main__":
    main()
