import os
import re
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers

transformers.logging.set_verbosity_error()

np.random.seed(42)
torch.manual_seed(42)


def load_tokenizer_and_model(model_name_or_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return tokenizer, model


def generate(
    model, tok, text,
    do_sample=True, max_length=100, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
):
    input_ids = tok.encode(text, return_tensors="pt").to(model.device)

    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size
    )

    return list(map(tok.decode, out))


def contains_words_in_order(text, words):
    # Функция проверяет, что необходимые слова встречаются в результате именно в заданном порядке и именно в такой форме.
    current_pos = 0

    for word in words:
        pattern = re.compile(r"\b" + word + r"\b")
        match = pattern.search(text, current_pos)

        if match is None:
            return False

        current_pos = match.end()

    return True


if __name__ == "__main__":
    model_name = "sberbank-ai/rugpt3large_based_on_gpt2"

    tok, model = load_tokenizer_and_model(model_name)
    prompt = (
        "Офтальмологи предупреждают, что зрение портится не только за персональным"
    )

    required_words = ["компьютером", "телевизором"]

    generated = generate(
        model,
        tok,
        prompt,
        do_sample=False,
        max_length=60,
        repetition_penalty=1.3,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        num_beams=20,
        no_repeat_ngram_size=3
    )

    result_text = generated[0]

    lines = [
        "Запрос:",
        prompt,
        "",
        "Результат:",
        result_text,
        "",
        "Проверка:", contains_words_in_order(result_text, required_words)
    ]

    os.makedirs("result", exist_ok=True)

    with open("result/result.txt", "w", encoding="utf-8") as f:
        for line in lines:
            print(line)
            print(line, file=f)