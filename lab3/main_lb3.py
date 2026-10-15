import os
import torch
from transformers import BertTokenizer, BertForMaskedLM

model_name = "DeepPavlov/rubert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)
model.eval()

target_words = {"голос", "правда"}

templates = [
    "В его словах слышится [MASK], ведь он никогда не лжёт.",
]

def get_top10(sentence):
    text = sentence.replace("[MASK]", tokenizer.mask_token)
    inputs = tokenizer(text, return_tensors="pt")
    
    # Поиск индекса токена [MASK]
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

    with torch.no_grad():
        outputs = model(**inputs)

    # Вычисление вероятностей и выбор топ-10
    logits = outputs.logits[0, mask_index, :]
    probs = torch.softmax(logits, dim=-1)
    top_ids = torch.topk(probs, 10).indices.tolist()

    top_words = []
    for tid in top_ids:
        # Очистка декодированных токенов
        word = tokenizer.decode([tid]).strip().lower().replace("##", "")
        top_words.append(word)
        
    return top_words

found = False
result = ""

for sent in templates:
    top = get_top10(sent)
    if target_words.issubset(set(top)):
        found = True
        result += f"Предложение: {sent}\n\nТоп-10:\n"
        for i, word in enumerate(top, 1):
            result += f"{i}. {word}\n"
        break

if not found:
    result = "Не найдено предложения, где оба слова в топ-10.\n\n"
    for sent in templates:
        top = get_top10(sent)
        result += f"Предложение: {sent}\n\nТоп-10:\n"
        for i, word in enumerate(top, 1):
            result += f"{i}. {word}\n"
        result += "\n"

os.makedirs("result", exist_ok=True)

with open("result/result.txt", "w", encoding="utf-8") as f:
    f.write(result)

print(result)