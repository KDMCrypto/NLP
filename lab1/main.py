import nltk
import pymorphy3

for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

from nltk.tokenize import sent_tokenize, word_tokenize

morph = pymorphy3.MorphAnalyzer()

TARGET_POS = {"NOUN", "ADJF", "ADJS"}

def get_target_parse(token):
    """
    Функция проверки на сущ или прил, если токен относится к одному из них, то возвращает морфологический разбор токена.
    """
    for parse in morph.parse(token):
        # Помогает пропускать ситуации типа: "наш инстинкт", так как "наш" это местоимение 
        if 'Apro' in parse.tag: 
            continue

        if parse.tag.POS in TARGET_POS:
            return parse
    return None

def have_agreement(left, right):
    """
    Проверка на строгое совпадение числа и падежа.
    Род проверяется только в единственном числе (так как во мн.ч. рода нет).
    """
    # Если у какого-то слова нет числа или падежа (например, у неизменяемых), не сопоставляем
    if left.tag.number is None or right.tag.number is None:
        return False
    if left.tag.case is None or right.tag.case is None:
        return False

    # Число и падеж должны совпадать
    if left.tag.number != right.tag.number:
        return False
    if left.tag.case != right.tag.case:
        return False

    # Проверяем совпадение рода
    left_gender = left.tag.gender
    right_gender = right.tag.gender
    if left_gender is not None and right_gender is not None:
        if left.tag.gender != right.tag.gender:
            return False
                
    return True

def process_text(text):
    for sentence in sent_tokenize(text, language="russian"):
        tokens = word_tokenize(sentence, language="russian")

        parsed_tokens = [get_target_parse(token) for token in tokens]

        for i in range(len(parsed_tokens) - 1):
            left = parsed_tokens[i]
            right = parsed_tokens[i + 1]

            if left is None or right is None:
                continue

            if have_agreement(left, right):
                print(f"{left.normal_form.lower()} {right.normal_form.lower()}")

if __name__ == "__main__":
    file_path = r"text/1.txt"

    with open(file_path, encoding="utf-8") as file:
        text = file.read()

    process_text(text)