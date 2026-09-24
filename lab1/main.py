import nltk
import pymorphy3

for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

from nltk.tokenize import sent_tokenize, word_tokenize

morph = pymorphy3.MorphAnalyzer()

# Целевые части речи существительные и прилагательные
TARGET_POS = {"NOUN", "ADJF"}

# Множество служебных слов
FUNCTION_WORDS = {
    # союзы
    "и", "а", "но", "да", "или", "либо", "что", "как", "чтобы", "если",
    "тоже", "также", "зато", "однако", "причём", "притом",
    # предлоги
    "с", "со", "по", "в", "во", "на", "к", "ко", "у", "о", "об", "обо",
    "от", "до", "за", "из", "без", "для", "над", "под", "при", "про",
    "через", "между", "перед", "около", "возле", "мимо", "вдоль",
    "вокруг", "среди", "кроме", "насчёт", "вместо", "благодаря",
    # частицы
    "не", "ни", "же", "ли", "бы", "ведь", "вот", "вон", "даже", "именно",
    "лишь", "только", "разве", "неужели",
    # местоимения-существительные
    "это", "то", "всё", "все", "он", "она", "оно", "они", "мы", "вы",
    "я", "ты", "себя", "себе",
}

def get_candidate_parses(token):
    """Функция возвращает все подходящие морфологические разборы токена"""
    # Сразу отбрасываем служебные слова
    if token.lower() in FUNCTION_WORDS:
        return []

    candidates = []

    for parse in morph.parse(token):
        # Пропускаем местоименные прилагательные: наш, этот, каждый и тд
        if "Apro" in parse.tag:
            continue

        # Оставляем только нужные части речи
        if parse.tag.POS not in TARGET_POS:
            continue

        # Дополнительно отсекаем служебные слова по лемме
        if parse.normal_form.lower() in FUNCTION_WORDS:
            continue

        candidates.append(parse)

    return candidates


def have_agreement(left_parse, right_parse):
    """Функция для проверки грамматической согласованности двух морфологических разборов по числу, падежу и роду"""
    left_tag = left_parse.tag
    right_tag = right_parse.tag

    # Проверяем, что слова имеют определенные число и падеж, исключаем неизменяемые слова
    if left_tag.number is None or right_tag.number is None:
        return False
    if left_tag.case is None or right_tag.case is None:
        return False

    # Проверка на строгое совпадение числа и падежа
    if left_tag.number != right_tag.number:
        return False
    if left_tag.case != right_tag.case:
        return False

    # Проверяем существование рода и его согласованность
    if left_tag.gender is not None and right_tag.gender is not None:
        if left_tag.gender != right_tag.gender:
            return False

    return True


def find_best_pair(left_candidates, right_candidates):
    """Функция находит пару морфологических разборов с максимальным суммарным весом, удовлетворяющую согласованию"""
    best_score = -1
    best_pair = None

    # Перебираем все возможные комбинации морфологических разборов левого и правого слов
    for left_parse in left_candidates:
        for right_parse in right_candidates:
            if have_agreement(left_parse, right_parse):
                # Суммируем веса слов для оценки вероятности пары
                current_score = left_parse.score + right_parse.score

                # Выбираем комбинацию с наивысшим суммарным весом
                if current_score > best_score:
                    best_score = current_score
                    best_pair = (
                        left_parse.normal_form.lower(),
                        right_parse.normal_form.lower(),
                    )

    return best_pair


def process_text(text, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as out_file:
        # Разбиваем текст на предложения, а затем на отдельные токены
        for sentence in sent_tokenize(text, language="russian"):
            tokens = word_tokenize(sentence, language="russian")
            
            # Получаем списки подходящих морфологических разборов для каждого токена
            parsed_tokens = [get_candidate_parses(token) for token in tokens]

            # Проходим по всем парам соседних токенов в предложении
            for i in range(len(parsed_tokens) - 1):
                left_candidates = parsed_tokens[i]
                right_candidates = parsed_tokens[i + 1]

                # Пропускаем пару, если хотя бы одно слово не является сущ или прил
                if not left_candidates or not right_candidates:
                    continue

                # Ищем лучшую согласованную пару среди всех вариантов
                best_pair = find_best_pair(left_candidates, right_candidates)

                if best_pair is not None:
                    left_lemma, right_lemma = best_pair
                    out_file.write(f"{left_lemma} {right_lemma}\n")


if __name__ == "__main__":
    input_file_path = r"text/1.txt"
    output_file_path = r"result/result_1.txt"

    with open(input_file_path, encoding="utf-8") as file:
        text = file.read()

    process_text(text, output_file_path)