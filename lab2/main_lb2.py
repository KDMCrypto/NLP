import os
import gensim

def check_candidates(model, candidates, target1, target2):
    '''Функция перебирает всех кандидатов и пытается найти такую комбинацию, в которой целевые слова попадут в десятку'''
    # Проверяем одиночные слова и суммы векторов
    for i in range(len(candidates)):
        w1 = candidates[i]

        top_words = [x[0] for x in model.most_similar(positive=[w1], topn=10)]
        if target1 in top_words and target2 in top_words:
            return True, f"positive=['{w1}']", top_words

        for j in range(i + 1, min(i + 100, len(candidates))):
            w2 = candidates[j]

            top_words = [
                x[0] for x in model.most_similar(
                    positive=[w1, w2],
                    topn=10
                )
            ]

            if target1 in top_words and target2 in top_words:
                return True, f"positive=['{w1}', '{w2}']", top_words

    # Проверяем разности векторов
    for i in range(len(candidates)):
        w1 = candidates[i]

        for j in range(len(candidates)):
            if i == j:
                continue

            w2 = candidates[j]

            top_words = [
                x[0] for x in model.most_similar(
                    positive=[w1],
                    negative=[w2],
                    topn=10
                )
            ]

            if target1 in top_words and target2 in top_words:
                return True, f"positive=['{w1}'], negative=['{w2}']", top_words

    return False, "", []


def main():
    model_path = r"data/cbow.txt"

    if not os.path.isfile(model_path):
        print(f"Файл модели не найден: {model_path}, положите cbow.txt в папку data.")
        return
    
    # Загружаем модель
    full_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)

    # Создаем отдельную модель только с существительными
    noun_keys = [w for w in full_model.index_to_key if w.endswith('_NOUN')]
    word2vec = gensim.models.KeyedVectors(vector_size=full_model.vector_size)
    word2vec.add_vectors(noun_keys, full_model[noun_keys])

    # Выданные слова
    target1 = "безделушка_NOUN"
    target2 = "комод_NOUN"

    # Вычисляем средний вектор двух целевых слов
    avg_vector = (word2vec[target1] + word2vec[target2]) / 2.0

    # Отбираем кандидатов - существительные, близкие к среднему вектору
    closest_words = word2vec.similar_by_vector(avg_vector, topn=300)
    candidates = [
        w for w, _ in closest_words
        if w not in (target1, target2)
    ]

    # Перебор кандидатов
    found, query, top_words = check_candidates(
        word2vec,
        candidates,
        target1,
        target2
    )

    if found:
        print("Комбинация найдена: ", query)

        with open("result/top_10.txt", "w", encoding="utf-8") as f:
            f.write(f"Комбинация: {query}\n\n")

            for i, w in enumerate(top_words[:10], start=1):
                clean_word = w[:-5]
                print(f"{i}. {clean_word}")
                f.write(f"{i}. {clean_word}\n")
    else:
        print("Комбинация не найдена")

if __name__ == "__main__":
    main()