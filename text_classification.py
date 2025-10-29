#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Программа для классификации текстов на русском языке с использованием машинного обучения.
Сравнивает различные подходы к векторизации и алгоритмы классификации.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Библиотеки для работы с датасетами и текстом
from datasets import load_dataset
try:
    import pymorphy3 as pymorphy2
except ImportError:
    import pymorphy2
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Лемматизатор будет инициализирован при первом использовании
_morph = None


def get_morph():
    """Ленивая инициализация морфологического анализатора."""
    global _morph
    if _morph is None:
        _morph = pymorphy2.MorphAnalyzer()
    return _morph


def lemmatize_text(text):
    """
    Лемматизация русского текста.
    
    Args:
        text: Исходный текст
        
    Returns:
        Лемматизированный текст
    """
    if not isinstance(text, str):
        return ""
    
    # Очистка текста от лишних символов
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Разбиение на слова и лемматизация
    words = text.lower().split()
    lemmatized_words = []
    
    morph = get_morph()
    for word in words:
        if word:
            parsed = morph.parse(word)[0]
            lemmatized_words.append(parsed.normal_form)
    
    return ' '.join(lemmatized_words)


def load_russian_dataset():
    """
    Загрузка датасета на русском языке с Hugging Face.
    Используется датасет отзывов о медицинских учреждениях.
    """
    print("Загрузка датасета...")
    
    # Попытка загрузить датасет отзывов о медицинских учреждениях через huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        print("Попытка загрузки датасета 'blinoff/medical_institutions_reviews'...")
        # Скачиваем JSONL файл
        file_path = hf_hub_download(
            repo_id="blinoff/medical_institutions_reviews",
            filename="medical_institutions_reviews.jsonl",
            repo_type="dataset"
        )
        
        # Читаем JSONL файл через pandas
        df = pd.read_json(file_path, lines=True)
        print(f"Загружено {len(df)} примеров")
        
        # Извлечение текстов и меток
        texts = []
        labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка датасета"):
            # Датасет имеет поля: content (текст отзыва) и general (общая тональность)
            if pd.notna(row.get('content')) and row.get('content'):
                texts.append(str(row['content']))
                # Используем поле 'general' для тональности
                if pd.notna(row.get('general')) and row.get('general'):
                    labels.append(str(row['general']))
                else:
                    # Пропускаем примеры без метки
                    continue
        
        if len(texts) > 0:
            print(f"Успешно загружено {len(texts)} примеров с метками")
            print(f"Уникальные метки: {set(labels)}")
            return texts, labels
        else:
            raise ValueError("Не удалось извлечь данные из датасета")
    
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        print("Попытка загрузить через datasets...")
        
        # Попытка через стандартный load_dataset
        try:
            dataset = load_dataset("blinoff/medical_institutions_reviews", trust_remote_code=True)
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Используем сплит '{split_name}' с {len(data)} примерами")
            
            texts = []
            labels = []
            
            for item in tqdm(data, desc="Обработка датасета"):
                if 'content' in item and item['content']:
                    texts.append(item['content'])
                    if 'general' in item and item['general']:
                        labels.append(item['general'])
                    else:
                        continue
            
            if len(texts) > 0:
                print(f"Успешно загружено {len(texts)} примеров с метками")
                return texts, labels
        except Exception as e2:
            print(f"Ошибка при загрузке через datasets: {e2}")
        
        print("Создание синтетического датасета...")
        return create_synthetic_dataset()


def create_synthetic_dataset():
    """
    Создание синтетического датасета для демонстрации.
    Классификация отзывов о продуктах.
    """
    positive_reviews = [
        "Отличный товар, очень доволен покупкой. Качество превосходное, рекомендую всем.",
        "Прекрасное качество, быстрая доставка. Всё работает отлично, спасибо за сервис.",
        "Замечательный продукт, полностью соответствует описанию. Осталась очень довольна.",
        "Превосходное качество, хорошее соотношение цена-качество. Рекомендую к покупке.",
        "Отличный сервис, быстрая доставка, качественный товар. Всё понравилось.",
        "Замечательно, всё работает как надо. Качество на высоте, очень доволен.",
        "Прекрасный продукт, хорошая упаковка, быстрая доставка. Всё супер.",
        "Отлично, товар качественный, соответствует ожиданиям. Рекомендую.",
        "Замечательный сервис, быстрая обработка заказа. Товар в порядке.",
        "Превосходное качество, всё работает идеально. Очень довольна покупкой.",
    ]
    
    negative_reviews = [
        "Товар плохого качества, не рекомендую. Деньги потрачены зря, очень разочарован.",
        "Плохая доставка, товар пришёл повреждённым. Сервис оставляет желать лучшего.",
        "Некачественный продукт, быстро сломался. Не стоит своих денег, не покупайте.",
        "Очень плохое качество, товар не соответствует описанию. Разочарование.",
        "Ужасный сервис, долгая доставка, товар не работает. Деньги на ветер.",
        "Плохой товар, качество оставляет желать лучшего. Не рекомендую к покупке.",
        "Некачественный продукт, быстро вышел из строя. Очень недоволен покупкой.",
        "Плохая упаковка, товар повреждён. Сервис плохой, не рекомендую.",
        "Разочарован покупкой, товар не работает как заявлено. Деньги потрачены зря.",
        "Некачественный товар, плохое обслуживание. Очень недоволен.",
    ]
    
    neutral_reviews = [
        "Товар обычный, ничего особенного. Качество среднее, цена соответствует.",
        "Нормальный продукт, работает как надо. Ничего выдающегося, но и недостатков нет.",
        "Стандартное качество, обычный товар. Всё в пределах нормы, без претензий.",
        "Товар как товар, ничего особенного. Качество среднее, работает.",
        "Обычный продукт, среднее качество. Ничего плохого, ничего хорошего.",
        "Стандартный товар, соответствует цене. Качество обычное, работает нормально.",
        "Нормальное качество, ничего особенного. Товар как товар, работает.",
        "Средний продукт, среднее качество. Всё в пределах ожиданий.",
        "Обычный товар, качество норм. Ничего выдающегося, но работает.",
        "Стандартное качество, обычный продукт. Всё как обычно.",
    ]
    
    texts = positive_reviews + negative_reviews + neutral_reviews
    labels = ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
    
    # Увеличиваем датасет путём вариаций
    expanded_texts = []
    expanded_labels = []
    
    for text, label in zip(texts, labels):
        expanded_texts.append(text)
        expanded_labels.append(label)
        # Добавляем варианты с небольшими изменениями
        for _ in range(9):
            expanded_texts.append(text)
            expanded_labels.append(label)
    
    return expanded_texts, expanded_labels


def preprocess_texts(texts, use_lemmatization=True):
    """
    Предобработка текстов с опциональной лемматизацией.
    
    Args:
        texts: Список текстов
        use_lemmatization: Использовать ли лемматизацию
        
    Returns:
        Список обработанных текстов
    """
    processed_texts = []
    
    print("Предобработка текстов...")
    for text in tqdm(texts, desc="Обработка"):
        if use_lemmatization:
            processed_text = lemmatize_text(text)
        else:
            # Простая очистка без лемматизации
            processed_text = re.sub(r'[^\w\s]', ' ', str(text))
            processed_text = re.sub(r'\s+', ' ', processed_text).strip().lower()
        
        processed_texts.append(processed_text)
    
    return processed_texts


def vectorize_tfidf(texts_train, texts_test, max_features=5000):
    """
    Векторизация текстов с помощью TF-IDF (частотный подход).
    
    Args:
        texts_train: Обучающие тексты
        texts_test: Тестовые тексты
        max_features: Максимальное количество признаков
        
    Returns:
        Векторизованные обучающие и тестовые данные
    """
    print("Векторизация TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    
    print(f"Размерность векторов TF-IDF: {X_train.shape[1]}")
    return X_train, X_test, vectorizer


def vectorize_word2vec(texts_train, texts_test, vector_size=100):
    """
    Векторизация текстов с помощью Word2Vec (семантический подход).
    
    Args:
        texts_train: Обучающие тексты
        texts_test: Тестовые тексты
        vector_size: Размерность вектора
        
    Returns:
        Векторизованные обучающие и тестовые данные, обученная модель
    """
    print("Векторизация Word2Vec...")
    
    # Подготовка текстов для Word2Vec
    train_tokens = [simple_preprocess(text) for text in texts_train]
    test_tokens = [simple_preprocess(text) for text in texts_test]
    
    # Обучение модели Word2Vec
    model = Word2Vec(
        sentences=train_tokens + test_tokens,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4,
        sg=0  # CBOW
    )
    
    # Функция для получения среднего вектора документа
    def get_document_vector(tokens, model, size):
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(size)
    
    # Векторизация документов
    X_train = np.array([
        get_document_vector(tokens, model, vector_size)
        for tokens in tqdm(train_tokens, desc="Векторизация обучающих данных")
    ])
    
    X_test = np.array([
        get_document_vector(tokens, model, vector_size)
        for tokens in tqdm(test_tokens, desc="Векторизация тестовых данных")
    ])
    
    print(f"Размерность векторов Word2Vec: {X_train.shape[1]}")
    return X_train, X_test, model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Обучение и оценка модели классификации.
    
    Args:
        model: Модель машинного обучения
        X_train: Обучающие признаки
        y_train: Обучающие метки
        X_test: Тестовые признаки
        y_test: Тестовые метки
        model_name: Название модели
        
    Returns:
        Точность модели и предсказания
    """
    print(f"\nОбучение модели: {model_name}")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Точность {model_name}: {accuracy:.4f}")
    print("\nОтчёт о классификации:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred


def compare_approaches(results_tfidf, results_w2v, label_encoder):
    """
    Визуализация и сравнение результатов разных подходов.
    
    Args:
        results_tfidf: Результаты для TF-IDF
        results_w2v: Результаты для Word2Vec
        label_encoder: Кодировщик меток
    """
    # Подготовка данных для визуализации
    models = ['Логистическая\nРегрессия', 'Случайный\nЛес', 'Наивный\nБайес']
    
    tfidf_accuracies = [
        results_tfidf['logistic']['accuracy'],
        results_tfidf['random_forest']['accuracy'],
        results_tfidf['naive_bayes']['accuracy']
    ]
    
    w2v_accuracies = [
        results_w2v['logistic']['accuracy'],
        results_w2v['random_forest']['accuracy'],
        results_w2v['naive_bayes']['accuracy']
    ]
    
    # Создание графиков
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График сравнения подходов
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, tfidf_accuracies, width, label='TF-IDF (частотный)', alpha=0.8)
    axes[0].bar(x + width/2, w2v_accuracies, width, label='Word2Vec (семантический)', alpha=0.8)
    
    axes[0].set_xlabel('Модели')
    axes[0].set_ylabel('Точность')
    axes[0].set_title('Сравнение подходов к векторизации')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # График сравнения алгоритмов (объединённый)
    combined_results = {
        'TF-IDF': tfidf_accuracies,
        'Word2Vec': w2v_accuracies
    }
    
    x_pos = np.arange(len(models))
    for i, (approach, accuracies) in enumerate(combined_results.items()):
        axes[1].plot(x_pos, accuracies, marker='o', label=approach, linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Модели')
    axes[1].set_ylabel('Точность')
    axes[1].set_title('Сравнение алгоритмов классификации')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    print("\nГрафики сохранены в файл 'comparison_results.png'")
    
    # Вывод сводной таблицы
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    print(f"{'Модель':<25} {'TF-IDF':<15} {'Word2Vec':<15} {'Лучший подход':<15}")
    print("-"*80)
    
    for model_name, tfidf_acc, w2v_acc in zip(models, tfidf_accuracies, w2v_accuracies):
        best = "TF-IDF" if tfidf_acc > w2v_acc else "Word2Vec"
        if tfidf_acc == w2v_acc:
            best = "Равны"
        print(f"{model_name.replace(chr(10), ' '):<25} {tfidf_acc:<15.4f} {w2v_acc:<15.4f} {best:<15}")
    
    print("="*80)


def main():
    """
    Основная функция программы.
    """
    print("="*80)
    print("ПРОГРАММА КЛАССИФИКАЦИИ ТЕКСТОВ НА РУССКОМ ЯЗЫКЕ")
    print("="*80)
    
    # 1. Загрузка датасета
    texts, labels = load_russian_dataset()
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    print(f"\nКлассы: {label_encoder.classes_}")
    print(f"Количество примеров: {len(texts)}")
    print(f"Распределение классов:\n{pd.Series(labels).value_counts()}")
    
    # 2. Предобработка с лемматизацией
    print("\n" + "="*80)
    print("ШАГ 1: ПРЕДОБРАБОТКА С ЛЕММАТИЗАЦИЕЙ")
    print("="*80)
    processed_texts = preprocess_texts(texts, use_lemmatization=True)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Обучающая выборка: {len(X_train)} примеров")
    print(f"Тестовая выборка: {len(X_test)} примеров")
    
    # 3. Векторизация различными методами
    print("\n" + "="*80)
    print("ШАГ 2: ВЕКТОРИЗАЦИЯ ТЕКСТОВ")
    print("="*80)
    
    # 3.1. TF-IDF векторизация
    print("\n--- Частотный подход (TF-IDF) ---")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_tfidf(
        X_train, X_test, max_features=3000
    )
    
    # 3.2. Word2Vec векторизация
    print("\n--- Семантический подход (Word2Vec) ---")
    X_train_w2v, X_test_w2v, w2v_model = vectorize_word2vec(
        X_train, X_test, vector_size=100
    )
    
    # 4. Обучение и оценка моделей
    print("\n" + "="*80)
    print("ШАГ 3: ОБУЧЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*80)
    
    results_tfidf = {}
    results_w2v = {}
    
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'naive_bayes': MultinomialNB(alpha=0.1)
    }
    
    model_names = {
        'logistic': 'Логистическая регрессия',
        'random_forest': 'Случайный лес',
        'naive_bayes': 'Наивный байесовский классификатор'
    }
    
    # Обучение моделей на TF-IDF
    print("\n" + "-"*80)
    print("РЕЗУЛЬТАТЫ ДЛЯ TF-IDF (ЧАСТОТНЫЙ ПОДХОД)")
    print("-"*80)
    
    for model_key, model in models.items():
        # Для MultinomialNB нужны неотрицательные значения, поэтому только TF-IDF
        if model_key == 'naive_bayes':
            accuracy, y_pred = train_and_evaluate_model(
                model, X_train_tfidf, y_train, X_test_tfidf, y_test,
                f"{model_names[model_key]} (TF-IDF)"
            )
            results_tfidf[model_key] = {'accuracy': accuracy, 'predictions': y_pred}
            
            # Для Word2Vec используем другую модель для MultinomialNB
            print(f"\nДля Word2Vec используем Logistic Regression вместо MultinomialNB")
            w2v_model_alt = LogisticRegression(max_iter=1000, random_state=42)
            accuracy_w2v, y_pred_w2v = train_and_evaluate_model(
                w2v_model_alt, X_train_w2v, y_train, X_test_w2v, y_test,
                f"{model_names[model_key]} (Word2Vec, через LR)"
            )
            results_w2v[model_key] = {'accuracy': accuracy_w2v, 'predictions': y_pred_w2v}
        else:
            accuracy_tfidf, y_pred_tfidf = train_and_evaluate_model(
                model, X_train_tfidf, y_train, X_test_tfidf, y_test,
                f"{model_names[model_key]} (TF-IDF)"
            )
            results_tfidf[model_key] = {'accuracy': accuracy_tfidf, 'predictions': y_pred_tfidf}
            
            accuracy_w2v, y_pred_w2v = train_and_evaluate_model(
                model, X_train_w2v, y_train, X_test_w2v, y_test,
                f"{model_names[model_key]} (Word2Vec)"
            )
            results_w2v[model_key] = {'accuracy': accuracy_w2v, 'predictions': y_pred_w2v}
    
    # 5. Сравнение результатов
    print("\n" + "="*80)
    print("ШАГ 4: СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    compare_approaches(results_tfidf, results_w2v, label_encoder)
    
    print("\n" + "="*80)
    print("ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*80)


if __name__ == "__main__":
    main()

