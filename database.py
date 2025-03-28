import os
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""


def filter_complex_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Фильтрует сложные метаданные, оставляя только простые типы
    Args:
        metadata: Исходные метаданные
    Returns:
        Dict[str, Any]: Отфильтрованные метаданные
    """
    filtered = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
    return filtered


class ChromaDatabase:
    def __init__(self, persist_directory: str = "chroma"):
        """
        Инициализация базы данных
        Args:
            persist_directory: Директория для хранения базы
        """
        # Загружаем переменные окружения
        load_dotenv()

        # Проверяем наличие необходимых переменных
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

        self.embedding_function = get_embedding_function()
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )
        self.prompt_template = """Используй следующие части контекста для ответа на вопрос. Если ты не знаешь ответа, просто скажи, что не знаешь. Не пытайся придумать ответ.

Контекст: {context}

Вопрос: {question}

Ответ:"""

    def _generate_chunk_id(
        self, chunk: Document, page_id: str, chunk_index: int
    ) -> str:
        """Генерирует уникальный идентификатор для чанка"""
        return f"{chunk.metadata.get('source', '')}:{page_id}:{chunk_index}"

    def add_documents(self, documents: List[Document], chunk_size: int = 1000) -> None:
        """
        Добавление документов в базу
        Args:
            documents: Список документов
            chunk_size: Размер чанка для разбиения
        """
        # Фильтруем сложные метаданные
        for doc in documents:
            doc.metadata = filter_complex_metadata(doc.metadata)

        # Добавляем документы
        self.db.add_documents(documents)

    def get_chain(self) -> RetrievalQA:
        """
        Создание цепочки для вопросно-ответной системы
        Returns:
            RetrievalQA: Цепочка для вопросно-ответной системы
        """
        prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
        )

    def _mmr_search(
        self, query_text: str, k: int = 5, lambda_param: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Выполняет MMR поиск
        Args:
            query_text: Текст запроса
            k: Количество возвращаемых результатов
            lambda_param: Параметр баланса между релевантностью и разнообразием
        Returns:
            List[Tuple[Document, float]]: Список документов с их релевантностью
        """
        # Получаем больше результатов для MMR
        initial_k = k * 2
        results = self.db.similarity_search_with_score(query_text, k=initial_k)

        if not results:
            return []

        # Получаем эмбеддинги запроса и документов
        query_embedding = self.embedding_function.embed_query(query_text)
        doc_embeddings = [
            self.embedding_function.embed_query(doc.page_content) for doc, _ in results
        ]

        # Преобразуем в numpy массивы
        query_embedding = np.array(query_embedding).reshape(1, -1)
        doc_embeddings = np.array(doc_embeddings)

        # Вычисляем сходство между запросом и документами
        query_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Инициализируем MMR
        selected_indices = []
        remaining_indices = list(range(len(results)))

        # Выбираем первый документ с максимальной релевантностью
        first_idx = np.argmax(query_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Итеративно выбираем остальные документы
        while len(selected_indices) < k and remaining_indices:
            # Вычисляем сходство между выбранными и оставшимися документами
            selected_embeddings = doc_embeddings[selected_indices]
            remaining_embeddings = doc_embeddings[remaining_indices]

            if len(selected_indices) == 1:
                similarities = cosine_similarity(
                    selected_embeddings, remaining_embeddings
                )[0]
            else:
                similarities = np.max(
                    cosine_similarity(selected_embeddings, remaining_embeddings), axis=0
                )

            # Вычисляем MMR score
            query_similarities_remaining = query_similarities[remaining_indices]
            mmr_scores = (
                lambda_param * query_similarities_remaining
                - (1 - lambda_param) * similarities
            )

            # Выбираем документ с максимальным MMR score
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        # Формируем результат
        return [(results[i][0], results[i][1]) for i in selected_indices]

    def _combine_results(
        self,
        similarity_results: List[Tuple[Document, float]],
        mmr_results: List[Tuple[Document, float]],
        k: int,
        similarity_weight: float = 0.7,
    ) -> List[Tuple[Document, float]]:
        """
        Объединяет результаты разных методов поиска
        Args:
            similarity_results: Результаты поиска по релевантности
            mmr_results: Результаты MMR поиска
            k: Количество возвращаемых результатов
            similarity_weight: Вес для результатов поиска по релевантности
        Returns:
            List[Tuple[Document, float]]: Объединенные результаты
        """
        # Создаем словарь для хранения объединенных результатов
        combined = {}

        # Добавляем результаты поиска по релевантности
        for doc, score in similarity_results:
            combined[doc.page_content] = (doc, score * similarity_weight)

        # Добавляем результаты MMR
        for doc, score in mmr_results:
            if doc.page_content in combined:
                # Если документ уже есть, обновляем score
                existing_doc, existing_score = combined[doc.page_content]
                combined[doc.page_content] = (
                    doc,
                    existing_score + score * (1 - similarity_weight),
                )
            else:
                combined[doc.page_content] = (doc, score * (1 - similarity_weight))

        # Сортируем по score и берем top-k
        sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def query(
        self,
        query_text: str,
        k: int = 5,
        min_score: float = 0.5,
        search_type: str = "similarity",
        lambda_param: float = 0.5,
        similarity_weight: float = 0.7,
    ) -> str:
        """
        Выполняет запрос к базе данных с различными стратегиями поиска
        Args:
            query_text: Текст запроса
            k: Количество возвращаемых результатов
            min_score: Минимальный порог релевантности
            search_type: Тип поиска ('similarity', 'mmr', 'hybrid')
            lambda_param: Параметр баланса для MMR
            similarity_weight: Вес для similarity поиска в hybrid режиме
        """
        if search_type == "similarity":
            results = self.db.similarity_search_with_score(query_text, k=k)
        elif search_type == "mmr":
            results = self._mmr_search(query_text, k=k, lambda_param=lambda_param)
        else:  # hybrid
            # Получаем результаты обоих методов
            similarity_results = self.db.similarity_search_with_score(query_text, k=k)
            mmr_results = self._mmr_search(query_text, k=k, lambda_param=lambda_param)

            # Объединяем результаты
            results = self._combine_results(
                similarity_results,
                mmr_results,
                k=k,
                similarity_weight=similarity_weight,
            )

        # Фильтруем по релевантности
        filtered_results = [
            (doc, score) for doc, score in results if score >= min_score
        ]

        if not filtered_results:
            return "Не найдено релевантных результатов"

        # Формируем контекст из найденных документов
        context = "\n\n".join([doc.page_content for doc, _ in filtered_results])

        # Создаем цепочку и получаем ответ
        chain = self.get_chain()
        response = chain.invoke({"query": query_text})

        return response["result"]

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые слова из текста"""
        # Простая реализация - берем первые 5 слов
        return text.split()[:5]

    def _calculate_importance(self, text: str) -> str:
        """Определяет важность текста"""
        # Простая реализация - на основе длины и наличия ключевых слов
        important_words = ["важно", "критично", "существенно", "необходимо"]
        text_lower = text.lower()
        if any(word in text_lower for word in important_words):
            return "high"
        elif len(text) > 500:
            return "medium"
        return "low"

    def _detect_content_type(self, text: str) -> str:
        """Определяет тип контента"""
        # Простая реализация - на основе характерных признаков
        if any(char.isdigit() for char in text):
            return "data"
        elif len(text.split()) > 100:
            return "article"
        return "note"


def main():
    db = ChromaDatabase()

    while True:
        query = input("\nВведите ваш вопрос (или 'выход' для завершения): ")
        if query.lower() == "выход":
            break

        try:
            # Метод query уже выводит фрагменты и возвращает ответ
            response = db.query(query)
            print("\nОтвет:", response)
        except Exception as e:
            print(f"\nПроизошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()
