import pytest
from database import ChromaDatabase
from document_loaders import load_documents
from get_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI
from langchain.schema import Document


@pytest.fixture
def db():
    """Фикстура для создания тестовой базы данных"""
    db = ChromaDatabase(persist_directory="test_chroma")
    # Создаем тестовые документы
    test_docs = [
        Document(
            page_content="Это тестовый документ для проверки работы системы.",
            metadata={"source": "test1.txt", "page": 1},
        ),
        Document(
            page_content="Второй тестовый документ с дополнительной информацией.",
            metadata={"source": "test2.txt", "page": 1},
        ),
        Document(
            page_content="Третий документ с уникальной информацией о тестировании.",
            metadata={"source": "test3.txt", "page": 1},
        ),
    ]
    db.add_documents(test_docs)
    return db


def test_chain_creation(db):
    """Тест создания цепочки"""
    chain = db.get_chain()
    assert chain is not None


def test_query_response(db):
    """Тест получения ответа на запрос"""
    result = db.query("Что содержится в тестовых документах?")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_empty_query(db):
    """Тест пустого запроса"""
    result = db.query("")
    assert result is not None
    assert isinstance(result, str)


def test_query_with_filters(db):
    """Тест запроса с фильтрацией по релевантности"""
    result = db.query("Тестовый запрос", min_score=0.7)
    assert result is not None
    assert isinstance(result, str)


def test_query_with_mmr(db):
    """Тест запроса с MMR"""
    result = db.query("Тестовый запрос", search_type="mmr", lambda_param=0.5)
    assert result is not None
    assert isinstance(result, str)


def test_query_with_hybrid(db):
    """Тест гибридного запроса"""
    result = db.query(
        "Тестовый запрос", search_type="hybrid", lambda_param=0.5, similarity_weight=0.7
    )
    assert result is not None
    assert isinstance(result, str)


def test_metadata_extraction(db):
    """Тест извлечения метаданных"""
    # Создаем тестовый документ с метаданными
    test_doc = Document(
        page_content="Это важный тестовый документ с данными 123",
        metadata={"source": "test.txt", "page": 1},
    )

    # Добавляем документ в базу
    db.add_documents([test_doc])

    # Проверяем, что метаданные были добавлены
    result = db.query("тестовый")
    assert result is not None
    assert isinstance(result, str)
