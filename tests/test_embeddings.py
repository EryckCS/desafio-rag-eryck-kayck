"""Testa a criação do adaptador LangChain e erros básicos da coleção ChromaDB."""

import unittest  # Biblioteca padrão usada para executar testes automatizados.

from langchain_ollama import OllamaEmbeddings  # Confirma o tipo do adaptador de embeddings.

from src.rag_httpx.config import SETTINGS  # Importa a tag local do modelo de embeddings.
from src.rag_httpx.embeddings import VectorStoreError, build_chroma_store, create_embedding_model  # Importa funções do módulo vetorial.


class LangChainEmbeddingTests(unittest.TestCase):  # Agrupa verificações que não chamam o modelo real.
    def test_creates_langchain_ollama_embedding_adapter(self) -> None:  # Confirma a integração do modelo local com LangChain.
        embedding_model = create_embedding_model(SETTINGS)  # Cria o adaptador sem gerar vetores ainda.
        self.assertIsInstance(embedding_model, OllamaEmbeddings)  # Confirma o tipo fornecido pela integração oficial.
        self.assertEqual(embedding_model.model, SETTINGS.embedding_model)  # Confirma que a tag local correta foi configurada.

    def test_empty_chunks_have_clear_error(self) -> None:  # Confirma o tratamento de coleção sem documentos.
        with self.assertRaises(VectorStoreError):  # Espera a exceção própria do módulo.
            build_chroma_store((), SETTINGS)  # Tenta criar uma coleção sem chunks.


if __name__ == "__main__":  # Permite executar este arquivo diretamente.
    unittest.main()  # Inicia o executor padrão de testes.
