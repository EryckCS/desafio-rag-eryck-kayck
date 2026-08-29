"""Testa validações e formatação de resultados sem consultar modelos reais."""

import unittest  # Biblioteca padrão usada para executar testes automatizados.
from unittest.mock import patch  # Substitui o ChromaDB por um objeto controlado no teste.

from langchain_core.documents import Document  # Cria documentos simulados iguais aos retornados pelo ChromaDB.

from src.rag_httpx.config import SETTINGS  # Importa limites e instrução configurados.
from src.rag_httpx.retriever import RetrievalError, search  # Importa a busca e o erro que serão verificados.


class FakeStore:  # Simula somente o método do ChromaDB usado pela função search.
    def similarity_search_with_relevance_scores(self, query: str, k: int) -> list[tuple[Document, float]]:  # Devolve resultados previsíveis.
        metadata = {"chunk_id": "docs/quickstart.md::chunk-0001", "source_path": "docs/quickstart.md", "title": "QuickStart", "section": "Response Content"}  # Define fonte completa.
        document = Document(page_content="HTTPX decodes response content.", metadata=metadata)  # Cria um documento recuperado simulado.
        return [(document, 0.82)] * k  # Repete resultado para simular a quantidade solicitada.


class RetrieverTests(unittest.TestCase):  # Agrupa verificações do módulo de busca semântica.
    @patch("src.rag_httpx.retriever.load_chroma_store", return_value=FakeStore())  # Evita abrir o banco vetorial real.
    def test_search_returns_ordered_results_with_sources(self, mocked_store) -> None:  # Confirma saída com ranking, score e fonte.
        results = search("Como ler o conteúdo de uma resposta?", SETTINGS, top_k=3)  # Executa busca com coleção simulada.
        self.assertEqual(len(results), 3)  # Confirma que top_k controla a quantidade de resultados.
        self.assertEqual(results[0].rank, 1)  # Confirma que o primeiro resultado recebe ranking um.
        self.assertEqual(results[0].score, 0.82)  # Confirma que o score é preservado na saída.
        self.assertEqual(results[0].source_path, "docs/quickstart.md")  # Confirma que a fonte é exibida.
        self.assertEqual(mocked_store.call_count, 1)  # Confirma que a coleção foi aberta uma única vez.

    def test_empty_question_has_clear_error(self) -> None:  # Confirma o tratamento obrigatório de pergunta vazia.
        with self.assertRaisesRegex(RetrievalError, "não pode estar vazia"):  # Espera mensagem compreensível.
            search("   ", SETTINGS)  # Tenta buscar usando somente espaços.

    def test_invalid_top_k_has_clear_error(self) -> None:  # Confirma o tratamento obrigatório de top_k inválido.
        with self.assertRaisesRegex(RetrievalError, "entre 3 e 5"):  # Espera mensagem com os limites aceitos.
            search("Como usar HTTPX?", SETTINGS, top_k=6)  # Tenta solicitar quantidade acima do permitido.


if __name__ == "__main__":  # Permite executar este arquivo diretamente.
    unittest.main()  # Inicia o executor padrão de testes.
