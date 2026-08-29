"""Testa a orquestração da preparação sem clonar ou gerar embeddings reais."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

import unittest  # Executa os testes automatizados do módulo de preparação.
from pathlib import Path  # Cria caminhos representativos para o resumo de corpus falso.
from unittest.mock import patch  # Substitui operações externas por retornos controlados.

from langchain_core.documents import Document  # Cria documentos mínimos para validar a orquestração.

from src.rag_httpx.config import RagSettings  # Cria a configuração usada no cenário de teste.
from src.rag_httpx.indexer import IndexingSummary, format_indexing_summary, prepare_index  # Importa o módulo que será validado.
from src.rag_httpx.ingestion import CorpusSummary  # Cria a evidência de corpus validado simulada.


class IndexerTests(unittest.TestCase):  # Agrupa verificações do comando único de preparação.
    def test_prepare_index_runs_complete_pipeline(self) -> None:  # Confirma ordem lógica: obter, validar, ler, dividir e indexar.
        settings = RagSettings()  # Usa a configuração padrão sem acessar recursos reais.
        corpus_summary = CorpusSummary(Path("data/httpx"), settings.httpx_commit, (Path("docs/quickstart.md"),))  # Simula corpus validado.
        document = Document(page_content="# QuickStart", metadata={"source_path": "docs/quickstart.md", "title": "QuickStart"})  # Simula documento carregado.
        chunk = Document(page_content="Texto", metadata={"chunk_id": "docs/quickstart.md::chunk-0001"})  # Simula chunk pronto para indexação.
        with patch("src.rag_httpx.indexer.obtain_httpx_repository"), patch("src.rag_httpx.indexer.validate_httpx_corpus", return_value=corpus_summary), patch("src.rag_httpx.indexer.load_httpx_documents", return_value=(document,)), patch("src.rag_httpx.indexer.create_chunks", return_value=(chunk,)), patch("src.rag_httpx.indexer.build_chroma_store") as build_store:  # Substitui cada operação externa.
            summary = prepare_index(settings)  # Executa a orquestração usando dados simulados.
        build_store.assert_called_once_with((chunk,), settings)  # Confirma que os chunks chegam ao ChromaDB.
        self.assertEqual(summary.commit, settings.httpx_commit)  # Confirma que o commit validado entra no resumo.
        self.assertEqual(summary.markdown_files, 1)  # Confirma a contagem de Markdown simulada.
        self.assertEqual(summary.documents, 1)  # Confirma a contagem de documentos carregados.
        self.assertEqual(summary.chunks, 1)  # Confirma a contagem de chunks gerados.

    def test_format_indexing_summary_displays_evidence(self) -> None:  # Confirma que o terminal mostra dados verificáveis da preparação.
        summary = IndexingSummary("commit-teste", 23, 23, 318)  # Cria um resumo conhecido para teste de formatação.
        output = format_indexing_summary(summary)  # Converte o resumo em saída de terminal.
        self.assertIn("commit-teste", output)  # Confirma a exibição do commit.
        self.assertIn("Markdown encontrados: 23", output)  # Confirma a exibição da contagem obrigatória.
        self.assertIn("Chunks indexados: 318", output)  # Confirma a exibição da contagem indexada.
