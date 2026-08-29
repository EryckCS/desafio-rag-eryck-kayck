"""Testa a descoberta dos arquivos Markdown já obtidos no módulo 2."""

import unittest  # Biblioteca padrão usada para executar testes automatizados.

from src.rag_httpx.config import SETTINGS  # Importa o caminho configurado da documentação HTTPX.
from src.rag_httpx.ingestion import CorpusError, discover_markdown_files  # Importa a função e o erro testados.


class MarkdownDiscoveryTests(unittest.TestCase):  # Agrupa os testes da descoberta recursiva de Markdown.
    def test_discovers_all_required_markdown_files(self) -> None:  # Confirma a busca recursiva no corpus real do desafio.
        markdown_files = discover_markdown_files(SETTINGS.docs_directory)  # Busca os Markdown do HTTPX já clonado.
        self.assertEqual(len(markdown_files), SETTINGS.expected_markdown_files)  # Confirma a contagem de 23 arquivos.
        self.assertTrue(any(file.parent != SETTINGS.docs_directory for file in markdown_files))  # Confirma que há arquivos em subpastas.

    def test_rejects_missing_docs_directory(self) -> None:  # Confirma que uma pasta inexistente gera mensagem clara.
        missing_directory = SETTINGS.corpus_directory / "pasta_inexistente"  # Define um caminho que não existe no corpus.
        with self.assertRaises(CorpusError):  # Espera a exceção própria do projeto.
            discover_markdown_files(missing_directory)  # Tenta buscar arquivos no caminho inexistente.


if __name__ == "__main__":  # Permite executar este teste diretamente.
    unittest.main()  # Inicia o executor padrão de testes.
