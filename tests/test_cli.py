"""Testa medição de tempo e formatação da interface de terminal."""

import unittest  # Biblioteca padrão usada para executar testes automatizados.
from unittest.mock import patch  # Substitui relógio e busca real por valores controlados.

from src.rag_httpx.cli import QueryExecution, execute_query, format_execution  # Importa funções da interface de terminal.
from src.rag_httpx.config import SETTINGS  # Importa a configuração usada durante o teste.
from src.rag_httpx.retriever import SearchResult  # Cria resultado previsível para testar a saída formatada.


class CliTests(unittest.TestCase):  # Agrupa verificações da interface de terminal.
    @patch("src.rag_httpx.cli.perf_counter", side_effect=[10.0, 12.5])  # Simula relógio de alta precisão com duração conhecida.
    @patch("src.rag_httpx.cli.search")  # Evita consultar o modelo ou ChromaDB durante o teste.
    def test_execute_query_measures_elapsed_time(self, mocked_search, mocked_clock) -> None:  # Confirma que a duração é calculada corretamente.
        result = SearchResult(1, 0.8, "Trecho", "chunk-1", "docs/quickstart.md", "QuickStart", "Resposta")  # Cria resultado mínimo simulado.
        mocked_search.return_value = (result,)  # Configura a busca simulada para retornar um resultado.
        execution = execute_query("Como usar HTTPX?", SETTINGS, top_k=3)  # Executa a consulta com relógio e busca simulados.
        self.assertEqual(execution.elapsed_seconds, 2.5)  # Confirma a diferença entre início e fim da medição.
        self.assertEqual(execution.results[0].source_path, "docs/quickstart.md")  # Confirma a preservação do resultado retornado.

    def test_format_execution_displays_required_fields(self) -> None:  # Confirma que todos os campos exigidos aparecem no terminal.
        result = SearchResult(1, 0.8, "Trecho recuperado", "chunk-1", "docs/quickstart.md", "QuickStart", "Resposta")  # Cria resultado previsível.
        output = format_execution(QueryExecution("Como usar HTTPX?", (result,), 1.234))  # Formata uma execução conhecida.
        self.assertIn("Tempo de resposta: 1.234 segundos", output)  # Confirma a exibição da duração.
        self.assertIn("Arquivo: docs/quickstart.md", output)  # Confirma a exibição da fonte.
        self.assertIn("Seção: Resposta", output)  # Confirma a exibição da seção.
        self.assertIn("Score: 0.8000", output)  # Confirma a exibição do score.


if __name__ == "__main__":  # Permite executar este arquivo diretamente.
    unittest.main()  # Inicia o executor padrão de testes.
