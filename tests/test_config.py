"""Testa a configuração inicial sem baixar modelos nem documentos."""

import unittest  # Biblioteca padrão usada para executar os testes.

from src.rag_httpx.config import RagSettings, SETTINGS  # Importa a configuração que será verificada.


class RagSettingsTests(unittest.TestCase):  # Agrupa verificações da classe de configuração.
    def test_default_settings_are_valid(self) -> None:  # Confirma que a configuração normal passa.
        SETTINGS.validate()  # Executa todas as validações da configuração padrão.
        self.assertEqual(SETTINGS.default_top_k, 3)  # Confirma o ranking padrão de três resultados.
        self.assertEqual(SETTINGS.embedding_model, "qwen3-embedding:4b")  # Confirma o modelo de busca local.
        self.assertEqual(SETTINGS.generator_model, "qwen3.5:4b")  # Confirma o modelo de geração.

    def test_overlap_cannot_equal_chunk_size(self) -> None:  # Garante que não há chunk inválido.
        settings = RagSettings(chunk_size_words=80, chunk_overlap_words=80)  # Cria uma configuração errada.
        with self.assertRaises(ValueError):  # Espera que a validação rejeite essa configuração.
            settings.validate()  # Executa a validação que deve gerar o erro.

    def test_default_top_k_must_be_in_allowed_range(self) -> None:  # Testa o limite máximo de resultados.
        settings = RagSettings(default_top_k=6)  # Cria valor acima do máximo aceito.
        with self.assertRaises(ValueError):  # Espera que a validação rejeite esse valor.
            settings.validate()  # Executa a validação que deve gerar o erro.


if __name__ == "__main__":  # Permite executar este arquivo de teste diretamente.
    unittest.main()  # Inicia o executor de testes da biblioteca padrão.
