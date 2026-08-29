"""Testa os chunks LangChain e seus metadados obrigatórios."""

import unittest  # Biblioteca padrão usada para executar testes automatizados.

from langchain_core.documents import Document  # Cria documento mínimo para testar o splitter.

from src.rag_httpx.chunking import ChunkingError, create_chunks  # Importa a função LangChain que será verificada.
from src.rag_httpx.config import RagSettings, SETTINGS  # Importa configurações normal e reduzida.
from src.rag_httpx.documents import load_httpx_documents  # Importa os Documents reais do HTTPX.


class LangChainChunkingTests(unittest.TestCase):  # Agrupa verificações do chunking LangChain.
    @classmethod  # Carrega e divide o corpus uma única vez para todos os testes.
    def setUpClass(cls) -> None:  # Prepara chunks reais a partir da documentação HTTPX.
        cls.chunks = create_chunks(load_httpx_documents(SETTINGS), SETTINGS)  # Executa os splitters LangChain configurados.

    def test_chunks_preserve_required_metadata(self) -> None:  # Confirma fonte, título, seção e identificador em cada chunk.
        chunk = self.chunks[0]  # Seleciona um chunk real para verificar seus metadados.
        self.assertTrue(chunk.metadata["chunk_id"])  # Confirma que o chunk possui identificador rastreável.
        self.assertTrue(chunk.metadata["source_path"].startswith("docs/"))  # Confirma a fonte relativa ao corpus.
        self.assertTrue(chunk.metadata["title"])  # Confirma o título do documento de origem.
        self.assertTrue(chunk.metadata["section"])  # Confirma a seção usada na saída da busca.
        self.assertLessEqual(chunk.metadata["word_count"], SETTINGS.chunk_size_words)  # Confirma o limite aproximado de palavras.

    def test_small_documents_receive_overlap(self) -> None:  # Confirma que o splitter aceita tamanho e overlap em palavras.
        document = Document(page_content="# Exemplo\n\n" + " ".join(f"palavra{number}" for number in range(30)), metadata={"source_path": "docs/exemplo.md", "title": "Exemplo"})  # Cria texto previsível.
        chunks = create_chunks((document,), RagSettings(chunk_size_words=10, chunk_overlap_words=3))  # Força vários chunks pequenos.
        self.assertGreater(len(chunks), 1)  # Confirma que o documento foi dividido em mais de um trecho.
        self.assertTrue(all(chunk.metadata["word_count"] <= 10 for chunk in chunks))  # Confirma o limite em todos os chunks.

    def test_empty_documents_have_clear_error(self) -> None:  # Confirma o tratamento de corpus vazio exigido pelo desafio.
        with self.assertRaises(ChunkingError):  # Espera a exceção própria do módulo.
            create_chunks((), SETTINGS)  # Tenta gerar chunks sem fornecer documentos.


if __name__ == "__main__":  # Permite executar este arquivo diretamente.
    unittest.main()  # Inicia o executor padrão de testes.
