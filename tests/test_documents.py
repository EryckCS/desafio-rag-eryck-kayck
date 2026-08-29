"""Testa os Documents LangChain criados a partir da documentação HTTPX."""

import unittest  # Biblioteca padrão usada para executar testes automatizados.

from langchain_core.documents import Document  # Confirma que a leitura produz objetos LangChain.

from src.rag_httpx.config import SETTINGS  # Importa o corpus configurado para o projeto.
from src.rag_httpx.documents import load_httpx_documents  # Importa o carregador LangChain de Markdown.


class LangChainDocumentTests(unittest.TestCase):  # Agrupa verificações dos documentos lidos.
    @classmethod  # Carrega o corpus uma única vez para todos os testes desta classe.
    def setUpClass(cls) -> None:  # Prepara os Documents LangChain do HTTPX.
        cls.documents = load_httpx_documents(SETTINGS)  # Lê os 23 Markdown exigidos pela atividade.

    def test_loads_required_documents_as_langchain_documents(self) -> None:  # Confirma a integração com a estrutura Document.
        self.assertEqual(len(self.documents), SETTINGS.expected_markdown_files)  # Confirma a quantidade de 23 documentos.
        self.assertTrue(all(isinstance(document, Document) for document in self.documents))  # Confirma o tipo LangChain.

    def test_preserves_source_and_title_metadata(self) -> None:  # Confirma metadados necessários para rastreabilidade.
        quickstart = next(document for document in self.documents if document.metadata["source_path"] == "docs/quickstart.md")  # Localiza QuickStart.
        self.assertEqual(quickstart.metadata["title"], "QuickStart")  # Confirma o título principal preservado.
        self.assertTrue(quickstart.page_content.startswith("# QuickStart"))  # Confirma que o texto original foi mantido.


if __name__ == "__main__":  # Permite executar este arquivo diretamente.
    unittest.main()  # Inicia o executor padrão de testes.
