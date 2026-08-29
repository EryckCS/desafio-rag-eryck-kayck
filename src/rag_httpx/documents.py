"""Converte os Markdown do corpus em objetos Document do LangChain."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

import re  # Localiza o título principal dentro do conteúdo Markdown.
from pathlib import Path  # Representa caminhos de arquivo de maneira portátil.

from langchain_core.documents import Document  # Representa texto e metadados no padrão LangChain.

from .config import RagSettings  # Importa os caminhos configurados do projeto.
from .ingestion import validate_httpx_corpus  # Confirma corpus, commit e quantidade de arquivos antes da leitura.


TITLE_PATTERN = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)  # Reconhece o primeiro título Markdown de nível um.


class DocumentLoadError(RuntimeError):  # Representa falhas compreensíveis durante a leitura de arquivos.
    """Indica que um Markdown não pôde ser lido ou não pertence ao corpus."""


def read_markdown_document(markdown_path: Path, corpus_directory: Path) -> Document:  # Lê um arquivo e devolve um Document LangChain.
    """Cria um Document com texto integral e metadados rastreáveis de origem."""
    if not markdown_path.is_file():  # Confere se o caminho realmente aponta para um arquivo Markdown.
        raise DocumentLoadError(f"Arquivo Markdown não encontrado: {markdown_path}")  # Explica qual arquivo está ausente.
    try:  # Trata falhas de acesso e codificação durante a leitura.
        page_content = markdown_path.read_text(encoding="utf-8")  # Lê todo o Markdown no padrão UTF-8.
    except OSError as error:  # Captura problemas do sistema ao abrir o arquivo.
        raise DocumentLoadError(f"Não foi possível ler o arquivo: {markdown_path}") from error  # Mantém a causa técnica do erro.
    try:  # Garante que a fonte exibida será relativa ao corpus obrigatório.
        source_path = markdown_path.relative_to(corpus_directory).as_posix()  # Produz caminho como docs/quickstart.md.
    except ValueError as error:  # Trata arquivo informado fora da pasta configurada.
        raise DocumentLoadError(f"Arquivo fora do corpus configurado: {markdown_path}") from error  # Explica a origem inválida.
    title_match = TITLE_PATTERN.search(page_content)  # Procura o título principal no conteúdo do arquivo.
    fallback_title = markdown_path.stem.replace("_", " ").title()  # Cria título legível quando não há título Markdown.
    title = title_match.group(1).strip() if title_match else fallback_title  # Prefere título escrito no documento.
    metadata = {"source_path": source_path, "title": title}  # Mantém somente metadados simples aceitos pelo ChromaDB.
    return Document(page_content=page_content, metadata=metadata)  # Devolve o objeto que os splitters do LangChain receberão.


def load_httpx_documents(settings: RagSettings) -> tuple[Document, ...]:  # Carrega todos os arquivos obrigatórios como Documents LangChain.
    """Valida o corpus HTTPX e retorna documentos ordenados com fonte e título."""
    corpus_summary = validate_httpx_corpus(settings)  # Confirma commit correto e os 23 Markdown exigidos.
    documents = tuple(  # Converte cada Markdown validado em um Document LangChain.
        read_markdown_document(markdown_path, settings.corpus_directory)  # Preserva texto, caminho e título de cada arquivo.
        for markdown_path in corpus_summary.markdown_files  # Mantém a ordem estável da descoberta de arquivos.
    )
    if not documents:  # Protege as próximas etapas contra um corpus vazio.
        raise DocumentLoadError("O corpus validado não possui documentos para leitura.")  # Informa o problema de forma clara.
    return documents  # Devolve os Documents prontos para o splitter LangChain.
