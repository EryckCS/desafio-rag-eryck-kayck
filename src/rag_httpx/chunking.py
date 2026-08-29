"""Cria chunks LangChain preservando cabeçalhos e metadados de origem."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

import re  # Conta palavras para configurar o tamanho dos chunks.

from langchain_core.documents import Document  # Representa documentos e chunks no padrão LangChain.
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  # Divide cabeçalhos e textos longos.

from .config import RagSettings  # Importa tamanho e overlap definidos para o projeto.


WORD_PATTERN = re.compile(r"\S+")  # Considera qualquer sequência sem espaço como palavra para medir chunks.
HEADERS_TO_SPLIT_ON = [("#", "header_1"), ("##", "header_2"), ("###", "header_3")]  # Define cabeçalhos preservados pelo splitter.


class ChunkingError(ValueError):  # Representa entradas inválidas para a criação de chunks.
    """Indica que não há documentos ou que os parâmetros de chunking são inválidos."""


def _word_length(text: str) -> int:  # Devolve a quantidade de palavras usada pelo splitter recursivo.
    return len(WORD_PATTERN.findall(text))  # Conta palavras sem depender de caracteres ou tokens do modelo.


def _section_name(metadata: dict[str, object]) -> str:  # Escolhe a seção mais específica disponível nos metadados.
    return str(metadata.get("header_3") or metadata.get("header_2") or metadata.get("header_1") or metadata["title"])  # Prioriza níveis profundos.


def create_chunks(documents: tuple[Document, ...], settings: RagSettings) -> tuple[Document, ...]:  # Divide Documents LangChain em chunks rastreáveis.
    """Aplica splitters LangChain e adiciona metadados exigidos ao resultado."""
    settings.validate()  # Confirma que tamanho e overlap continuam em uma faixa válida.
    if not documents:  # Trata explicitamente um corpus sem documentos.
        raise ChunkingError("Não é possível criar chunks: o corpus não possui documentos.")  # Informa o erro de maneira compreensível.
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)  # Separa o Markdown por títulos.
    text_splitter = RecursiveCharacterTextSplitter(  # Cria o splitter que limita o tamanho de cada seção longa.
        chunk_size=settings.chunk_size_words,  # Usa o limite de aproximadamente 80 palavras definido na configuração.
        chunk_overlap=settings.chunk_overlap_words,  # Repete 15 palavras entre chunks consecutivos para manter contexto.
        length_function=_word_length,  # Mede tamanho por palavras, não por quantidade de caracteres.
        separators=["\n\n", "\n", " ", ""],  # Prefere quebrar em parágrafos, linhas e palavras antes de caracteres.
    )
    chunks: list[Document] = []  # Armazena todos os chunks em ordem estável.
    for document in documents:  # Processa cada Markdown já lido pelo módulo anterior.
        header_documents = header_splitter.split_text(document.page_content)  # Separa o conteúdo de acordo com seus cabeçalhos Markdown.
        if not header_documents:  # Mantém documentos sem cabeçalho disponíveis para a etapa seguinte.
            header_documents = [Document(page_content=document.page_content, metadata={})]  # Cria seção única quando não há títulos.
        for header_document in header_documents:  # Processa cada seção obtida do splitter de cabeçalhos.
            metadata = {**document.metadata, **header_document.metadata}  # Junta fonte original aos cabeçalhos detectados pelo LangChain.
            section_documents = text_splitter.split_documents([Document(page_content=header_document.page_content, metadata=metadata)])  # Divide seções grandes.
            for section_document in section_documents:  # Adiciona metadados específicos a cada chunk produzido.
                chunk_number = len(chunks) + 1  # Calcula posição global estável para criar o identificador único.
                chunk_metadata = dict(section_document.metadata)  # Copia metadados para não alterar objetos anteriores.
                chunk_metadata["chunk_id"] = f"{chunk_metadata['source_path']}::chunk-{chunk_number:04d}"  # Cria ID rastreável do chunk.
                chunk_metadata["chunk_index"] = chunk_number  # Registra a posição global do chunk no corpus.
                chunk_metadata["section"] = _section_name(chunk_metadata)  # Registra a seção exibida no resultado da busca.
                chunk_metadata["word_count"] = _word_length(section_document.page_content)  # Registra o tamanho real do trecho em palavras.
                chunks.append(Document(page_content=section_document.page_content, metadata=chunk_metadata))  # Salva o chunk LangChain completo.
    if not chunks:  # Impede indexação de um resultado vazio.
        raise ChunkingError("Não foi possível criar chunks a partir dos documentos fornecidos.")  # Informa a falha de forma clara.
    return tuple(chunks)  # Devolve chunks imutáveis e ordenados para o ChromaDB.
