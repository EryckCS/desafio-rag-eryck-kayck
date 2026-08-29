"""Prepara o corpus HTTPX e reconstrói o índice ChromaDB em um único comando."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

from dataclasses import dataclass  # Cria um resumo imutável da preparação concluída.

from .chunking import create_chunks  # Divide documentos Markdown em trechos rastreáveis.
from .config import SETTINGS, RagSettings  # Importa a configuração padrão e seu tipo.
from .documents import load_httpx_documents  # Lê os Markdown validados como Documents LangChain.
from .embeddings import build_chroma_store  # Gera embeddings e grava a coleção vetorial persistente.
from .ingestion import obtain_httpx_repository, validate_httpx_corpus  # Obtém e valida o repositório exigido.


class IndexingError(RuntimeError):  # Representa uma falha compreensível na preparação do RAG.
    """Indica que o corpus ou o índice não pôde ser preparado."""


@dataclass(frozen=True)  # Evita alteração dos números registrados depois da preparação.
class IndexingSummary:  # Reúne evidências objetivas da indexação realizada.
    """Guarda commit, quantidade de Markdown, documentos e chunks indexados."""

    commit: str  # Hash do commit HTTPX efetivamente validado.
    markdown_files: int  # Quantidade de Markdown encontrados em docs recursivamente.
    documents: int  # Quantidade de Documents LangChain criados a partir do corpus.
    chunks: int  # Quantidade de chunks inseridos no ChromaDB.


def prepare_index(settings: RagSettings) -> IndexingSummary:  # Executa todas as etapas de preparação na ordem correta.
    """Obtém HTTPX, valida o corpus, cria chunks e reconstrói o ChromaDB."""
    try:  # Converte erros dos módulos internos em uma mensagem única de preparação.
        obtain_httpx_repository(settings)  # Clona o repositório quando necessário e fixa o commit obrigatório.
        corpus_summary = validate_httpx_corpus(settings)  # Confirma commit correto e exatamente 23 Markdown.
        documents = load_httpx_documents(settings)  # Lê cada Markdown validado preservando origem e título.
        chunks = create_chunks(documents, settings)  # Divide os documentos mantendo seção e identificadores únicos.
        build_chroma_store(chunks, settings)  # Gera embeddings locais e substitui a coleção ChromaDB anterior.
    except Exception as error:  # Trata Git, leitura, chunking, Ollama e ChromaDB com uma mensagem clara.
        raise IndexingError(f"Não foi possível preparar o índice: {error}") from error  # Mantém a causa técnica encadeada para diagnóstico.
    return IndexingSummary(  # Devolve dados que comprovam a preparação para o terminal e README.
        commit=corpus_summary.commit,  # Registra o commit do corpus validado.
        markdown_files=len(corpus_summary.markdown_files),  # Registra a contagem recursiva de Markdown.
        documents=len(documents),  # Registra quantos documentos foram lidos.
        chunks=len(chunks),  # Registra quantos chunks receberam embeddings.
    )


def format_indexing_summary(summary: IndexingSummary) -> str:  # Converte o resumo em linhas legíveis no terminal.
    """Exibe as evidências necessárias para confirmar que a preparação terminou."""
    return "\n".join(  # Junta os campos em uma única saída textual.
        [
            "Índice preparado com sucesso.",  # Confirma que todas as etapas foram concluídas.
            f"Commit HTTPX: {summary.commit}",  # Mostra a versão exata do corpus.
            f"Markdown encontrados: {summary.markdown_files}",  # Mostra a evidência de descoberta recursiva.
            f"Documentos lidos: {summary.documents}",  # Mostra a quantidade de documentos LangChain.
            f"Chunks indexados: {summary.chunks}",  # Mostra a quantidade gravada no ChromaDB.
        ]
    )


def main() -> int:  # Disponibiliza a preparação por meio de python -m.
    """Reconstrói o índice local e devolve código de saída apropriado."""
    try:  # Evita mostrar traceback técnico para erros esperados de preparação.
        summary = prepare_index(SETTINGS)  # Executa a preparação usando as configurações padrão.
        print(format_indexing_summary(summary))  # Exibe as evidências depois da conclusão.
        return 0  # Informa ao terminal que a preparação terminou corretamente.
    except IndexingError as error:  # Captura falhas convertidas pela função de preparação.
        print(f"Erro: {error}")  # Mostra uma orientação legível para diagnosticar o problema.
        return 2  # Informa ao terminal que houve erro de preparação.


if __name__ == "__main__":  # Permite executar o módulo diretamente pelo terminal.
    raise SystemExit(main())  # Encerra o processo usando o código retornado pela função principal.
