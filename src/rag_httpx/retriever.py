"""Consulta a coleção ChromaDB e transforma documentos recuperados em resultados claros."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

from dataclasses import dataclass  # Cria uma estrutura simples para a saída da busca.

from .config import RagSettings  # Importa limites de top_k e a instrução da pergunta.
from .embeddings import VectorStoreError, load_chroma_store  # Abre a coleção ChromaDB já persistida.


class RetrievalError(ValueError):  # Representa entradas inválidas ou falhas compreensíveis na busca.
    """Indica que uma pergunta ou um parâmetro de busca não pode ser processado."""


@dataclass(frozen=True)  # Evita alteração acidental de ranking, score ou fonte após a busca.
class SearchResult:  # Representa um chunk retornado pela busca semântica.
    """Guarda posição, relevância, trecho e metadados da fonte recuperada."""

    rank: int  # Posição do resultado na ordem de maior relevância.
    score: float  # Score de relevância fornecido pela busca de similaridade do ChromaDB.
    text: str  # Trecho original do chunk recuperado.
    chunk_id: str  # Identificador único do chunk armazenado no ChromaDB.
    source_path: str  # Caminho Markdown relativo ao corpus HTTPX.
    title: str  # Título principal do documento de origem.
    section: str  # Seção específica que contém o trecho retornado.


def _validate_query(query: str) -> str:  # Confirma que a pergunta possui conteúdo útil.
    if not isinstance(query, str):  # Impede objetos que não podem ser enviados ao modelo de embedding.
        raise RetrievalError("A pergunta deve ser um texto.")  # Explica o tipo de entrada esperado.
    cleaned_query = query.strip()  # Remove espaços antes e depois da pergunta.
    if not cleaned_query:  # Trata explicitamente uma pergunta vazia exigida pelo desafio.
        raise RetrievalError("A pergunta não pode estar vazia.")  # Informa como corrigir a entrada.
    return cleaned_query  # Devolve a pergunta limpa para a busca semântica.


def _validate_top_k(top_k: int, settings: RagSettings) -> int:  # Confirma a quantidade de resultados permitida.
    if isinstance(top_k, bool) or not isinstance(top_k, int):  # Impede valores booleanos, texto e números decimais.
        raise RetrievalError("top_k deve ser um número inteiro entre 3 e 5.")  # Explica o formato aceito.
    if not settings.min_top_k <= top_k <= settings.max_top_k:  # Aplica a faixa exigida pelo desafio.
        raise RetrievalError(f"top_k deve estar entre {settings.min_top_k} e {settings.max_top_k}.")  # Mostra os limites corretos.
    return top_k  # Devolve a quantidade validada de resultados.


def _format_query(query: str, settings: RagSettings) -> str:  # Prepara a pergunta com a instrução recomendada pelo modelo de embedding.
    return f"Instruct: {settings.embedding_query_instruction}\nQuery: {query}"  # Diferencia embedding de consulta de embedding de documento.


def search(query: str, settings: RagSettings, top_k: int | None = None) -> tuple[SearchResult, ...]:  # Executa busca semântica no ChromaDB.
    """Retorna de três a cinco chunks ordenados com score e metadados de fonte."""
    settings.validate()  # Confirma os limites de configuração antes de realizar a consulta.
    cleaned_query = _validate_query(query)  # Valida e normaliza a pergunta recebida do usuário.
    requested_top_k = settings.default_top_k if top_k is None else top_k  # Usa três resultados quando o usuário não escolhe outro valor.
    validated_top_k = _validate_top_k(requested_top_k, settings)  # Garante que serão retornados entre três e cinco itens.
    try:  # Converte falha ao abrir o índice em uma mensagem específica da busca.
        store = load_chroma_store(settings)  # Reconecta a coleção persistida ao modelo de embeddings local.
        matches = store.similarity_search_with_relevance_scores(  # Obtém documentos e scores de relevância da consulta.
            _format_query(cleaned_query, settings),  # Envia a pergunta com instrução de recuperação semântica.
            k=validated_top_k,  # Solicita exatamente a quantidade de resultados validada.
        )
    except VectorStoreError as error:  # Captura índice ausente ou inacessível.
        raise RetrievalError(str(error)) from error  # Mantém uma mensagem compreensível para quem executa o projeto.
    except Exception as error:  # Captura falhas do ChromaDB, Ollama ou modelo local.
        raise RetrievalError(f"Não foi possível executar a busca semântica: {error}") from error  # Preserva a causa técnica do problema.
    ordered_matches = sorted(matches, key=lambda item: item[1], reverse=True)  # Reforça a ordem do maior para o menor score.
    results: list[SearchResult] = []  # Armazena os resultados convertidos para uma saída explícita.
    for rank, (document, score) in enumerate(ordered_matches, start=1):  # Percorre resultados atribuindo posição de ranking.
        metadata = document.metadata  # Lê os metadados salvos junto ao chunk no ChromaDB.
        try:  # Garante que todos os campos obrigatórios existem antes de criar a saída.
            result = SearchResult(  # Converte o Document LangChain em resultado pronto para exibição.
                rank=rank,  # Registra a posição ordenada do resultado.
                score=float(score),  # Converte o score para número simples e serializável.
                text=document.page_content,  # Mantém o trecho original recuperado.
                chunk_id=str(metadata["chunk_id"]),  # Mantém o identificador rastreável do chunk.
                source_path=str(metadata["source_path"]),  # Mantém o caminho do arquivo de origem.
                title=str(metadata["title"]),  # Mantém o título principal do documento.
                section=str(metadata["section"]),  # Mantém a seção específica do trecho.
            )
        except KeyError as error:  # Trata coleção criada sem metadados obrigatórios.
            raise RetrievalError(f"Chunk recuperado sem metadado obrigatório: {error}") from error  # Explica qual campo está ausente.
        results.append(result)  # Acrescenta o resultado completo à lista ordenada.
    return tuple(results)  # Devolve resultados imutáveis para a interface ou geração opcional.
