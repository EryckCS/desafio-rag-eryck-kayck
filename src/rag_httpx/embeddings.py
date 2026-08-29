"""Cria a coleção ChromaDB usando embeddings locais do Ollama via LangChain."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

from langchain_chroma import Chroma  # Persiste embeddings e metadados em uma coleção ChromaDB.
from langchain_core.documents import Document  # Representa os chunks que serão inseridos no banco vetorial.
from langchain_ollama import OllamaEmbeddings  # Conecta o modelo qwen3-embedding local ao LangChain.

from .config import RagSettings  # Importa modelo, coleção e caminho configurados do projeto.


class VectorStoreError(RuntimeError):  # Representa falhas compreensíveis de embeddings ou ChromaDB.
    """Indica problema ao criar, abrir ou alimentar a coleção vetorial."""


def create_embedding_model(settings: RagSettings) -> OllamaEmbeddings:  # Cria o adaptador LangChain para o modelo local.
    """Retorna o objeto que transforma documentos e perguntas em embeddings."""
    return OllamaEmbeddings(model=settings.embedding_model)  # Usa exatamente a tag instalada no Ollama local.


def build_chroma_store(chunks: tuple[Document, ...], settings: RagSettings) -> Chroma:  # Gera embeddings e persiste a coleção ChromaDB.
    """Cria uma coleção nova com todos os chunks e seus metadados rastreáveis."""
    settings.validate()  # Confirma valores básicos antes de iniciar a indexação.
    if not chunks:  # Trata explicitamente um corpus sem chunks.
        raise VectorStoreError("Não é possível criar a coleção: não existem chunks.")  # Informa o erro de maneira compreensível.
    if settings.chroma_directory.exists():  # Verifica se existe uma coleção de uma indexação anterior.
        previous_store = Chroma(  # Abre a coleção anterior para removê-la antes da reconstrução.
            collection_name=settings.chroma_collection_name,  # Seleciona a coleção exclusiva da documentação HTTPX.
            persist_directory=str(settings.chroma_directory),  # Usa a pasta local de persistência configurada.
            embedding_function=create_embedding_model(settings),  # Mantém a mesma função de embeddings ao abrir a coleção.
        )
        try:  # Trata o caso de a pasta existir, mas a coleção ainda não ter sido criada.
            previous_store.delete_collection()  # Remove somente a coleção anterior antes de inserir dados atualizados.
        except Exception:  # Ignora ausência de coleção anterior, pois a nova será criada logo em seguida.
            pass  # Prossegue para a criação da coleção nova.
    try:  # Converte qualquer erro do ChromaDB ou Ollama em uma mensagem do projeto.
        store = Chroma(  # Cria uma coleção vazia que receberá os chunks em grupos menores.
            collection_name=settings.chroma_collection_name,  # Cria a coleção com nome explícito.
            persist_directory=str(settings.chroma_directory),  # Salva a coleção dentro de data/chroma.
            embedding_function=create_embedding_model(settings),  # Usa qwen3-embedding:4b pelo adaptador LangChain.
            collection_metadata={"hnsw:space": "cosine"},  # Configura distância cosseno para a futura busca semântica.
        )
        for batch_start in range(0, len(chunks), settings.embedding_batch_size):  # Percorre os chunks respeitando o batch configurado.
            batch = chunks[batch_start : batch_start + settings.embedding_batch_size]  # Seleciona no máximo oito chunks para a chamada atual.
            ids = [str(chunk.metadata["chunk_id"]) for chunk in batch]  # Mantém IDs estáveis para todos os chunks deste batch.
            store.add_documents(documents=list(batch), ids=ids)  # Gera embeddings pelo LangChain e grava o batch no ChromaDB.
        return store  # Devolve a coleção persistida depois de inserir todos os chunks.
    except Exception as error:  # Captura erros de servidor, modelo ausente ou banco vetorial.
        raise VectorStoreError(f"Falha ao criar coleção ChromaDB: {error}") from error  # Mantém a causa técnica disponível.


def load_chroma_store(settings: RagSettings) -> Chroma:  # Abre uma coleção persistida sem gerar embeddings dos documentos novamente.
    """Retorna a coleção ChromaDB pronta para consultas semânticas."""
    if not settings.chroma_directory.is_dir():  # Confere se a indexação já criou a pasta persistida.
        raise VectorStoreError("Coleção ChromaDB ausente. Gere os embeddings antes de pesquisar.")  # Orienta o próximo passo necessário.
    return Chroma(  # Reconecta a coleção ao mesmo modelo usado durante a indexação.
        collection_name=settings.chroma_collection_name,  # Seleciona a coleção exclusiva do corpus HTTPX.
        persist_directory=str(settings.chroma_directory),  # Informa onde o ChromaDB guardou seus arquivos.
        embedding_function=create_embedding_model(settings),  # Usa o modelo local também para embeddings das perguntas.
    )
