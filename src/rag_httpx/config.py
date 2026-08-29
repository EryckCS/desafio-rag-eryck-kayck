"""Centraliza os valores usados em todas as etapas do Mini-RAG.

Este arquivo não baixa documentos nem modelos; apenas descreve as decisões que
os outros módulos devem usar de forma consistente.
"""

from __future__ import annotations  # Permite anotações de tipo modernas.

from dataclasses import dataclass  # Evita escrever manualmente o construtor da configuração.
from pathlib import Path  # Representa caminhos de arquivos de modo portátil.


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Localiza a raiz do projeto a partir deste arquivo.


@dataclass(frozen=True)  # Cria uma configuração imutável após sua definição.
class RagSettings:  # Reúne todos os parâmetros que os módulos compartilharão.
    """Guarda parâmetros estáveis do projeto e seus caminhos locais."""

    # Define o repositório obrigatório que contém a documentação consultada.
    httpx_repository_url: str = "https://github.com/encode/httpx.git"  # URL Git oficial do HTTPX.
    httpx_commit: str = "b5addb64f0161ff6bfe94c124ef76f6a1fba5254"  # Versão exigida pelo desafio.
    corpus_directory: Path = PROJECT_ROOT / "data" / "httpx"  # Onde o clone do HTTPX ficará salvo.
    docs_relative_path: Path = Path("docs")  # Pasta de documentação dentro do repositório HTTPX.

    # Define um modelo para busca semântica e outro para a resposta opcional.
    embedding_model: str = "qwen3-embedding:4b"  # Tag Ollama do modelo que converte texto em vetores.
    embedding_batch_size: int = 8  # Quantidade de chunks enviada ao Ollama em cada requisição de embedding.
    embedding_query_instruction: str = "Given a question about HTTPX documentation, retrieve relevant passages that directly answer the question."  # Instrução que será adicionada às perguntas.
    generator_model: str = "qwen3.5:4b"  # Tag Ollama do modelo que redige respostas.

    # Determina o tamanho e a sobreposição dos trechos pesquisáveis.
    chunk_size_words: int = 80  # Quantidade aproximada de palavras por chunk.
    chunk_overlap_words: int = 15  # Palavras repetidas entre chunks consecutivos.
    default_top_k: int = 3  # Número de resultados retornados quando nada for informado.
    min_top_k: int = 3  # Menor quantidade permitida pela regra do desafio.
    max_top_k: int = 5  # Maior quantidade permitida pela regra do desafio.
    expected_markdown_files: int = 23  # Quantidade de Markdown esperada no commit obrigatório.

    # Define onde o ChromaDB persistirá vetores e metadados dos chunks.
    chroma_directory: Path = PROJECT_ROOT / "data" / "chroma"  # Local do banco vetorial persistente.
    chroma_collection_name: str = "httpx_documentation"  # Nome da coleção que conterá o corpus HTTPX.
    metrics_directory: Path = PROJECT_ROOT / "data" / "metrics"  # Local dos tempos medidos nas consultas executadas.

    @property  # Permite acessar o caminho calculado como se fosse um atributo.
    def docs_directory(self) -> Path:  # Informa onde os arquivos Markdown serão procurados.
        """Retorna a pasta que conterá os Markdown do corpus."""
        return self.corpus_directory / self.docs_relative_path  # Junta clone HTTPX com a pasta docs.

    @property  # Expõe o caminho calculado do histórico de desempenho das consultas.
    def search_metrics_file(self) -> Path:  # Informa onde cada tempo de consulta será registrado.
        """Retorna o arquivo JSON Lines que guarda as métricas de busca."""
        return self.metrics_directory / "search_history.jsonl"  # Usa uma linha JSON para cada consulta executada.

    def create_required_directories(self) -> None:  # Prepara as pastas que serão usadas futuramente.
        """Cria somente pastas locais vazias; não baixa corpus nem modelos."""
        self.corpus_directory.mkdir(parents=True, exist_ok=True)  # Cria a pasta de dados do HTTPX.
        self.chroma_directory.mkdir(parents=True, exist_ok=True)  # Cria a pasta do banco vetorial.
        self.metrics_directory.mkdir(parents=True, exist_ok=True)  # Cria a pasta que armazenará tempos de consultas.

    def validate(self) -> None:  # Confere as regras antes de o RAG começar a executar.
        """Interrompe cedo se alguma decisão de configuração for inválida."""
        if not self.httpx_commit or len(self.httpx_commit) != 40:  # Garante um hash Git completo.
            raise ValueError("O commit do HTTPX deve ser um hash Git de 40 caracteres.")  # Explica o erro.
        if self.chunk_size_words <= 0:  # Impede chunks sem conteúdo.
            raise ValueError("chunk_size_words deve ser maior que zero.")  # Explica o erro.
        if not 0 <= self.chunk_overlap_words < self.chunk_size_words:  # Mantém overlap em faixa válida.
            raise ValueError("O overlap deve ser não negativo e menor que o chunk.")  # Explica o erro.
        if not self.min_top_k <= self.default_top_k <= self.max_top_k:  # Mantém o padrão entre os limites.
            raise ValueError("default_top_k deve estar entre min_top_k e max_top_k.")  # Explica o erro.
        if self.min_top_k < 1:  # Impede pedir zero ou quantidade negativa de resultados.
            raise ValueError("min_top_k deve ser pelo menos 1.")  # Explica o erro.


SETTINGS = RagSettings()  # Cria a configuração padrão que o projeto utilizará.
SETTINGS.validate()  # Valida imediatamente para detectar erro de configuração cedo.
