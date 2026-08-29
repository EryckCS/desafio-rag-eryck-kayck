"""Obtém e valida o corpus obrigatório da documentação HTTPX."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

from dataclasses import dataclass  # Cria uma estrutura simples para o resumo do corpus.
from pathlib import Path  # Representa caminhos de modo portátil.
from subprocess import CalledProcessError, run  # Executa os comandos Git necessários.

from .config import RagSettings  # Importa as configurações centralizadas do projeto.


class CorpusError(RuntimeError):  # Representa um problema compreensível na obtenção do corpus.
    """Indica que o corpus HTTPX não está disponível ou não é o esperado."""


@dataclass(frozen=True)  # Cria um resumo imutável que poderá ser exibido no terminal.
class CorpusSummary:  # Agrupa as evidências de que o corpus está correto.
    """Guarda caminho, commit e arquivos Markdown descobertos."""

    repository_path: Path  # Local onde o repositório HTTPX foi salvo.
    commit: str  # Hash do commit que está realmente em uso.
    markdown_files: tuple[Path, ...]  # Arquivos Markdown encontrados de forma recursiva.


def _run_git(arguments: list[str], repository_path: Path | None = None) -> str:  # Executa Git e devolve a saída.
    command = ["git"]  # Inicia o comando pelo executável Git.
    if repository_path is not None:  # Verifica se o comando deve ser executado dentro de um repositório.
        command.extend(["-C", str(repository_path)])  # Informa ao Git a pasta do repositório.
    command.extend(arguments)  # Acrescenta a operação Git pedida pela função chamadora.
    try:  # Converte falhas técnicas em uma mensagem mais clara para o projeto.
        completed_process = run(command, check=True, capture_output=True, text=True)  # Executa sem abrir shell.
    except CalledProcessError as error:  # Captura falha de clone, checkout ou leitura do commit.
        detail = error.stderr.strip() or error.stdout.strip() or str(error)  # Escolhe a melhor descrição disponível.
        raise CorpusError(f"Falha ao executar Git: {detail}") from error  # Explica o erro mantendo a causa original.
    return completed_process.stdout.strip()  # Remove espaços extras e devolve a saída do Git.


def obtain_httpx_repository(settings: RagSettings) -> Path:  # Baixa o HTTPX ou confirma que ele já existe.
    """Obtém o repositório HTTPX e fixa exatamente o commit do desafio."""
    repository_path = settings.corpus_directory  # Usa o caminho configurado para armazenar o corpus.
    if not repository_path.exists():  # Baixa o repositório somente quando ele ainda não existe.
        repository_path.parent.mkdir(parents=True, exist_ok=True)  # Cria a pasta data quando necessário.
        _run_git(["clone", settings.httpx_repository_url, str(repository_path)])  # Clona o repositório oficial.
    if not (repository_path / ".git").is_dir():  # Confere se o caminho encontrado é realmente um clone Git.
        raise CorpusError(f"O caminho do corpus não é um repositório Git: {repository_path}")  # Evita usar dados errados.
    _run_git(["checkout", settings.httpx_commit], repository_path)  # Fixa a versão exata exigida pelo desafio.
    return repository_path  # Devolve o caminho pronto para leitura dos documentos.


def discover_markdown_files(docs_directory: Path) -> tuple[Path, ...]:  # Localiza os Markdown do corpus recursivamente.
    """Retorna todos os arquivos .md dentro de docs e de suas subpastas."""
    if not docs_directory.is_dir():  # Confere se a pasta docs existe antes da busca.
        raise CorpusError(f"Pasta de documentação não encontrada: {docs_directory}")  # Explica a origem do problema.
    markdown_files = tuple(sorted(docs_directory.rglob("*.md")))  # Busca Markdown também nas subpastas.
    if not markdown_files:  # Impede que o restante do RAG rode com corpus vazio.
        raise CorpusError("Nenhum arquivo Markdown foi encontrado no corpus.")  # Informa o erro de forma clara.
    return markdown_files  # Mantém a lista ordenada para resultados reproduzíveis.


def validate_httpx_corpus(settings: RagSettings) -> CorpusSummary:  # Produz a evidência de que o corpus está certo.
    """Valida commit e contagem de Markdown antes de iniciar o processamento."""
    repository_path = settings.corpus_directory  # Lê o caminho configurado para o clone HTTPX.
    if not (repository_path / ".git").is_dir():  # Garante que o clone foi obtido antes da validação.
        raise CorpusError("Repositório HTTPX ausente. Execute obtain_httpx_repository primeiro.")  # Orienta o próximo passo.
    current_commit = _run_git(["rev-parse", "HEAD"], repository_path)  # Descobre o commit atualmente selecionado.
    if current_commit != settings.httpx_commit:  # Impede que documentos de outra versão sejam indexados.
        raise CorpusError(f"Commit incorreto: {current_commit}. Esperado: {settings.httpx_commit}")  # Mostra ambos os hashes.
    markdown_files = discover_markdown_files(settings.docs_directory)  # Encontra os documentos obrigatórios.
    if len(markdown_files) != settings.expected_markdown_files:  # Usa a contagem fornecida como verificação.
        raise CorpusError(  # Explica a divergência de documentos encontrada.
            f"Foram encontrados {len(markdown_files)} Markdown; esperados: {settings.expected_markdown_files}."
        )
    return CorpusSummary(repository_path, current_commit, markdown_files)  # Devolve as evidências validadas.
