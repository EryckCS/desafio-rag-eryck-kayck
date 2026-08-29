"""Oferece uma interface de terminal para consultar o RAG sem editar código."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

import argparse  # Lê pergunta e top_k informados na linha de comando.
import json  # Registra métricas de consulta em formato estruturado.
from dataclasses import dataclass  # Cria uma estrutura para resposta e tempo da consulta.
from datetime import datetime, timezone  # Registra o momento de cada consulta em horário UTC.
from time import perf_counter  # Mede o tempo de processamento com alta precisão.

from .config import SETTINGS, RagSettings  # Importa a configuração padrão e seu tipo.
from .generator import GeneratedAnswer, GenerationError, generate_answer  # Adiciona a camada opcional de resposta fundamentada.
from .retriever import RetrievalError, SearchResult, search  # Reaproveita a busca semântica e seus resultados.


@dataclass(frozen=True)  # Impede alteração de resultados ou tempo depois que a consulta termina.
class QueryExecution:  # Agrupa pergunta, resultados e tempo de uma execução completa.
    """Guarda os dados de uma consulta executada pela interface de terminal."""

    question: str  # Pergunta informada pela pessoa usuária.
    results: tuple[SearchResult, ...]  # Resultados retornados pela busca semântica.
    elapsed_seconds: float  # Tempo total gasto para gerar embedding, consultar e formatar resultados.
    generated_answer: GeneratedAnswer | None = None  # Guarda a resposta opcional criada a partir dos resultados.


def execute_query(question: str, settings: RagSettings, top_k: int | None = None, should_generate: bool = False) -> QueryExecution:  # Executa a busca e opcionalmente gera uma resposta.
    """Retorna resultados do RAG junto do tempo total de resposta em segundos."""
    start_time = perf_counter()  # Marca o instante imediatamente antes da consulta ao RAG.
    results = search(question, settings, top_k=top_k)  # Gera embedding da pergunta e consulta o ChromaDB.
    generated_answer = generate_answer(question, results, settings) if should_generate else None  # Usa o modelo gerador somente quando solicitado.
    elapsed_seconds = perf_counter() - start_time  # Calcula quanto tempo a consulta levou até receber os resultados.
    return QueryExecution(question, results, elapsed_seconds, generated_answer)  # Devolve pergunta, resultados, resposta opcional e tempo.


def save_query_metric(execution: QueryExecution, settings: RagSettings) -> None:  # Persiste uma métrica resumida em uma linha JSON.
    """Registra pergunta, duração e fontes retornadas sem salvar conteúdo completo dos chunks."""
    settings.metrics_directory.mkdir(parents=True, exist_ok=True)  # Cria a pasta de métricas quando ainda não existe.
    record = {  # Monta apenas dados úteis para evidência de testes e desempenho.
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),  # Registra quando a consulta foi executada.
        "question": execution.question,  # Guarda a pergunta usada no teste.
        "elapsed_seconds": round(execution.elapsed_seconds, 3),  # Guarda duração arredondada a milissegundos.
        "result_count": len(execution.results),  # Guarda quantos resultados foram retornados.
        "generated_answer": execution.generated_answer is not None,  # Registra se a consulta também usou o modelo gerador.
        "sources": [result.source_path for result in execution.results],  # Guarda somente os caminhos das fontes obtidas.
    }
    with settings.search_metrics_file.open("a", encoding="utf-8") as metrics_file:  # Abre o histórico sem apagar consultas anteriores.
        metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")  # Salva uma consulta por linha para leitura simples.


def format_execution(execution: QueryExecution) -> str:  # Converte uma execução em texto claro para o terminal.
    """Exibe tempo, ranking, score, trecho, arquivo, título e seção de cada resultado."""
    lines = [  # Inicia a saída com dados gerais da consulta realizada.
        f"Pergunta: {execution.question}",  # Mostra a pergunta que gerou os resultados.
        f"Tempo de resposta: {execution.elapsed_seconds:.3f} segundos",  # Mostra a duração medida automaticamente.
        f"Resultados: {len(execution.results)}",  # Mostra a quantidade de chunks retornados.
    ]
    for result in execution.results:  # Percorre os resultados na ordem de relevância já calculada.
        lines.extend(  # Adiciona todos os campos obrigatórios solicitados na saída do desafio.
            [
                "",  # Separa visualmente um resultado do próximo.
                f"{result.rank}. Score: {result.score:.4f}",  # Mostra ranking e score de similaridade.
                f"Arquivo: {result.source_path}",  # Mostra o caminho da fonte original.
                f"Título: {result.title}",  # Mostra o título principal do documento.
                f"Seção: {result.section}",  # Mostra a seção específica do trecho.
                f"Chunk: {result.chunk_id}",  # Mostra o identificador rastreável do trecho.
                "Trecho:",  # Identifica o início do texto recuperado.
                result.text,  # Mostra o conteúdo completo do chunk retornado.
            ]
        )
    if execution.generated_answer is not None:  # Exibe a resposta somente quando a geração foi solicitada.
        lines.extend(  # Separa a resposta final dos chunks mostrados como evidência.
            [
                "",  # Cria uma linha em branco antes da resposta gerada.
                "Resposta gerada:",  # Identifica claramente o texto produzido pelo modelo local.
                execution.generated_answer.text,  # Mostra a resposta baseada nos trechos recuperados.
                "Fontes da resposta:",  # Identifica os arquivos usados como contexto pelo modelo.
                *[f"- {source}" for source in execution.generated_answer.sources],  # Lista cada caminho de fonte uma única vez.
            ]
        )
    return "\n".join(lines)  # Junta todas as linhas em uma saída pronta para imprimir.


def build_parser() -> argparse.ArgumentParser:  # Cria a definição dos argumentos aceitos pelo terminal.
    """Retorna o parser da interface de consulta do Mini-RAG."""
    parser = argparse.ArgumentParser(description="Consulta semântica na documentação HTTPX.")  # Define texto de ajuda da interface.
    parser.add_argument("--question", "-q", help="Pergunta a ser respondida pela documentação HTTPX.")  # Aceita pergunta sem editar código.
    parser.add_argument("--top-k", type=int, default=SETTINGS.default_top_k, help="Quantidade de resultados entre 3 e 5.")  # Aceita quantidade de resultados.
    parser.add_argument("--no-save-metric", action="store_true", help="Não registra a duração desta consulta no histórico local.")  # Permite desativar o histórico opcionalmente.
    parser.add_argument("--generate", action="store_true", help="Gera uma resposta final com qwen3.5:4b usando os trechos recuperados.")  # Ativa a etapa opcional de geração fundamentada.
    return parser  # Devolve o parser configurado para a função principal.


def main() -> int:  # Coordena leitura da pergunta, busca, registro e impressão da resposta.
    """Executa a interface de terminal e devolve código de saída apropriado."""
    arguments = build_parser().parse_args()  # Lê os argumentos informados após o comando Python.
    question = arguments.question or input("Digite sua pergunta sobre HTTPX: ")  # Usa pergunta interativa quando não há argumento.
    try:  # Converte erros de entrada ou busca em mensagem clara no terminal.
        execution = execute_query(question, SETTINGS, top_k=arguments.top_k, should_generate=arguments.generate)  # Executa a busca e, se pedido, a geração cronometrada.
        if not arguments.no_save_metric:  # Verifica se a pessoa usuária permitiu registrar a métrica local.
            save_query_metric(execution, SETTINGS)  # Armazena duração e fontes retornadas no histórico.
        print(format_execution(execution))  # Mostra resultados completos e rastreáveis no terminal.
        return 0  # Indica execução concluída sem erro.
    except (RetrievalError, GenerationError) as error:  # Captura entrada inválida, busca ou geração indisponível.
        print(f"Erro: {error}")  # Mostra mensagem compreensível sem traceback técnico.
        return 2  # Indica erro de entrada ou consulta para o terminal.


if __name__ == "__main__":  # Permite executar o módulo diretamente pelo comando python -m.
    raise SystemExit(main())  # Encerra o processo usando o código retornado pela função principal.
