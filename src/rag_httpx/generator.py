"""Gera respostas opcionais usando somente os trechos recuperados do RAG."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

from dataclasses import dataclass  # Cria uma estrutura imutável para a resposta gerada.

from langchain_ollama import ChatOllama  # Conecta o modelo gerador local do Ollama ao LangChain.

from .config import RagSettings  # Importa o nome do modelo e demais configurações do projeto.
from .retriever import SearchResult  # Reaproveita os chunks e fontes retornados pela busca semântica.


class GenerationError(ValueError):  # Representa uma falha compreensível durante a geração opcional.
    """Indica que a resposta não pôde ser gerada com o contexto recuperado."""


@dataclass(frozen=True)  # Impede alterar resposta ou fontes depois de concluída a geração.
class GeneratedAnswer:  # Agrupa o texto final com as fontes realmente entregues ao modelo.
    """Guarda uma resposta fundamentada e os arquivos utilizados como contexto."""

    text: str  # Resposta final devolvida pelo modelo gerador.
    sources: tuple[str, ...]  # Caminhos Markdown únicos usados para fundamentar a resposta.


def create_chat_model(settings: RagSettings) -> ChatOllama:  # Cria o adaptador LangChain para o modelo gerador local.
    """Retorna o Qwen configurado para responder de forma determinística e sem bloco de raciocínio."""
    return ChatOllama(  # Configura o chat model somente quando a geração é solicitada.
        model=settings.generator_model,  # Seleciona a tag local qwen3.5:4b definida na configuração.
        temperature=0,  # Reduz variações para tornar respostas técnicas mais consistentes.
        reasoning=False,  # Solicita apenas a resposta final, equivalente a think=False no adaptador LangChain instalado.
    )


def _unique_sources(results: tuple[SearchResult, ...]) -> tuple[str, ...]:  # Remove caminhos repetidos preservando a primeira ocorrência.
    return tuple(dict.fromkeys(result.source_path for result in results))  # Usa a ordem de ranking para organizar as fontes.


def build_context(results: tuple[SearchResult, ...]) -> str:  # Converte cada chunk em contexto identificável para o modelo.
    """Inclui texto e metadados de origem para limitar a resposta ao material recuperado."""
    context_parts: list[str] = []  # Acumula blocos de contexto sem perder a separação entre fontes.
    for result in results:  # Percorre os resultados já ordenados por relevância.
        context_parts.append(  # Registra arquivo, título, seção e texto de cada evidência.
            "\n".join(  # Mantém os campos legíveis e úteis para a resposta fundamentada.
                [
                    f"[Fonte: {result.source_path}]",  # Identifica o arquivo Markdown de origem.
                    f"[Título: {result.title}]",  # Identifica o título principal do documento.
                    f"[Seção: {result.section}]",  # Identifica a seção exata do trecho.
                    result.text,  # Insere o conteúdo original recuperado no banco vetorial.
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)  # Separa os chunks para impedir mistura acidental das fontes.


def build_prompt(question: str, context: str) -> str:  # Monta as instruções que restringem a geração ao contexto recuperado.
    """Pede uma resposta objetiva em português e impede a invenção de informações."""
    return (  # Forma uma única instrução completa para o modelo local.
        "Responda à pergunta usando exclusivamente o CONTEXTO da documentação HTTPX abaixo.\n"  # Define a regra principal de fundamentação.
        "Escreva em português, de forma objetiva e tecnicamente correta.\n"  # Define idioma e estilo da resposta.
        "Se o contexto não contiver informação suficiente, diga claramente que a documentação recuperada não responde à pergunta.\n"  # Define como agir fora de escopo.
        "Não invente APIs, comportamentos, parâmetros ou fontes.\n\n"  # Proíbe conteúdo sem evidência.
        f"PERGUNTA:\n{question}\n\n"  # Inclui a pergunta original.
        f"CONTEXTO:\n{context}"  # Inclui somente chunks provenientes da busca semântica.
    )


def _message_text(content: object) -> str:  # Normaliza o conteúdo da resposta LangChain em texto simples.
    if isinstance(content, str):  # Trata a forma normal de resposta de modelos de chat.
        return content.strip()  # Remove espaços extras antes de validar o texto.
    if isinstance(content, list):  # Trata provedores que devolvem partes de conteúdo em lista.
        return "".join(str(part) for part in content).strip()  # Une as partes para exibir uma única resposta final.
    return str(content).strip()  # Converte formatos inesperados para texto verificável.


def generate_answer(question: str, results: tuple[SearchResult, ...], settings: RagSettings) -> GeneratedAnswer:  # Gera uma resposta baseada nos resultados recuperados.
    """Devolve resposta final e fontes sem modificar o ranking ou os chunks da busca."""
    cleaned_question = question.strip()  # Remove espaços que não agregam significado à pergunta.
    if not cleaned_question:  # Impede enviar uma pergunta vazia ao modelo de chat.
        raise GenerationError("A pergunta não pode estar vazia para gerar uma resposta.")  # Informa a correção necessária.
    if not results:  # Impede geração sem nenhuma evidência recuperada.
        raise GenerationError("Não há trechos recuperados para fundamentar a resposta.")  # Explica que a busca deve retornar contexto.
    try:  # Converte indisponibilidade do Ollama ou do modelo em mensagem clara.
        model = create_chat_model(settings)  # Inicializa o Qwen somente quando a geração foi solicitada.
        response = model.invoke(build_prompt(cleaned_question, build_context(results)))  # Envia pergunta e evidências ao modelo local.
    except Exception as error:  # Captura falhas de conexão, modelo ausente ou execução local.
        raise GenerationError(f"Não foi possível gerar a resposta: {error}") from error  # Preserva a causa em uma mensagem contextualizada.
    answer_text = _message_text(response.content)  # Extrai somente o texto final retornado pelo modelo.
    if not answer_text:  # Trata respostas vazias para não apresentar sucesso enganoso.
        raise GenerationError("O modelo gerador retornou uma resposta vazia.")  # Indica que nenhuma resposta utilizável foi recebida.
    return GeneratedAnswer(answer_text, _unique_sources(results))  # Mantém a resposta acompanhada das fontes recuperadas.
