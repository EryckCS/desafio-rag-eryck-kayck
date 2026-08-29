"""Testa a geração opcional sem depender de um servidor Ollama ativo."""

from __future__ import annotations  # Permite usar anotações de tipo modernas.

import unittest  # Executa os testes automatizados do módulo de geração.
from unittest.mock import patch  # Substitui o modelo local por um objeto controlado no teste.

from src.rag_httpx.config import RagSettings  # Cria configurações isoladas para os cenários de teste.
from src.rag_httpx.generator import GenerationError, build_context, create_chat_model, generate_answer  # Importa funções públicas do módulo testado.
from src.rag_httpx.retriever import SearchResult  # Cria evidências equivalentes às retornadas pela busca.


class FakeResponse:  # Representa a resposta mínima devolvida por um modelo LangChain.
    def __init__(self, content: str) -> None:  # Guarda o texto que o modelo falso deverá devolver.
        self.content = content  # Expõe o mesmo atributo usado pelo código de produção.


class FakeModel:  # Simula um modelo de chat sem realizar inferência local.
    def __init__(self) -> None:  # Prepara uma lista para verificar o prompt recebido.
        self.prompts: list[str] = []  # Registra cada prompt enviado ao modelo falso.

    def invoke(self, prompt: str) -> FakeResponse:  # Simula a chamada LangChain ao modelo de chat.
        self.prompts.append(prompt)  # Guarda o prompt para validação do teste.
        return FakeResponse("Use o argumento params= para enviar parâmetros na URL.")  # Devolve uma resposta fundamentada simulada.


def make_result(rank: int, source_path: str) -> SearchResult:  # Cria um resultado completo sem acessar o ChromaDB.
    return SearchResult(  # Reproduz os campos exigidos da busca semântica.
        rank=rank,  # Define a posição do resultado.
        score=0.9,  # Define um score ilustrativo de relevância.
        text="Use params= para adicionar query parameters à URL.",  # Define a evidência que irá para o contexto.
        chunk_id=f"{source_path}::chunk-{rank:04d}",  # Define um identificador rastreável.
        source_path=source_path,  # Define o arquivo Markdown de origem.
        title="QuickStart",  # Define o título principal do documento.
        section="Passing Parameters in URLs",  # Define a seção da evidência.
    )


class GeneratorTests(unittest.TestCase):  # Agrupa cenários de geração opcional fundamentada.
    def test_chat_model_disables_reasoning_output(self) -> None:  # Confirma que a configuração evita exibir blocos de raciocínio.
        model = create_chat_model(RagSettings())  # Cria o adaptador sem executar uma geração real.
        self.assertFalse(model.reasoning)  # Confirma a opção equivalente a think=False no LangChain.

    def test_context_preserves_text_and_source_metadata(self) -> None:  # Confirma que o modelo recebe texto e referências da evidência.
        context = build_context((make_result(1, "docs/quickstart.md"),))  # Monta contexto com um resultado conhecido.
        self.assertIn("docs/quickstart.md", context)  # Confirma o caminho da fonte no contexto.
        self.assertIn("Passing Parameters in URLs", context)  # Confirma a seção da fonte no contexto.
        self.assertIn("params=", context)  # Confirma o texto recuperado no contexto.

    def test_generation_returns_answer_and_unique_sources(self) -> None:  # Confirma resposta e remoção de fontes repetidas.
        fake_model = FakeModel()  # Cria um modelo controlado para o teste.
        results = (make_result(1, "docs/quickstart.md"), make_result(2, "docs/quickstart.md"))  # Cria dois chunks da mesma fonte.
        with patch("src.rag_httpx.generator.create_chat_model", return_value=fake_model):  # Evita executar o Ollama no teste.
            answer = generate_answer("Como enviar parâmetros?", results, RagSettings())  # Gera resposta a partir de evidências simuladas.
        self.assertIn("params=", answer.text)  # Confirma o texto devolvido pelo modelo falso.
        self.assertEqual(answer.sources, ("docs/quickstart.md",))  # Confirma que fontes repetidas aparecem apenas uma vez.
        self.assertIn("Como enviar parâmetros?", fake_model.prompts[0])  # Confirma que a pergunta entrou no prompt.

    def test_generation_rejects_empty_results(self) -> None:  # Confirma erro claro quando não há contexto para fundamentar resposta.
        with self.assertRaisesRegex(GenerationError, "Não há trechos recuperados"):  # Exige mensagem útil para o caso inválido.
            generate_answer("Como enviar parâmetros?", (), RagSettings())  # Tenta gerar sem resultados recuperados.
