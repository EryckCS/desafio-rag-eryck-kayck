# Mini-RAG da documentação HTTPX

## Identificação

- Nome do aluno: Eryck Kayck
- Formato da solução: script de terminal em Python
- Link do vídeo: https://youtu.be/loZ-q-XdwEA?si=AVGi0-o1BcJrA-eQ
- Link do Colab: não se aplica; o projeto é executado localmente

## Objetivo

Este projeto implementa o núcleo de recuperação de um RAG sobre a documentação do HTTPX. Ele encontra os arquivos Markdown do repositório exigido, cria chunks com metadados, gera embeddings locais e recupera os trechos mais relevantes para uma pergunta. Opcionalmente, também gera uma resposta em linguagem natural fundamentada nesses trechos.

## Arquitetura resumida

```text
repositório HTTPX → arquivos Markdown → Documents LangChain → chunks + metadados
→ embeddings qwen3-embedding:4b → ChromaDB com similaridade cosseno
→ trechos relevantes + fontes → resposta opcional qwen3.5:4b
```

## Como executar do zero

### Pré-requisitos

- Python 3.11 ou superior; o projeto foi validado com Python 3.13.
- Git instalado e disponível no terminal.
- Ollama instalado e em execução no computador.
- Espaço em disco para os modelos locais e memória suficiente para executá-los.

Baixe os dois modelos locais necessários:

```powershell
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
```

Os modelos usados ocupavam aproximadamente 3,4 GB (`qwen3.5:4b`) e 2,5 GB
(`qwen3-embedding:4b`) no ambiente de validação. O caminho é gratuito e não
usa API key, mas pode ser relativamente pesado para computadores mais simples.
Não exige GPU, porém a velocidade e a viabilidade dependem da RAM, CPU e da
configuração local do Ollama. Para uma entrega em máquina pouco potente, este é
um risco a considerar e explicar, não uma garantia de desempenho mínimo.

### Instalação e preparação

No terminal integrado do VS Code, abra a pasta raiz do projeto:

```powershell
cd "C:\Users\eryck\OneDrive\Documentos\eryck_kayck_Rag"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare o corpus e o banco vetorial com um único comando:

```powershell
.\.venv\Scripts\python.exe -m src.rag_httpx.indexer
```

Esse comando clona o repositório HTTPX, fixa o commit `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, valida os 23 arquivos Markdown em `httpx/docs/`, cria os chunks e grava os embeddings no ChromaDB.

### Fazer uma pergunta

Para buscar os trechos relevantes sem gerar uma resposta adicional:

```powershell
.\.venv\Scripts\python.exe -m src.rag_httpx.cli --question "Como enviar parâmetros na URL com HTTPX?" --top-k 3
```

Para digitar a pergunta de forma interativa e também gerar uma resposta final:

```powershell
.\.venv\Scripts\python.exe -m src.rag_httpx.cli --generate
```

O terminal solicitará a pergunta. A opção `--generate` é opcional; sem ela, o projeto executa somente a recuperação de trechos e fontes, que é o requisito principal do desafio.

Para executar os testes automatizados:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Decisões técnicas

### Chunking

- Estratégia: `MarkdownHeaderTextSplitter` seguido de `RecursiveCharacterTextSplitter`, ambos do LangChain.
- Tamanho aproximado: 80 palavras por chunk.
- Overlap: 15 palavras entre chunks consecutivos.
- Justificativa: o intervalo de 60 a 90 palavras foi sugerido pelo desafio. O tamanho de 80 preserva contexto sem misturar muitos assuntos, enquanto o overlap reduz cortes bruscos entre trechos consecutivos. Os cabeçalhos Markdown são preservados para identificar a seção de cada chunk.

### Embeddings e busca

- Modelo: `qwen3-embedding:4b`, executado localmente pelo Ollama.
- Técnica: embeddings compatíveis para documentos e perguntas, armazenados no ChromaDB com espaço de similaridade cosseno.
- Valor de `top_k`: 3 por padrão, com limite de 3 a 5 resultados.
- Justificativa: o desafio exige de 3 a 5 itens e sugere começar com 3. Essa quantidade apresenta evidências suficientes sem poluir a saída.

O índice reconstruído durante a validação contém 23 documentos e 318 chunks.

### Metadados e fontes

Cada chunk preserva `source_path`, `title`, `section`, `chunk_id`, `chunk_index` e `word_count`. Depois da busca, esses campos são exibidos junto com o score e o texto recuperado; assim, cada resultado pode ser rastreado até o arquivo e a seção de origem.

### Geração opcional

Quando `--generate` é usado, o `qwen3.5:4b` recebe apenas a pergunta e os chunks recuperados. O prompt proíbe inventar informações e pede que o modelo admita quando o contexto não for suficiente. O adaptador LangChain é configurado com `reasoning=False`, equivalente a desativar a saída de `think`, para exibir somente a resposta final. As fontes usadas também são listadas após a resposta.

## Perguntas de teste

### 1. Pergunta com resposta clara

- Pergunta: `Como enviar parâmetros na URL com HTTPX?`
- Resultado esperado: menção ao argumento `params=` e ao envio de parâmetros de query na URL.
- Resultado observado: os dois primeiros resultados vieram de `docs/quickstart.md`, seção `Passing Parameters in URLs`, com scores 0.7685 e 0.7378. O terceiro veio da seção `Query Parameters` em `docs/compatibility.md`, com score 0.6954.
- O resultado foi relevante: sim. Os trechos apresentam diretamente o uso de `params=` e exemplos de múltiplos valores.
- Tempo observado: o mesmo teste levou 1,163 s em uma execução e 15,480 s em
  outra, sem alteração dos resultados ou scores. A duração varia conforme o
  carregamento do Ollama, o modelo estar ou não na memória e os recursos locais.

### 2. Pergunta ampla ou ambígua

- Pergunta: `Como configurar timeouts e reutilizar conexões no HTTPX?`
- Resultado esperado: documentação de timeout, cliente e pool de conexões.
- Resultado observado: o primeiro resultado foi `Fine tuning the configuration` em `docs/advanced/timeouts.md` (score 0.7235), seguido por `Timeouts` em `docs/quickstart.md` (0.7171) e pela extensão `"timeout"` em `docs/advanced/extensions.md` (0.7116).
- O resultado foi relevante: parcialmente. A busca recuperou corretamente configurações de timeout e pool de conexões, mas a pergunta reúne dois temas e pode exigir uma consulta mais específica para aprofundar reutilização de conexões.
- Tempo observado: 0,853 s em uma execução sem geração opcional. Esse valor é
  apenas evidência do ambiente de teste e pode variar nas próximas execuções.

### 3. Pergunta fora do escopo

- Pergunta: `Qual é a capital do Brasil?`
- Como o sistema reagiu: retornou três trechos do HTTPX, com score máximo baixo de 0.4707, porque uma busca por similaridade sempre devolve os itens mais próximos dentro do corpus disponível.
- Como a reação poderia melhorar: adicionar um limite mínimo de score, por exemplo configurável, para responder que não há evidência suficiente quando a maior similaridade estiver abaixo desse limite. Essa melhoria não foi aplicada para não assumir um limiar sem avaliação adicional.
- Tempo observado: 0,840 s em uma execução sem geração opcional. Esse valor é
  apenas evidência do ambiente de teste e pode variar nas próximas execuções.

## Testes executados

- Validação do corpus: commit obrigatório confirmado e 23 arquivos Markdown encontrados recursivamente em `docs/`.
- Reconstrução real do ChromaDB: 318 chunks indexados.
- Busca real: as três perguntas acima foram executadas e seus resultados foram registrados nesta documentação.
- Testes automatizados: 23 testes aprovados com o comando de testes indicado acima. Eles cobrem configuração, descoberta, leitura, metadados, chunking, embeddings, busca, erros de entrada, geração opcional, interface de terminal e preparação completa do índice com dependências simuladas.

## Limitações conhecidas

- A recuperação semântica atual sempre devolve de 3 a 5 resultados, inclusive para perguntas fora do escopo. Um limiar de score é uma melhoria futura.
- A qualidade e o tempo de resposta dependem do desempenho local do Ollama, especialmente quando a geração opcional está ativada.
- Embora o projeto seja gratuito e não use API key, os dois modelos locais de
  4B exigem cerca de 5,9 GB de arquivos baixados e recursos locais para rodar.
  Assim, esta escolha é mais pesada que o caminho mínimo sugerido no desafio e
  pode não ser adequada para computadores com pouca memória ou CPU limitada.
- O modelo gerador pode apresentar pequenos problemas de formatação textual; os chunks e fontes originais continuam exibidos para permitir conferência.
- O projeto é uma consulta de uma pergunta por execução, sem memória de uma conversa contínua, pois esse recurso não é necessário para o desafio.

## Uso de ferramentas de IA

- Ferramenta utilizada: Codex.
- Tarefas em que ajudou: Programação, criação de testes e interpretação de resultados locais.
- Exemplo representativo de orientação: criar um RAG que siga o fluxo de repositório, Markdown, chunks com metadados, embeddings, busca semântica e fontes; usar modelos locais Qwen pelo Ollama.
- O que foi testado, modificado ou validado: o commit e a contagem de arquivos foram verificados; o índice foi reconstruído; as três perguntas obrigatórias foram executadas; e os 23 testes automatizados foram aprovados.

## Referências e código externo

- Enunciado e modelos de entrega fornecidos para a prova prática.
- [Repositório HTTPX](https://github.com/encode/httpx).
- [Documentação do LangChain](https://python.langchain.com/docs/).
- [Integração LangChain com Ollama](https://python.langchain.com/docs/integrations/text_embedding/ollama/).
- [Integração LangChain com Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/).
- [Ollama](https://ollama.com/).
- [ChromaDB](https://www.trychroma.com/).

## Segurança

- [x] Minha solução não usa API key.

Não há tokens, senhas ou chaves no código. Os modelos são executados localmente pelo Ollama, e os dados indexados são documentação pública do HTTPX.

## Checklist antes do envio

- [x] Código-fonte, dependências e instruções de execução estão incluídos.
- [x] Commit HTTPX e 23 arquivos Markdown foram validados.
- [x] Trechos, fontes, ranking, score e metadados são exibidos.
- [x] Três tipos de pergunta foram testados e documentados.
- [x] Limitações conhecidas foram registradas.
- [x] Uso de IA, referências e segurança foram declarados.
- [x] Vídeo de apresentação gravado e link inserido no início deste README.
- [x] Vídeo confirmado em janela anônima, com acesso público e duração dentro do máximo permitido.
