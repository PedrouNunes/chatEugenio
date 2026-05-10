# Eugênio -AI Chat

Chatbot baseado em agentes de IA, desenvolvido como protótipo para apoio à criação de teclados para o sistema de **Comunicação Aumentativa e Alternativa (CAA)**.

Construído com **smolagents**, **FastAPI** e **HuggingFace**, o Eugênio oferece dois modos de interação: um agente simples para conversas gerais e o **Alfred**, um agente avançado com RAG e múltiplas ferramentas.

---

## Arquitetura

```
chat.html  ->  FastAPI (main.py)  ->  smolagents  ->  Qwen2.5 (HuggingFace)
 Frontend        Backend API          Agente IA        Modelo LLM remoto
```

### Arquivos principais

```
chat_api/
├── main.py              # Servidor FastAPI -expõe /chat, /alfred e /keyboard
├── agent.py             # Agente simples (DuckDuckGo + fuso horário)
├── alfred.py            # Agente Alfred (RAG + clima + notícias + HuggingFace Hub)
├── keyboard_agent.py    # Gerador de teclados .tec com IA
└── chat.html            # Interface do usuário
```

---

## Pré-requisitos

- Python 3.10+
- Conta no [HuggingFace](https://huggingface.co) com token de acesso
- Conexão com a internet (o modelo roda remotamente)

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo/chat_api
```

### 2. Instalar dependências

```bash
pip install fastapi uvicorn smolagents pytz
pip install langchain langchain-community datasets rank_bm25
```

### 3. Obter o token do HuggingFace

1. Acesse [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Clique em **New token**
3. Escolha o tipo **Read**
4. Copie o token gerado -começa com `hf_...`

---

## Como rodar

### Passo 1 -Definir o token no terminal

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_seu_token_aqui"
```

**Mac/Linux:**
```bash
export HF_TOKEN="hf_seu_token_aqui"
```

> Este passo é necessário sempre que abrir um terminal novo.

### Passo 2 -Iniciar o servidor

```bash
cd chat_api
uvicorn main:app --reload --port 8000
```

O terminal deve exibir:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

### Passo 3 -Abrir o chat

Abra o arquivo `chat.html` diretamente no navegador (duplo clique).

---

## Como usar

### Modo Simple

Agente rápido para perguntas gerais. Tem acesso a:
- Pesquisa web (DuckDuckGo)
- Fuso horário em tempo real

**Exemplos:**
```
What time is it in Tokyo?
What are the latest news about artificial intelligence?
What is machine learning?
```

### Modo Alfred

Agente avançado com planejamento e múltiplas ferramentas. Tem acesso a:
- Pesquisa web
- Informação meteorológica
- Notícias recentes
- Estatísticas do HuggingFace Hub
- Base de dados RAG de convidados

**Exemplos:**
```
Tell me about Lady Ada Lovelace
What's the weather like in Paris tonight?
What is the most popular model from Google on HuggingFace?
Give me the latest news about robotics
```

### Modo Teclado AAC

O modo principal da dissertação. Permite criar arquivos `.tec` para o sistema Eugênio através de linguagem natural.

**Como usar:**
1. Clique em **Teclado** na interface
2. Descreva o teclado que deseja criar em português
3. Clique em **Download teclado.tec**
4. Importe o arquivo no sistema Eugênio

**Exemplos de prompts:**
```
Quero um teclado simples com as vogais A E I O U e um botão de apagar
Cria um teclado com números de 1 a 10 e um botão de espaço
Quero um teclado com saudações: Olá, Bom dia, Boa tarde, Obrigado
Cria um teclado de emoções: Feliz, Triste, Com fome, Com sono, Com dor
```

**Como funciona internamente:**
```
Descrição em linguagem natural
        |
FastAPI recebe em POST /keyboard
        |
Qwen2.5 lê o exemplo do formato .tec e gera um novo
        |
Arquivo .tec devolvido para download
        |
Importar no sistema Eugênio AAC
```

#### Usar um teclado existente como referência

É possível carregar um arquivo `.tec` existente como base para o modelo gerar variações com o mesmo estilo e estrutura.

1. Clique no botão de upload ao lado do campo de texto
2. Selecione um arquivo `.tec` do seu computador
3. O sistema muda automaticamente para o modo Teclado
4. Descreva o que deseja criar com base nesse teclado

**Exemplos com referência:**
```
Cria um teclado igual mas só com as vogais
Mantém a estrutura mas adiciona uma linha com pontuação
Usa o mesmo formato mas com palavras de comunicação básica
```

> O modelo usa o teclado carregado como exemplo de formato e estilo, garantindo compatibilidade com o sistema Eugênio.

#### Formato do arquivo `.tec`

O Eugênio usa um formato de texto simples:

```
LINHA nome;;;da;;;linha
GRUPO nome;;;do;;;grupo
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL [imagem] [label] [valor] 1 -1 -1
```

- **`;;;`** -separador de espaços nos nomes
- **`TECLA_VAZIA`** -célula vazia obrigatória no início de cada linha
- **`TECLA_NORMAL`** -tecla clicável com imagem, label e valor
- **`<--->`** -indica tecla de ação (sem imagem própria)
- **`[Nome-Da-Acao]`** -ações especiais como `[Synthesize-Word-To-Speech]`

## Como a geração de teclados funciona com IA

Esta é a parte central da dissertação. A ideia é simples: o usuário descreve em português o que quer, e o sistema entrega um arquivo pronto para ser usado no Eugênio AAC. Por baixo, há uma cadeia de componentes que tornam isso possível.

### 1. O usuário descreve o teclado

Tudo começa com uma mensagem em linguagem natural no chat, por exemplo:

```
Quero um teclado com as vogais A E I O U e um botão de apagar
```

Não é necessário saber nada sobre o formato `.tec` ou sobre programação.

### 2. O frontend envia para o backend

O `chat.html` detecta que o modo ativo é o **Teclado AAC** e envia a descrição para o endpoint correto:

```
POST http://localhost:8000/keyboard
{ "description": "Quero um teclado com as vogais A E I O U e um botão de apagar" }
```

Se o usuário carregou um arquivo `.tec` de referência, ele é incluído no mesmo pedido como `reference_keyboard`.

### 3. O backend monta o prompt para o modelo

O `keyboard_agent.py` não usa um agente com múltiplos passos aqui -chama o modelo diretamente com um **prompt de sistema** cuidadosamente construído. Esse prompt contém:

- As regras do formato `.tec` (LINHA, GRUPO, TECLA, separadores)
- Um exemplo real de teclado válido
- A descrição que o usuário enviou
- A instrução de retornar apenas o conteúdo do arquivo, sem explicações

Isso é chamado de **prompt engineering** -a forma como o prompt é escrito determina a qualidade e o formato da resposta.

### 4. O modelo de linguagem gera o arquivo

O prompt é enviado ao modelo **Qwen2.5-Coder-32B-Instruct** hospedado no HuggingFace. Este é um Large Language Model (LLM) especializado em código, com 32 bilhões de parâmetros.

O modelo lê o exemplo fornecido, entende o padrão do formato `.tec` e gera um novo arquivo seguindo as mesmas regras, adaptado à descrição do usuário. Ele não foi treinado especificamente para o Eugênio -ele aprende o formato apenas pelo exemplo dado no prompt, no momento da chamada. Essa técnica é chamada de **few-shot learning**.

### 5. O backend devolve o arquivo para download

O conteúdo gerado pelo modelo é devolvido ao frontend como um arquivo para download, sem nenhuma transformação adicional. O FastAPI define o nome `teclado.tec` e o tipo de conteúdo correto para que o navegador faça o download automaticamente.

### 6. O usuário importa no sistema Eugênio

O arquivo `.tec` baixado pode ser importado diretamente no software Eugênio AAC, onde o teclado gerado estará pronto para uso.

### Relação com o chat

A geração de teclados é um caso especializado do mesmo padrão usado nos modos Simple e Alfred: o usuário escreve em linguagem natural, o sistema interpreta e age. A diferença é que em vez de responder com texto, o sistema produz um arquivo estruturado num formato específico.

Isso demonstra que a mesma arquitetura de chat pode ser adaptada para diferentes domínios -no caso da dissertação, para o domínio da CAA -bastando alterar as ferramentas disponíveis e o prompt de sistema.

```
Modo Simple    ->  responde com texto
Modo Alfred    ->  responde com texto + consulta base de dados
Modo Teclado   ->  responde com arquivo .tec estruturado
     |
     |__ mesmo modelo (Qwen2.5)
     |__ mesmo backend (FastAPI)
     |__ mesmo frontend (chat.html)
     |__ prompt de sistema diferente
```

### Memória de conversa

Ative o toggle **Memória** na interface para que o agente se lembre do contexto entre mensagens.

---

## API

O servidor expõe os seguintes endpoints REST. Todos podem ser testados em: **http://localhost:8000/docs**

### `POST /chat`
```json
// Request
{ "message": "What time is it in Europe/Lisbon?", "reset": true }

// Response
{ "response": "The current time is 14:32:10", "elapsed_seconds": 12.4 }
```

### `POST /alfred`
```json
// Request
{ "message": "Tell me about Ada Lovelace", "reset": true }
```

### `POST /keyboard`

Gera um arquivo `.tec` para o sistema Eugênio AAC. Retorna o arquivo diretamente para download.

```json
// Request -sem referência
{ "description": "Quero um teclado com as vogais A E I O U e um botão de apagar" }

// Request -com teclado de referência
{
  "description": "Cria um teclado igual mas só com as vogais",
  "reference_keyboard": "LINHA vogais\nGRUPO vogais\n..."
}
```

> O parâmetro `reference_keyboard` é opcional. Quando presente, o modelo usa esse teclado como base de formato e estilo.

### `GET /health`
```json
{ "status": "ok" }
```

> O parâmetro `reset: false` mantém o contexto da conversa anterior (memória).

---

## Como funciona o agente

O Eugênio usa um **CodeAgent** do smolagents -raciocina escrevendo e executando código Python internamente para orquestrar suas ferramentas.

Quando recebe uma mensagem:
1. Envia o prompt ao modelo **Qwen2.5-Coder-32B** no HuggingFace
2. O modelo decide se precisa usar alguma ferramenta
3. Se sim, executa a ferramenta e incorpora o resultado
4. Repete até ter informação suficiente (máx. 4 passos no Simple)
5. Retorna a resposta final

O **Alfred** usa adicionalmente **RAG com BM25** -pesquisa em uma base de dados de documentos antes de responder, passando os resultados mais relevantes ao modelo como contexto.

---

## Limitações conhecidas

- **Latência**: respostas podem demorar entre 10 e 120 segundos
- **Internet obrigatória**: o modelo roda nos servidores do HuggingFace
- **Sem persistência**: o histórico é perdido ao reiniciar o servidor
- **Token gratuito**: possui limites de utilização no HuggingFace

---

## Stack técnica

| Componente | Tecnologia |
|---|---|
| Agente IA | [smolagents](https://github.com/huggingface/smolagents) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Servidor | [Uvicorn](https://www.uvicorn.org) |
| Modelo LLM | Qwen2.5-Coder-32B-Instruct |
| Inferência | [HuggingFace Inference API](https://huggingface.co/inference-api) |
| RAG | LangChain + BM25 |
| Frontend | HTML + CSS + JavaScript puro |

---

## Contexto acadêmico

Este protótipo foi desenvolvido no âmbito de uma dissertação de mestrado que propõe o estudo e desenvolvimento de um chatbot como ferramenta de apoio à criação de teclados para o sistema de Comunicação Aumentativa e Alternativa **Eugênio**.

O objetivo é permitir a participação ativa do usuário no processo de criação do teclado através de diálogo em linguagem natural, contribuindo para a usabilidade e autonomia no cenário da CAA, explorando a união entre CAA, Inteligência Artificial e Interfaces Conversacionais.

---

## Versão Offline — `chat_api_llama`

Para além da versão principal, o projeto inclui uma versão alternativa que funciona **completamente sem internet**, sem necessidade de conta no HuggingFace nem de token de acesso.

Esta versão foi desenvolvida especificamente para o modo **Teclado AAC**, substituindo o modelo remoto Qwen2.5 por um modelo Llama a correr localmente através do **Ollama**.

---

### Arquitetura

```
chat.html  ->  FastAPI (main.py)  ->  smolagents  ->  Llama (Ollama local)
 Frontend        Backend API          Agente IA       Modelo LLM local
```

### Arquivos principais

```
chat_api_llama/
├── main.py              # Servidor FastAPI — expõe apenas /keyboard
├── keyboard_agent.py    # Gerador de teclados .tec com IA local
└── chat.html            # Interface do utilizador (só modo Teclado)
```

---

### Diferenças em relação à versão principal

| | `chat_api` (online) | `chat_api_llama` (offline) |
|---|---|---|
| Modelo LLM | Qwen2.5-Coder-32B (HuggingFace) | Llama 3.1 8B (local via Ollama) |
| Ligação à internet | Obrigatória | Não necessária |
| Token HuggingFace | Obrigatório | Não necessário |
| Modos disponíveis | Simple, Alfred, Teclado | Apenas Teclado AAC |
| Classe do modelo | `InferenceClientModel` | `LiteLLMModel` |
| Latência | 10–120s (depende da rede) | Depende do hardware local |

A principal mudança técnica é a substituição do `InferenceClientModel` pelo `LiteLLMModel`, que serve de ponte entre o smolagents e o Ollama:

```python
# Versão online
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

# Versão offline
model = LiteLLMModel(
    model_id="ollama/llama3.1:8b",
    api_base="http://localhost:11434",
)
```

---

### Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado

---

### Instalação e execução

**1. Instalar o Ollama**

Acesse [ollama.com](https://ollama.com) e instale para o seu sistema operativo.

**2. Baixar o modelo Llama**

```bash
ollama pull llama3.1:8b
```

**3. Instalar dependências Python**

```bash
pip install fastapi uvicorn smolagents pytz litellm
```

**4. Iniciar o servidor**

```bash
cd chat_api_llama
uvicorn main:app --reload --port 8000
```

**5. Abrir o chat**

Abra o ficheiro `chat.html` da pasta `chat_api_llama` diretamente no navegador.

---

### Como funciona a geração de teclados offline

O fluxo é idêntico ao da versão principal, com a diferença de que a chamada ao modelo é feita localmente:

```
Descrição em linguagem natural
        |
FastAPI recebe em POST /keyboard
        |
LiteLLMModel envia o prompt ao Ollama (localhost:11434)
        |
Llama 3.1 lê o exemplo do formato .tec e gera um novo
        |
Ficheiro .tec devolvido para download
        |
Importar no sistema Eugénio AAC
```

O modelo recebe o mesmo prompt de sistema da versão online — incluindo as regras do formato `.tec` e um exemplo real de teclado válido — e aplica **few-shot learning** para gerar o ficheiro correto, sem necessidade de treino específico para o sistema Eugénio.

---

### Requisitos de hardware (aproximado)

| Modelo | RAM necessária | Observação |
|---|---|---|
| `llama3.2:3b` | 4 GB | Rápido, qualidade básica |
| `llama3.1:8b` | 8 GB | Recomendado — boa qualidade e equilíbrio ✅ |
| `llama3.1:70b` | 40 GB | Qualidade próxima ao Qwen2.5-32B |

Para trocar o modelo, basta alterar a constante `OLLAMA_MODEL` no topo do ficheiro `keyboard_agent.py`:

```python
OLLAMA_MODEL = "ollama/llama3.1:8b"  # altere aqui
```