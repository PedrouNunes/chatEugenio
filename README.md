# Eugénio — AI Chat

Chatbot baseado em agentes de IA, desenvolvido como protótipo para apoio à criação de teclados para o sistema de **Comunicação Aumentativa e Alternativa (CAA)**.

Construído com **smolagents**, **FastAPI** e **HuggingFace**, o Eugénio oferece dois modos de interação: um agente simples para conversas gerais e o **Alfred**, um agente avançado com RAG e múltiplas ferramentas.

---

## Arquitectura

```
chat.html  →  FastAPI (main.py)  →  smolagents  →  Qwen2.5 (HuggingFace)
 Frontend       Backend API          Agente IA        Modelo LLM remoto
```

### Ficheiros principais

```
chat_api/
├── main.py        # Servidor FastAPI — expõe /chat e /alfred
├── agent.py       # Agente simples (DuckDuckGo + fuso horário)
├── alfred.py      # Agente Alfred (RAG + clima + notícias + HuggingFace Hub)
└── chat.html      # Interface do utilizador
```

---

## Pré-requisitos

- Python 3.10+
- Conta na [HuggingFace](https://huggingface.co) com token de acesso
---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/o-teu-user/o-teu-repo.git
cd o-teu-repo/chat_api
```

### 2. Instalar dependências

```bash
pip install fastapi uvicorn smolagents pytz
pip install langchain langchain-community datasets rank_bm25
```

### 3. Obter o token HuggingFace

1. Vai a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Clica em **New token**
3. Escolhe o tipo **Read**
4. Copia o token gerado — começa com `hf_...`

---

## Como rodar

### Passo 1 — Definir o token no terminal

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_o_teu_token_aqui"
```

> Este passo é necessário sempre que abres um terminal novo.

### Passo 2 — Iniciar o servidor

```bash
cd chat_api
uvicorn main:app --reload --port 8000
```

O terminal deve mostrar:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

### Passo 3 — Abrir o chat

Abre o ficheiro `chat.html` directamente no browser (duplo clique).

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
Agente avançado com planeamento e múltiplas ferramentas. Tem acesso a:
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

### Memória de conversa
Activa o toggle **Memória** na interface para que o agente se lembre do contexto entre mensagens.

---

## API

O servidor expõe dois endpoints REST. Podes testar todos em: **http://localhost:8000/docs**

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

### `GET /health`
```json
{ "status": "ok" }
```

> O parâmetro `reset: false` mantém o contexto da conversa anterior (memória).

---

## Como funciona o agente

O Eugénio usa um **CodeAgent** do smolagents — raciocina escrevendo e executando código Python internamente para orquestrar as suas ferramentas.

Quando recebe uma mensagem:
1. Envia o prompt ao modelo **Qwen2.5-Coder-32B** na HuggingFace
2. O modelo decide se precisa de usar alguma ferramenta
3. Se sim, executa a ferramenta e incorpora o resultado
4. Repete até ter informação suficiente (máx. 4 passos no Simple)
5. Devolve a resposta final

O **Alfred** usa adicionalmente **RAG com BM25** — pesquisa numa base de dados de documentos antes de responder, passando os resultados mais relevantes ao modelo como contexto.

---

## Limitações conhecidas

- **Latência**: respostas podem demorar entre 10 e 120 segundos
- **Internet obrigatória**: o modelo corre nos servidores da HuggingFace
- **Sem persistência**: o histórico perde-se ao reiniciar o servidor
- **Token gratuito**: tem limites de utilização na HuggingFace

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

## Contexto académico

Este protótipo foi desenvolvido no âmbito de uma dissertação de mestrado que propõe o estudo e desenvolvimento de um chatbot como ferramenta de apoio à criação de teclados para o sistema de Comunicação Aumentativa e Alternativa **Eugénio**.

O objectivo é permitir a participação activa do utilizador no processo de criação do teclado através de diálogo em linguagem natural, contribuindo para a usabilidade e autonomia no cenário da CAA, explorando a união entre CAA, Inteligência Artificial e Interfaces Conversacionais.
