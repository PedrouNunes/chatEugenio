# Eugênio — AI Chat

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
├── main.py              # Servidor FastAPI — expõe /chat, /alfred e /keyboard
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
4. Copie o token gerado — começa com `hf_...`

---

## Como rodar

### Passo 1 — Definir o token no terminal

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_seu_token_aqui"
```

**Mac/Linux:**
```bash
export HF_TOKEN="hf_seu_token_aqui"
```

> Este passo é necessário sempre que abrir um terminal novo.

### Passo 2 — Iniciar o servidor

```bash
cd chat_api
uvicorn main:app --reload --port 8000
```

O terminal deve exibir:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

### Passo 3 — Abrir o chat

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

- **`;;;`** — separador de espaços nos nomes
- **`TECLA_VAZIA`** — célula vazia obrigatória no início de cada linha
- **`TECLA_NORMAL`** — tecla clicável com imagem, label e valor
- **`<--->`** — indica tecla de ação (sem imagem própria)
- **`[Nome-Da-Acao]`** — ações especiais como `[Synthesize-Word-To-Speech]`

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
// Request — sem referência
{ "description": "Quero um teclado com as vogais A E I O U e um botão de apagar" }

// Request — com teclado de referência
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

O Eugênio usa um **CodeAgent** do smolagents — raciocina escrevendo e executando código Python internamente para orquestrar suas ferramentas.

Quando recebe uma mensagem:
1. Envia o prompt ao modelo **Qwen2.5-Coder-32B** no HuggingFace
2. O modelo decide se precisa usar alguma ferramenta
3. Se sim, executa a ferramenta e incorpora o resultado
4. Repete até ter informação suficiente (máx. 4 passos no Simple)
5. Retorna a resposta final

O **Alfred** usa adicionalmente **RAG com BM25** — pesquisa em uma base de dados de documentos antes de responder, passando os resultados mais relevantes ao modelo como contexto.

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
