# chatEugenio

Chatbot desenvolvido como protótipo de dissertação de mestrado para apoio à criação de teclados para o sistema de **Comunicação Aumentativa e Alternativa (CAA) Eugénio V3**.

Construído com **smolagents**, **FastAPI** e modelos de linguagem locais via Ollama, o sistema permite descrever um teclado em português e receber um ficheiro `.tec` pronto a importar no Eugénio.

---

## Arquitetura

```
chat.html  ->  FastAPI (main.py)  ->  smolagents  ->  Qwen2.5 (HuggingFace)
 Frontend        Backend API          Agente IA        Modelo LLM remoto
```

### Arquivos principais

```
chat_api/
├── main.py              # Servidor FastAPI - expoe /chat, /alfred e /keyboard
├── agent.py             # Agente simples (DuckDuckGo + fuso horario)
├── alfred.py            # Agente Alfred (RAG + clima + noticias + HuggingFace Hub)
├── keyboard_agent.py    # Gerador de teclados .tec com IA
└── chat.html            # Interface do usuario
```

---

## Pre-requisitos

- Python 3.10+
- Conta no [HuggingFace](https://huggingface.co) com token de acesso
- Conexao com a internet (o modelo roda remotamente)

---

## Instalacao

### 1. Clonar o repositorio

```bash
git clone https://github.com/PedrouNunes/chatEugenio.git
cd chatEugenio/chat_api
```

### 2. Instalar dependencias

```bash
pip install fastapi uvicorn smolagents pytz
pip install langchain langchain-community datasets rank_bm25
```

### 3. Obter o token do HuggingFace

1. Acesse [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Clique em **New token**
3. Escolha o tipo **Read**
4. Copie o token gerado - comeca com `hf_...`

---

## Como rodar

### Passo 1 - Definir o token no terminal

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_seu_token_aqui"
```

**Mac/Linux:**
```bash
export HF_TOKEN="hf_seu_token_aqui"
```

> Este passo e necessario sempre que abrir um terminal novo.

### Passo 2 - Iniciar o servidor

```bash
cd chat_api
uvicorn main:app --reload --port 8000
```

O terminal deve exibir:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

### Passo 3 - Abrir o chat

Abra o arquivo `chat.html` diretamente no navegador (duplo clique).

---

## Como usar

### Modo Simple

Agente rapido para perguntas gerais. Tem acesso a:
- Pesquisa web (DuckDuckGo)
- Fuso horario em tempo real

**Exemplos:**
```
What time is it in Tokyo?
What are the latest news about artificial intelligence?
What is machine learning?
```

### Modo Alfred

Agente avancado com planejamento e multiplas ferramentas. Tem acesso a:
- Pesquisa web
- Informacao meteorologica
- Noticias recentes
- Estatisticas do HuggingFace Hub
- Base de dados RAG de convidados

**Exemplos:**
```
Tell me about Lady Ada Lovelace
What's the weather like in Paris tonight?
What is the most popular model from Google on HuggingFace?
Give me the latest news about robotics
```

### Modo Teclado AAC

O modo principal da dissertacao. Permite criar arquivos `.tec` para o sistema Eugenio atraves de linguagem natural.

**Como usar:**
1. Clique em **Teclado** na interface
2. Descreva o teclado que deseja criar em portugues
3. Clique em **Download teclado.tec**
4. Importe o arquivo no sistema Eugenio

**Exemplos de prompts:**
```
Quero um teclado simples com as vogais A E I O U e um botao de apagar
Cria um teclado com numeros de 1 a 10 e um botao de espaco
Quero um teclado com saudacoes: Ola, Bom dia, Boa tarde, Obrigado
Cria um teclado de emocoes: Feliz, Triste, Com fome, Com sono, Com dor
```

**Como funciona internamente:**
```
Descricao em linguagem natural
        |
FastAPI recebe em POST /keyboard
        |
Qwen2.5 le o exemplo do formato .tec e gera um novo
        |
Arquivo .tec devolvido para download
        |
Importar no sistema Eugenio AAC
```

#### Usar um teclado existente como referencia

E possivel carregar um arquivo `.tec` existente como base para o modelo gerar variacoes com o mesmo estilo e estrutura.

1. Clique no botao de upload ao lado do campo de texto
2. Selecione um arquivo `.tec` do seu computador
3. O sistema muda automaticamente para o modo Teclado
4. Descreva o que deseja criar com base nesse teclado

**Exemplos com referencia:**
```
Cria um teclado igual mas so com as vogais
Mantem a estrutura mas adiciona uma linha com pontuacao
Usa o mesmo formato mas com palavras de comunicacao basica
```

> O modelo usa o teclado carregado como exemplo de formato e estilo, garantindo compatibilidade com o sistema Eugenio.

#### Formato do arquivo `.tec`

O Eugenio usa um formato de texto simples:

```
LINHA nome;;;da;;;linha
GRUPO nome;;;do;;;grupo
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL [imagem] [label] [valor] 1 -1 -1
```

- **`;;;`** - separador de espacos nos nomes
- **`TECLA_VAZIA`** - celula vazia obrigatoria no inicio de cada linha
- **`TECLA_NORMAL`** - tecla clicavel com imagem, label e valor
- **`<--->`** - indica tecla de acao (sem imagem propria)
- **`[Nome-Da-Acao]`** - acoes especiais como `[Synthesize-Word-To-Speech]`

## Como a geracao de teclados funciona com IA

O utilizador descreve em portugues o que quer e o sistema entrega um ficheiro pronto a usar no Eugenio AAC. Por baixo ha uma cadeia de componentes que tornam isso possivel.

### 1. O usuario descreve o teclado

Tudo comeca com uma mensagem em linguagem natural, por exemplo:

```
Quero um teclado com as vogais A E I O U e um botao de apagar
```

Nao e necessario saber nada sobre o formato `.tec` ou sobre programacao.

### 2. O frontend envia para o backend

O `chat.html` deteta que o modo ativo e o **Teclado AAC** e envia a descricao para o endpoint correto:

```
POST http://localhost:8000/keyboard
{ "description": "Quero um teclado com as vogais A E I O U e um botao de apagar" }
```

Se o usuario carregou um arquivo `.tec` de referencia, ele e incluido no mesmo pedido como `reference_keyboard`.

### 3. O backend monta o prompt para o modelo

O `keyboard_agent.py` chama o modelo diretamente com um prompt de sistema construido ao longo do desenvolvimento. Esse prompt contem:

- As regras do formato `.tec` (LINHA, GRUPO, TECLA, separadores)
- Um exemplo real de teclado valido
- A descricao que o usuario enviou
- A instrucao de devolver apenas o conteudo do arquivo, sem explicacoes

A forma como o prompt e escrito determina a qualidade e o formato da resposta - cada bloco nasceu de um bug concreto encontrado durante o desenvolvimento.

### 4. O modelo de linguagem gera o arquivo

O prompt e enviado ao modelo **Qwen2.5-Coder-32B-Instruct** hospedado no HuggingFace. Este e um Large Language Model (LLM) especializado em codigo, com 32 bilhoes de parametros.

O modelo le o exemplo fornecido, entende o padrao do formato `.tec` e gera um novo arquivo seguindo as mesmas regras, adaptado a descricao do usuario. Nao foi treinado especificamente para o Eugenio - aprende o formato apenas pelo exemplo dado no prompt, no momento da chamada. Esta tecnica chama-se **few-shot learning**.

### 5. O backend devolve o arquivo para download

O conteudo gerado pelo modelo e devolvido ao frontend como um arquivo para download. O FastAPI define o nome `teclado.tec` e o tipo de conteudo correto para que o navegador faca o download automaticamente.

### 6. O usuario importa no sistema Eugenio

O arquivo `.tec` baixado pode ser importado diretamente no software Eugenio AAC, onde o teclado gerado estara pronto para uso.

### Relacao com o chat

A geracao de teclados e um caso especializado do mesmo padrao usado nos modos Simple e Alfred: o usuario escreve em linguagem natural, o sistema interpreta e age. A diferenca e que em vez de responder com texto, o sistema produz um arquivo estruturado num formato especifico.

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

### Memoria de conversa

Ative o toggle **Memoria** na interface para que o agente se lembre do contexto entre mensagens.

---

## API

O servidor expoe os seguintes endpoints REST. Todos podem ser testados em: **http://localhost:8000/docs**

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

Gera um arquivo `.tec` para o sistema Eugenio AAC. Retorna o arquivo diretamente para download.

```json
// Request - sem referencia
{ "description": "Quero um teclado com as vogais A E I O U e um botao de apagar" }

// Request - com teclado de referencia
{
  "description": "Cria um teclado igual mas so com as vogais",
  "reference_keyboard": "LINHA vogais\nGRUPO vogais\n..."
}
```

> O parametro `reference_keyboard` e opcional. Quando presente, o modelo usa esse teclado como base de formato e estilo.

### `GET /health`
```json
{ "status": "ok" }
```

> O parametro `reset: false` mantem o contexto da conversa anterior (memoria).

---

## Como funciona o agente

O sistema usa um **CodeAgent** do smolagents - raciocina escrevendo e executando codigo Python internamente para orquestrar as suas ferramentas.

Quando recebe uma mensagem:
1. Envia o prompt ao modelo **Qwen2.5-Coder-32B** no HuggingFace
2. O modelo decide se precisa usar alguma ferramenta
3. Se sim, executa a ferramenta e incorpora o resultado
4. Repete ate ter informacao suficiente (max. 4 passos no Simple)
5. Retorna a resposta final

O **Alfred** usa adicionalmente **RAG com BM25** - pesquisa numa base de dados de documentos antes de responder, passando os resultados mais relevantes ao modelo como contexto.

---

## Limitacoes conhecidas

- **Latencia**: respostas podem demorar entre 10 e 120 segundos
- **Internet obrigatoria**: o modelo roda nos servidores do HuggingFace
- **Sem persistencia**: o historico e perdido ao reiniciar o servidor
- **Token gratuito**: possui limites de utilizacao no HuggingFace

---

## Stack tecnica

| Componente | Tecnologia |
|---|---|
| Agente IA | [smolagents](https://github.com/huggingface/smolagents) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Servidor | [Uvicorn](https://www.uvicorn.org) |
| Modelo LLM | Qwen2.5-Coder-32B-Instruct |
| Inferencia | [HuggingFace Inference API](https://huggingface.co/inference-api) |
| RAG | LangChain + BM25 |
| Frontend | HTML + CSS + JavaScript puro |

---

## Contexto academico

Este prototipo foi desenvolvido no ambito de uma dissertacao de mestrado em Engenharia Informatica e IoT no ESTIG/IPBeja, que propoe o estudo e desenvolvimento de um chatbot como ferramenta de apoio a criacao de teclados para o sistema de Comunicacao Aumentativa e Alternativa Eugenio V3.

O objetivo e permitir que terapeutas e educadores criem teclados personalizados atraves de dialogo em linguagem natural, sem necessidade de conhecimentos tecnicos sobre o formato do sistema.

---

## Versao Offline - `chat_api_llama`

Para alem da versao principal, o projeto inclui uma versao que funciona **completamente sem internet**, sem necessidade de conta no HuggingFace nem de token de acesso.

Esta versao foi desenvolvida especificamente para o modo **Teclado AAC**, substituindo o modelo remoto por **qwen2.5-coder:3b** a correr localmente atraves do **Ollama**.

---

### Arquitetura

```
chat.html  ->  FastAPI (main.py)  ->  smolagents/LiteLLM  ->  Ollama (qwen2.5-coder:3b)
 Frontend        Backend API                                    Modelo local
```

A geracao usa duas camadas em vez de confiar cegamente no modelo:

- **Camada probabilistica (LLM):** o modelo gera uma proposta de `.tec` a partir do `SYSTEM_PROMPT` e da descricao do utilizador
- **Camada deterministica (`buildKeyboard()`):** para conjuntos fechados (digitos 0-9, alfabeto A-Z, acentos PT-PT, simbolos especiais), ignora o output do modelo e usa listas canonicas fixas. Para palavras tematicas livres, usa o output do modelo com filtragem de ruido

Isto existe porque um modelo de 3B parametros e inconsistente em conjuntos fechados: duplica teclas, omite letras, gera acentos de outras linguas. A camada deterministica resolve isso sem fine-tuning.

### Arquivos principais

```
chat_api_llama/
├── main.py              # Servidor FastAPI - 6 endpoints
├── keyboard_agent.py    # SYSTEM_PROMPT + chamada ao Ollama via smolagents/LiteLLM
└── chat.html            # Interface completa num unico ficheiro (HTML + CSS + JS)
```

---

### Diferencas em relacao a versao principal

| | `chat_api` (online) | `chat_api_llama` (offline) |
|---|---|---|
| Modelo LLM | Qwen2.5-Coder-32B (HuggingFace) | qwen2.5-coder:3b (local via Ollama) |
| Ligacao a internet | Obrigatoria | Nao necessaria |
| Token HuggingFace | Obrigatorio | Nao necessario |
| Modos disponiveis | Simple, Alfred, Teclado | Apenas Teclado AAC |
| Classe do modelo | `InferenceClientModel` | `LiteLLMModel` |
| Latencia | 10-120s (depende da rede) | Depende do hardware local |

A principal mudanca tecnica e a substituicao do `InferenceClientModel` pelo `LiteLLMModel`, que serve de ponte entre o smolagents e o Ollama:

```python
# Versao online
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

# Versao offline
model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:3b",
    api_base="http://localhost:11434",
)
```

---

### Pre-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e a correr

---

### Instalacao e execucao

**1. Instalar o Ollama**

Acesse [ollama.com](https://ollama.com) e instale para o seu sistema operativo.

**2. Baixar o modelo**

```bash
ollama pull qwen2.5-coder:3b
```

**3. Instalar dependencias Python**

```bash
pip install fastapi uvicorn smolagents pytz litellm Pillow
```

> `Pillow` e necessario para teclados com pictogramas (conversao PNG para BMP). Sem ele o resto funciona normalmente.

**4. Iniciar o servidor**

```bash
cd chat_api_llama
uvicorn main:app --reload --port 8000
```

**5. Abrir o chat**

Abra o ficheiro `chat.html` da pasta `chat_api_llama` diretamente no navegador (duplo clique).

---

### Como funciona a geracao de teclados offline

#### 1. O utilizador descreve o teclado

```
quero um teclado com os numeros de 0 a 9 e um botao de apagar
```

#### 2. O frontend envia para o backend

O `chat.html` envia a descricao para `/keyboard` com o historico da conversa (se a memoria estiver ligada) e o estado atual do preview (se ja existe um teclado em curso).

#### 3. O backend monta o prompt para o modelo

O `keyboard_agent.py` constroi a lista de mensagens `[system, ...historico, user_atual]` e chama o modelo via Ollama. O `SYSTEM_PROMPT` (~160 linhas) ensina o formato `.tec`, as regras de acentuacao PT-PT, a distincao entre teclas normais e botoes de acao, e como representar caracteres especiais e pictogramas.

#### 4. O modelo gera o ficheiro

O `qwen2.5-coder:3b` le os exemplos e gera um novo `.tec`. Nao foi treinado especificamente para o Eugenio - aprende o formato pelo exemplo dado no prompt (few-shot learning). `temperature=0.0` para maximizar o determinismo.

#### 5. O frontend filtra e reconstroi

O `.tec` gerado passa por `parseTec()` e `buildKeyboard()`. Para conjuntos fechados (digitos, alfabeto, acentos), o frontend ignora o output do modelo e usa listas canonicas. Para palavras tematicas, usa o que o modelo gerou com filtragem de ruido.

#### 6. O utilizador reve e aprova

O preview aparece no chat com as teclas geradas. E possivel editar, remover ou adicionar teclas diretamente no preview, refinar com novos pedidos em linguagem natural, e aprovar para gravar o `.tec` diretamente na pasta de teclados do Eugenio.

---

### Funcionalidades da interface

| Funcionalidade | Descricao |
|---|---|
| Sidebar de conversas | Historico estilo ChatGPT com memoria independente por conversa |
| Refinamento por ronda | "adicione um botao de apagar" preserva todo o teclado anterior |
| Remocao via texto | "remova o numero 0", "remova os acentos", "sem o simbolo @" |
| Edicao inline | Duplo-clique numa tecla no preview para editar |
| Adicao inline | Botao `+` no fim de cada linha do preview |
| Pictogramas ARASAAC | "quero um teclado de pictogramas com comer, beber, dormir" |
| Memoria por conversa | Cada conversa guarda o seu proprio estado ao sair e restaura ao voltar |

---

### Endpoints da API

| Metodo | Rota | Funcao |
|---|---|---|
| `POST` | `/keyboard` | Gera `.tec` a partir da descricao e historico |
| `POST` | `/save_keyboard` | Grava `.tec` na pasta do Eugenio |
| `GET` | `/pictogram?q=` | Proxy ARASAAC - devolve ID e URL do pictograma |
| `POST` | `/save_pictogram` | Descarrega PNG do ARASAAC, converte para BMP, guarda em `CAT_IMG_pic\` |
| `GET` | `/list_keyboards` | Lista `.tec` existentes na pasta do Eugenio |
| `GET` | `/health` | Estado do servidor |

---

### Requisitos de hardware (aproximado)

| Modelo | RAM necessaria | Observacao |
|---|---|---|
| `qwen2.5-coder:3b` | 4 GB | Modelo atual - bom equilibrio velocidade/qualidade |
| `qwen2.5-coder:7b` | 8 GB | Mais qualidade, mais lento |
| `llama3.1:8b` | 8 GB | Alternativa generalista |

Para trocar o modelo, alterar a constante `OLLAMA_MODEL` no topo do ficheiro `keyboard_agent.py`:

```python
OLLAMA_MODEL = "ollama/qwen2.5-coder:3b"  # altere aqui
```