import os
import re
import time
import io
import json
import unicodedata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from typing import List
from keyboard_agent import generate_keyboard_with_history
import httpx
try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_HTTP = httpx.Client(timeout=8.0)

# Caminho da pasta de teclados do Eugénio — mudar se necessário
EUGENIO_FOLDER  = r"C:\Users\Pedro Nunes\AppData\Roaming\LabSI2-INESC-ID\Eugénio 3.0"
PICTO_FOLDER    = os.path.join(EUGENIO_FOLDER, "CAT_IMG_pic")

# Cache offline de pictogramas: pasta dentro do próprio projeto (não da pasta
# do Eugénio) com PNGs pré-descarregados para as palavras já mapeadas em
# PICTO_CATEGORIES. Populada por seed_offline_pictos.py, correndo uma vez com
# internet - ver esse ficheiro para mais detalhe. O manifest.json é o índice:
# palavra (já sem acentos, minúscula) -> categoria + lista de {id, file}.
OFFLINE_PICTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_pictos")
OFFLINE_MANIFEST_PATH = os.path.join(OFFLINE_PICTO_DIR, "manifest.json")


def load_offline_manifest():
    try:
        with open(OFFLINE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# Carregado uma vez no arranque - se quiseres recarregar depois de correr o
# seed script de novo sem reiniciar o servidor, usa o endpoint /health que
# não depende disto, ou reinicia o uvicorn (--reload já trata disso sozinho).
_OFFLINE_MANIFEST = load_offline_manifest()

# Classificação de pictogramas por categoria — sem depender de nenhuma API
# externa para isso. Chave: nome da categoria (também o nome da subpasta
# dentro de CAT_IMG_pic\). Valor: palavras associadas, já em minúsculas e
# sem acentos (mesma normalização usada no nome do ficheiro, para a
# comparação em classify_pictogram ser direta). Palavra que não aparece em
# nenhuma destas listas cai em "geral". Para adicionar uma categoria nova,
# basta acrescentar outra entrada aqui — não precisa de mexer em mais nada.
PICTO_CATEGORIES = {
    "alimentos": {
        "agua", "sumo", "leite", "cafe", "cha", "sopa", "pao", "arroz", "massa",
        "carne", "frango", "fruta", "maca", "banana", "laranja", "pera", "uva",
        "manga", "kiwi", "melao", "morango", "queijo", "manteiga", "ovo", "bolo",
        "chocolate", "biscoito", "iogurte", "sal", "acucar", "azeite", "legumes",
        "salada", "batata", "cenoura", "tomate",
    },
    "animais": {
        "cao", "gato", "passaro", "peixe", "cavalo", "vaca", "porco", "galinha",
        "pato", "coelho", "rato", "leao", "tigre", "elefante", "urso", "macaco",
        "borboleta", "abelha", "aranha", "cobra", "tartaruga", "sapo", "ovelha",
        "cabra", "burro", "girafa", "zebra",
    },
    "acoes": {
        "comer", "beber", "dormir", "brincar", "correr", "andar", "sentar",
        "levantar", "chorar", "rir", "ajudar", "ler", "escrever", "pintar",
        "cantar", "dancar", "saltar", "nadar", "jogar", "estudar", "trabalhar",
        "lavar", "vestir", "abrir", "fechar", "olhar", "ouvir", "falar",
    },
    "emocoes": {
        "feliz", "triste", "zangado", "cansado", "assustado", "calmo",
        "surpreso", "nervoso", "contente", "aborrecido", "envergonhado",
        "orgulhoso", "preocupado",
    },
    "corpo": {
        "cabeca", "olhos", "olho", "nariz", "boca", "orelhas", "orelha", "maos",
        "mao", "pes", "pe", "bracos", "braco", "pernas", "perna", "barriga",
        "dentes", "dente", "cabelo", "costas", "dedos", "dedo",
    },
    "familia": {
        "mae", "pai", "irmao", "irma", "avo", "tio", "tia", "primo", "prima",
        "bebe", "filho", "filha", "familia",
    },
    "roupa": {
        "camisola", "calcas", "sapatos", "sapato", "casaco", "chapeu", "meias",
        "meia", "vestido", "saia", "camisa", "cachecol", "luvas",
    },
    "casa": {
        "casa", "quarto", "cozinha", "sala", "cama", "mesa", "cadeira", "porta",
        "janela", "banheira", "chuveiro", "sofa", "televisao",
    },
    "escola": {
        "escola", "professor", "professora", "livro", "caderno", "lapis",
        "borracha", "mochila", "colega", "regua", "tesoura", "cola",
    },
    "transportes": {
        "carro", "autocarro", "comboio", "aviao", "bicicleta", "mota", "barco",
        "metro", "taxi",
    },
    "tempo": {
        "sol", "chuva", "vento", "nuvem", "neve", "frio", "calor", "hoje",
        "amanha", "ontem", "manha", "tarde", "noite",
    },
    "brinquedos": {
        "bola", "boneca", "carrinho", "puzzle", "jogo", "brinquedo", "peluche",
    },
}

app = FastAPI(title="Eugénio — Teclado AAC (Offline)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Elapsed-Seconds"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class KeyboardRequest(BaseModel):
    description: str
    reference_keyboard: str = ""
    history: List[HistoryMessage] = []


class SaveRequest(BaseModel):
    content: str
    name: str


class PictogramSaveRequest(BaseModel):
    word: str
    pic_id: int
    force: bool = False


def fix_encoding(text):
    # O modelo às vezes gera \xe7 ou \xc3\xa7 como texto em vez do caracter real.
    # Aqui converte essas sequências de volta para os caracteres correctos.
    def fix_utf8_seq(m):
        hex_vals = re.findall(r'[0-9a-fA-F]{2}', m.group(0))
        try:
            return bytes.fromhex(''.join(hex_vals)).decode('utf-8')
        except Exception:
            return m.group(0)

    def fix_single(m):
        try:
            return bytes.fromhex(m.group(1)).decode('latin1')
        except Exception:
            return m.group(0)

    text = re.sub(r'(?:\\x[0-9a-fA-F]{2}){2,}', fix_utf8_seq, text)
    text = re.sub(r'\\x([0-9a-fA-F]{2})', fix_single, text)
    text = text.replace('\uFFFD', '')
    text = unicodedata.normalize('NFC', text)
    return text


def to_latin1(text):
    try:
        return text.encode("latin1")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode("latin1", errors="replace")


def ascii_fold(text):
    # Minúsculas e sem acentos - usada tanto para o nome do ficheiro como
    # para a classificação por categoria, para as duas ficarem consistentes.
    norm = unicodedata.normalize('NFKD', text)
    return norm.encode('ascii', 'ignore').decode().lower().strip()


def classify_pictogram(word):
    """Classifica a palavra numa categoria fixa (nome da subpasta em
    CAT_IMG_pic\\), sem depender de nenhuma API externa. Palavra que não
    bate com nenhuma categoria conhecida cai em 'geral'."""
    ascii_word = ascii_fold(word)
    for category, words in PICTO_CATEGORIES.items():
        if ascii_word in words:
            return category
    return "geral"


def offline_lookup(word):
    """Procura a palavra no cache offline (manifest.json). Devolve
    {"category": ..., "options": [{"id":, "file":}, ...]} ou None se a
    palavra não foi pré-descarregada pelo seed_offline_pictos.py."""
    return _OFFLINE_MANIFEST.get(ascii_fold(word))


@app.post("/keyboard")
def create_keyboard(req: KeyboardRequest):
    start = time.time()

    history = [{"role": m.role, "content": m.content} for m in req.history]

    if req.reference_keyboard and not history:
        history = [
            {"role": "user",      "content": "Usa este teclado como base."},
            {"role": "assistant", "content": req.reference_keyboard},
        ]
    elif req.reference_keyboard:
        history.append({"role": "assistant", "content": req.reference_keyboard})

    tec_content = generate_keyboard_with_history(req.description, history)
    tec_content = fix_encoding(tec_content)
    elapsed = round(time.time() - start, 2)

    return Response(
        content=to_latin1(tec_content),
        media_type="text/plain; charset=latin1",
        headers={
            "Content-Disposition": "attachment; filename=teclado.tec",
            "X-Elapsed-Seconds": str(elapsed),
        }
    )


@app.post("/save_keyboard")
def save_keyboard(req: SaveRequest):
    try:
        if not os.path.isdir(EUGENIO_FOLDER):
            return {"ok": False, "error": f"Pasta não encontrada: {EUGENIO_FOLDER}"}

        safe_name = "".join(c for c in req.name if c not in r'\/:*?"<>|').strip() or "teclado"
        filepath = os.path.join(EUGENIO_FOLDER, safe_name + ".tec")
        content = fix_encoding(req.content)

        with open(filepath, "wb") as f:
            f.write(to_latin1(content))

        return {"ok": True, "path": filepath, "name": safe_name + ".tec"}

    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/list_keyboards")
def list_keyboards():
    try:
        if not os.path.isdir(EUGENIO_FOLDER):
            return {"ok": False, "files": []}
        files = [f for f in os.listdir(EUGENIO_FOLDER) if f.endswith(".tec")]
        return {"ok": True, "files": sorted(files)}
    except Exception:
        return {"ok": False, "files": []}


@app.get("/pictogram")
def get_pictogram(q: str):
    # Tenta sempre o ARASAAC primeiro - mais opções e mais atual. O cache
    # offline só entra em jogo se isto falhar (sem internet, ARASAAC fora, etc).
    try:
        res = _HTTP.get(f"https://api.arasaac.org/v1/pictograms/pt/search/{q}", timeout=6.0)
        data = res.json()
        if data and isinstance(data, list) and data:
            pic_id = data[0]["_id"]
            return {"id": pic_id, "url": f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_300.png", "offline": False}
    except Exception:
        pass

    cached = offline_lookup(q)
    if cached and cached["options"]:
        opt = cached["options"][0]
        return {
            "id": opt["id"],
            "url": f"http://localhost:8000/offline_picto_image/{cached['category']}/{opt['file']}",
            "offline": True,
        }
    return {"id": None, "url": None}


@app.get("/pictogram_search")
def search_pictograms(q: str, limit: int = 12):
    """Devolve várias correspondências do ARASAAC para a palavra, não só a
    primeira - usado no seletor de pictogramas alternativos do preview.
    Sem internet (ou com o ARASAAC em baixo), cai no cache offline das
    palavras já pré-descarregadas por seed_offline_pictos.py."""
    try:
        res = _HTTP.get(f"https://api.arasaac.org/v1/pictograms/pt/search/{q}", timeout=6.0)
        data = res.json()
        if data and isinstance(data, list) and data:
            results = []
            for item in data[:limit]:
                pic_id = item.get("_id")
                if pic_id is None:
                    continue
                results.append({
                    "id": pic_id,
                    "url": f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_300.png",
                })
            if results:
                return {"ok": True, "results": results, "offline": False}
    except Exception:
        pass

    cached = offline_lookup(q)
    if cached and cached["options"]:
        results = [
            {"id": o["id"], "url": f"http://localhost:8000/offline_picto_image/{cached['category']}/{o['file']}"}
            for o in cached["options"][:limit]
        ]
        return {"ok": True, "results": results, "offline": True}
    return {"ok": True, "results": []}


@app.get("/offline_picto_image/{category}/{filename}")
def get_offline_picto_image(category: str, filename: str):
    """Serve os PNGs do cache offline (populados por seed_offline_pictos.py).
    Sanitiza os dois parâmetros porque vêm da URL - evita sair da pasta
    offline_pictos por path traversal."""
    safe_category = re.sub(r'[^a-z0-9_]', '', category)
    safe_filename = re.sub(r'[^a-zA-Z0-9_.]', '', filename)
    path = os.path.join(OFFLINE_PICTO_DIR, safe_category, safe_filename)
    if not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/save_pictogram")
def save_pictogram(req: PictogramSaveRequest):
    """Descarrega o pictograma ARASAAC, converte para BMP e guarda em
    CAT_IMG_pic\\<categoria>\\ dentro da pasta do Eugénio.

    Com force=True sobrescreve o ficheiro já existente em vez de reaproveitar
    o cache - usado para trocar a imagem de uma tecla que já tem pictograma,
    sem nunca criar um ficheiro novo nem duplicar.

    Sem internet, tenta o cache offline (seed_offline_pictos.py) para a
    mesma palavra antes de desistir.
    """
    if not _PIL_OK:
        return {"ok": False, "error": "Pillow não instalado — corre: pip install Pillow"}
    try:
        category = classify_pictogram(req.word)
        category_folder = os.path.join(PICTO_FOLDER, category)
        # Cria a pasta da categoria só se ainda não existir - se já existir
        # (por já ter outro pictograma lá dentro), reaproveita-a tal e qual,
        # nunca cria uma pasta nova ou duplicada para a mesma categoria.
        os.makedirs(category_folder, exist_ok=True)

        # Nome de ficheiro seguro a partir da palavra (sem acentos, lowercase)
        ascii_word = ascii_fold(req.word)
        safe = re.sub(r'[^a-z0-9]', '_', ascii_word).strip('_') or f"picto_{req.pic_id}"
        filename = safe + '.bmp'
        filepath = os.path.join(category_folder, filename)
        relpath = f"{category}\\{filename}"

        # Se já existe e não é para forçar, devolve sem re-descarregar
        if os.path.isfile(filepath) and not req.force:
            return {"ok": True, "filename": filename, "category": category, "relpath": relpath, "cached": True}

        png_bytes = None
        used_offline = False

        # Tenta o ARASAAC primeiro
        try:
            url = f"https://static.arasaac.org/pictograms/{req.pic_id}/{req.pic_id}_300.png"
            png_res = _HTTP.get(url, timeout=12.0)
            if png_res.status_code == 200:
                png_bytes = png_res.content
        except Exception:
            pass

        # Sem internet (ou falhou) - tenta o cache offline da mesma palavra.
        # Se o pic_id pedido não foi pré-descarregado, usa a primeira opção
        # em cache mesmo assim (melhor uma imagem plausível do que nenhuma).
        if png_bytes is None:
            cached = offline_lookup(req.word)
            if cached and cached["options"]:
                match = next((o for o in cached["options"] if o["id"] == req.pic_id), cached["options"][0])
                cached_path = os.path.join(OFFLINE_PICTO_DIR, cached["category"], match["file"])
                if os.path.isfile(cached_path):
                    with open(cached_path, "rb") as f:
                        png_bytes = f.read()
                    used_offline = True

        if png_bytes is None:
            return {"ok": False, "error": "Sem internet e sem cache offline para esta palavra"}

        # Converte PNG → BMP (sem canal alpha — BMP não suporta transparência)
        img = _PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
        bmp_buf = io.BytesIO()
        img.save(bmp_buf, format="BMP")

        # Mesmo ficheiro sempre que a palavra é a mesma - com force=True isto
        # sobrescreve o conteúdo anterior em vez de criar um novo
        with open(filepath, "wb") as f:
            f.write(bmp_buf.getvalue())

        return {"ok": True, "filename": filename, "category": category, "relpath": relpath, "cached": False, "offline": used_offline}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "Ollama (offline)",
        "pillow": _PIL_OK,
        "offline_pictos_cached": len(_OFFLINE_MANIFEST),
    }