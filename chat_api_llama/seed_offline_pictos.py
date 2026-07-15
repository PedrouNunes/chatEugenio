"""
Popula o cache offline de pictogramas (offline_pictos/) com as ~150 palavras
já mapeadas em PICTO_CATEGORIES (main.py), para que o teclado continue a
conseguir usar pictogramas mesmo sem internet.

Corre isto UMA VEZ, com internet ligada:

    python seed_offline_pictos.py

Pode rodar de novo mais tarde sem problema - palavras já guardadas são
ignoradas, só preenche o que falta (por exemplo se caiu a meio por falta de
rede). Se quiseres forçar tudo de novo, apaga offline_pictos/manifest.json.

O main.py usa este cache automaticamente como *fallback*: tenta sempre o
ARASAAC ao vivo primeiro, e só recorre ao que está aqui se não houver
internet ou o ARASAAC estiver em baixo.
"""

import os
import io
import json
import time
import httpx
from PIL import Image

from main import PICTO_CATEGORIES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OFFLINE_DIR = os.path.join(SCRIPT_DIR, "offline_pictos")
MANIFEST_PATH = os.path.join(OFFLINE_DIR, "manifest.json")

OPTIONS_PER_WORD = 3   # quantas alternativas guardar por palavra (serve o seletor de imagens offline também)
PAUSE_BETWEEN_WORDS = 0.3   # segundos - só para não martelar a API do ARASAAC

# Algumas palavras em PICTO_CATEGORIES estão sem acento (para bater certo com
# a classificação por categoria), mas pesquisar no ARASAAC com o acento certo
# dá resultados mais precisos. Isto só afeta a pesquisa aqui no seed - a
# categorização em main.py continua a funcionar sobre a palavra sem acento.
ACCENT_HINTS = {
    "agua": "água", "cafe": "café", "cha": "chá", "pao": "pão", "maca": "maçã",
    "melao": "melão", "acucar": "açúcar",
    "cao": "cão", "passaro": "pássaro", "leao": "leão",
    "dancar": "dançar",
    "cabeca": "cabeça", "maos": "mãos", "mao": "mão", "pes": "pés", "pe": "pé",
    "bracos": "braços", "braco": "braço",
    "mae": "mãe", "irmao": "irmão", "irma": "irmã", "avo": "avó", "bebe": "bebé",
    "familia": "família",
    "calcas": "calças", "chapeu": "chapéu",
    "sofa": "sofá", "televisao": "televisão",
    "lapis": "lápis", "regua": "régua",
    "aviao": "avião", "taxi": "táxi",
    "amanha": "amanhã", "manha": "manhã",
}


def load_manifest():
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(OFFLINE_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def fetch_ids(client, search_term, limit):
    res = client.get(f"https://api.arasaac.org/v1/pictograms/pt/search/{search_term}", timeout=10.0)
    data = res.json()
    if not data or not isinstance(data, list):
        return []
    return [item["_id"] for item in data[:limit] if item.get("_id") is not None]


def download_png(client, pic_id):
    url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_300.png"
    res = client.get(url, timeout=15.0)
    res.raise_for_status()
    return res.content


def main():
    manifest = load_manifest()

    words_by_category = {}
    for category, words in PICTO_CATEGORIES.items():
        for w in words:
            words_by_category[w] = category

    total = len(words_by_category)
    ok_count = 0
    skip_count = 0
    fail_count = 0

    print(f"A popular o cache offline com {total} palavras...\n")

    with httpx.Client() as client:
        for i, (word, category) in enumerate(words_by_category.items(), start=1):
            if word in manifest:
                skip_count += 1
                print(f"[{i}/{total}] '{word}' já em cache, a ignorar")
                continue

            search_term = ACCENT_HINTS.get(word, word)
            print(f"[{i}/{total}] a procurar '{search_term}' ({category})...")

            try:
                ids = fetch_ids(client, search_term, OPTIONS_PER_WORD)
                if not ids:
                    print("  sem resultados no ARASAAC")
                    fail_count += 1
                    continue

                category_dir = os.path.join(OFFLINE_DIR, category)
                os.makedirs(category_dir, exist_ok=True)

                options = []
                for pic_id in ids:
                    try:
                        png_bytes = download_png(client, pic_id)
                        # valida que é mesmo uma imagem antes de guardar
                        Image.open(io.BytesIO(png_bytes)).verify()
                        filename = f"{word}_{pic_id}.png"
                        with open(os.path.join(category_dir, filename), "wb") as f:
                            f.write(png_bytes)
                        options.append({"id": pic_id, "file": filename})
                    except Exception as e:
                        print(f"  falhou a baixar id {pic_id}: {e}")

                if options:
                    manifest[word] = {"category": category, "options": options}
                    save_manifest(manifest)  # grava a cada palavra - se cair a meio, não perde o que já fez
                    ok_count += 1
                    print(f"  guardado ({len(options)} opção(ões))")
                else:
                    fail_count += 1

            except Exception as e:
                print(f"  erro em '{word}': {e}")
                fail_count += 1

            time.sleep(PAUSE_BETWEEN_WORDS)

    print(f"\nConcluído: {ok_count} guardadas, {skip_count} já estavam em cache, {fail_count} falharam (de {total}).")
    print(f"Cache em: {OFFLINE_DIR}")
    if fail_count:
        print("Podes correr o script de novo mais tarde para tentar preencher as que falharam.")


if __name__ == "__main__":
    main()