"""Baixa os CSVs oficiais da PRF, agrupados por ocorrência (2017-2025)."""

from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prf_ml import IDS_GOOGLE_DRIVE  # noqa: E402


def baixar_arquivo(ano: int, destino: Path) -> None:
    """Baixa um ZIP se ele ainda não existir."""
    if destino.exists() and destino.stat().st_size > 100_000:
        print(f"{ano}: arquivo já existe")
        return

    identificador = IDS_GOOGLE_DRIVE[ano]
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={identificador}&export=download&confirm=t"
    )
    print(f"{ano}: baixando...")
    with requests.get(url, stream=True, timeout=120) as resposta:
        resposta.raise_for_status()
        with destino.open("wb") as arquivo:
            for bloco in resposta.iter_content(chunk_size=1024 * 1024):
                if bloco:
                    arquivo.write(bloco)

    if destino.stat().st_size < 100_000:
        raise RuntimeError(f"O arquivo de {ano} parece inválido.")


def main() -> None:
    pasta = ROOT / "dados" / "raw"
    pasta.mkdir(parents=True, exist_ok=True)
    for ano in IDS_GOOGLE_DRIVE:
        baixar_arquivo(ano, pasta / f"datatran{ano}.zip")


if __name__ == "__main__":
    main()
