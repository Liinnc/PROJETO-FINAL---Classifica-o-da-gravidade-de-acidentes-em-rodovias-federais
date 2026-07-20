"""Núcleo de dados e avaliação do projeto da PRF.

O notebook mantém a narrativa e os experimentos. Este módulo concentra as
regras que precisam permanecer consistentes e testáveis: fonte dos arquivos,
definição do alvo, atributos permitidos e métricas.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 42
ANOS = list(range(2017, 2026))
ANOS_TREINO = list(range(2017, 2024))
ANO_VALIDACAO = 2024
ANO_TESTE = 2025

IDS_GOOGLE_DRIVE = {
    2017: "1HPLWt5f_l4RIX3tKjI4tUXyZOev52W0N",
    2018: "1cM4IgGMIiR-u4gBIH5IEe3DcvBvUzedi",
    2019: "1pN3fn2wY34GH6cY-gKfbxRJJBFE0lb_l",
    2020: "1esu6IiH5TVTxFoedv6DBGDd01Gvi8785",
    2021: "12xH8LX9aN2gObR766YN3cMcuycwyCJDz",
    2022: "1PRQjuV5gOn_nn6UNvaJyVURDIfbSAK4-",
    2023: "1-WO3SfNrwwZ5_l7fRTiwBKRw7mi1-HUq",
    2024: "14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO",
    2025: "1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR",
}

COLUNAS_LEITURA = [
    "data_inversa",
    "dia_semana",
    "horario",
    "uf",
    "br",
    "km",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "mortos",
    "feridos_graves",
    "veiculos",
    "latitude",
    "longitude",
]

COLUNAS_CATEGORICAS = [
    "dia_semana",
    "uf",
    "br",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]

COLUNAS_NUMERICAS = [
    "km",
    "veiculos",
    "latitude",
    "longitude",
    "hora_seno",
    "hora_cosseno",
    "mes_seno",
    "mes_cosseno",
]

ATRIBUTOS = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS

# Essas colunas jamais podem aparecer em X. A lista é mais ampla que a leitura
# atual para proteger futuras alterações no notebook.
COLUNAS_VAZAMENTO = {
    "id",
    "classificacao_acidente",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
}


def ler_ano(pasta_raw: Path, ano: int) -> pd.DataFrame:
    """Lê um ZIP anual oficial, selecionando apenas as colunas necessárias."""
    caminho = Path(pasta_raw) / f"datatran{ano}.zip"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    with zipfile.ZipFile(caminho) as arquivo_zip:
        nomes = [nome for nome in arquivo_zip.namelist() if nome.lower().endswith(".csv")]
        if len(nomes) != 1:
            raise ValueError(f"Conteúdo inesperado no ZIP de {ano}: {nomes}")
        dados = pd.read_csv(
            arquivo_zip.open(nomes[0]),
            sep=";",
            encoding="latin1",
            usecols=COLUNAS_LEITURA,
            low_memory=False,
        )
    dados["ano"] = ano
    return dados


def carregar_dados(pasta_raw: Path, anos: list[int] | None = None) -> pd.DataFrame:
    """Une os anos solicitados e garante que a origem temporal seja preservada."""
    anos = ANOS if anos is None else anos
    return pd.concat([ler_ano(pasta_raw, ano) for ano in anos], ignore_index=True)


def criar_alvo(dados: pd.DataFrame) -> pd.Series:
    """Retorna 1 quando há morte ou ferimento grave, sem imputar o desfecho."""
    colunas = ["mortos", "feridos_graves"]
    faltantes = [coluna for coluna in colunas if coluna not in dados]
    if faltantes:
        raise ValueError(f"Colunas necessárias para o alvo ausentes: {faltantes}")
    if dados[colunas].isna().any().any():
        raise ValueError("O alvo possui valores ausentes; não é seguro imputá-los como zero.")
    return ((dados["mortos"] > 0) | (dados["feridos_graves"] > 0)).astype("int8")


def converter_numero(serie: pd.Series) -> pd.Series:
    """Converte números brasileiros com vírgula decimal, preservando ausentes."""
    return pd.to_numeric(
        serie.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def preparar_atributos(dados: pd.DataFrame) -> pd.DataFrame:
    """Constrói somente atributos disponíveis na comunicação inicial."""
    base = pd.DataFrame(index=dados.index)
    data = pd.to_datetime(dados["data_inversa"], errors="coerce")
    horario = pd.to_timedelta(dados["horario"].astype("string"), errors="coerce")
    hora_decimal = horario.dt.total_seconds() / 3600

    for coluna in COLUNAS_CATEGORICAS:
        base[coluna] = (
            dados[coluna].astype("string").str.strip().fillna("Não informado")
        )
    base["br"] = "BR-" + base["br"]

    base["km"] = converter_numero(dados["km"])
    base["veiculos"] = pd.to_numeric(dados["veiculos"], errors="coerce")
    base["latitude"] = converter_numero(dados["latitude"])
    base["longitude"] = converter_numero(dados["longitude"])
    base["hora_seno"] = np.sin(2 * np.pi * hora_decimal / 24)
    base["hora_cosseno"] = np.cos(2 * np.pi * hora_decimal / 24)
    base["mes_seno"] = np.sin(2 * np.pi * data.dt.month / 12)
    base["mes_cosseno"] = np.cos(2 * np.pi * data.dt.month / 12)

    if set(base.columns) != set(ATRIBUTOS):
        raise AssertionError("A lista de atributos e a preparação ficaram inconsistentes.")
    if set(base.columns) & COLUNAS_VAZAMENTO:
        raise AssertionError("Uma coluna de desfecho foi incluída nos atributos.")
    return base[ATRIBUTOS]


def metricas_probabilidade(y_real, probabilidade) -> dict[str, float]:
    """Métricas independentes de limiar e erro de calibração."""
    return {
        "PR_AUC": average_precision_score(y_real, probabilidade),
        "ROC_AUC": roc_auc_score(y_real, probabilidade),
        "Brier": brier_score_loss(y_real, probabilidade),
    }


def selecionar_limiar_f1(y_real, probabilidade) -> tuple[float, dict[str, float]]:
    """Seleciona no conjunto de validação o limiar que maximiza o F1."""
    precisoes, recalls, limiares = precision_recall_curve(y_real, probabilidade)
    f1_valores = (2 * precisoes * recalls) / (precisoes + recalls + 1e-12)
    indice = int(np.nanargmax(f1_valores[:-1]))
    return float(limiares[indice]), {
        "Precisao": float(precisoes[indice]),
        "Recall": float(recalls[indice]),
        "F1": float(f1_valores[indice]),
    }


def metricas_classificacao(y_real, probabilidade, limiar: float) -> dict[str, float]:
    """Combina métricas probabilísticas e métricas no limiar informado."""
    previsto = (np.asarray(probabilidade) >= limiar).astype(int)
    return {
        **metricas_probabilidade(y_real, probabilidade),
        "Limiar": limiar,
        "Acuracia": accuracy_score(y_real, previsto),
        "Balanced_accuracy": balanced_accuracy_score(y_real, previsto),
        "Precisao": precision_score(y_real, previsto, zero_division=0),
        "Recall": recall_score(y_real, previsto, zero_division=0),
        "F1": f1_score(y_real, previsto, zero_division=0),
        "F2": fbeta_score(y_real, previsto, beta=2, zero_division=0),
    }


def avaliar_priorizacao(y_real, probabilidade, fracao: float = 0.20) -> dict[str, float]:
    """Avalia uma fila ordenada por risco sob capacidade operacional fixa."""
    if not 0 < fracao <= 1:
        raise ValueError("A fração priorizada deve estar no intervalo (0, 1].")
    y_array = np.asarray(y_real)
    ordem = np.argsort(-np.asarray(probabilidade))
    quantidade = int(np.ceil(fracao * len(y_array)))
    y_prioritario = y_array[ordem[:quantidade]]
    recall = y_prioritario.sum() / y_array.sum()
    precisao = y_prioritario.mean()
    prevalencia = y_array.mean()
    return {
        "Fracao": fracao,
        "Recall": float(recall),
        "Precisao": float(precisao),
        "Lift": float(precisao / prevalencia),
    }


def intervalo_bootstrap(
    y_real,
    probabilidade,
    metrica,
    repeticoes: int = 500,
    seed: int = SEED,
) -> tuple[float, float]:
    """Intervalo percentil de 95% por reamostragem das ocorrências."""
    y_array = np.asarray(y_real)
    p_array = np.asarray(probabilidade)
    rng = np.random.default_rng(seed)
    valores = np.empty(repeticoes)
    for indice in range(repeticoes):
        amostra = rng.integers(0, len(y_array), len(y_array))
        valores[indice] = metrica(y_array[amostra], p_array[amostra])
    inferior, superior = np.quantile(valores, [0.025, 0.975])
    return float(inferior), float(superior)

