import numpy as np
import pandas as pd
import pytest

from src.prf_ml import (
    ATRIBUTOS,
    COLUNAS_VAZAMENTO,
    avaliar_priorizacao,
    converter_numero,
    criar_alvo,
    preparar_atributos,
    selecionar_limiar_f1,
)


def dados_exemplo():
    return pd.DataFrame({
        "data_inversa": ["2025-01-01", "2025-02-02"],
        "dia_semana": ["quarta-feira", "domingo"],
        "horario": ["06:30:00", "18:00:00"],
        "uf": ["SP", "MG"],
        "br": [116, 381],
        "km": ["225,5", "10"],
        "tipo_acidente": ["Colisão traseira", "Tombamento"],
        "fase_dia": ["Pleno dia", "Plena noite"],
        "sentido_via": ["Crescente", "Decrescente"],
        "condicao_metereologica": ["Céu Claro", "Chuva"],
        "tipo_pista": ["Dupla", "Simples"],
        "tracado_via": ["Reta", "Curva"],
        "uso_solo": ["Sim", "Não"],
        "mortos": [0, 1],
        "feridos_graves": [0, 0],
        "veiculos": [2, 1],
        "latitude": ["-23,5", "-19,9"],
        "longitude": ["-46,6", "-44,0"],
    })


def test_criar_alvo_sem_imputar_desfecho():
    dados = dados_exemplo()
    assert criar_alvo(dados).tolist() == [0, 1]

    dados.loc[0, "mortos"] = np.nan
    with pytest.raises(ValueError, match="valores ausentes"):
        criar_alvo(dados)


def test_preparar_atributos_exclui_vazamento():
    atributos = preparar_atributos(dados_exemplo())
    assert atributos.columns.tolist() == ATRIBUTOS
    assert not (set(atributos.columns) & COLUNAS_VAZAMENTO)
    assert atributos.loc[0, "br"] == "BR-116"
    assert atributos.loc[0, "km"] == pytest.approx(225.5)


def test_converter_numero_brasileiro():
    resultado = converter_numero(pd.Series(["1,25", "-3,5", None, "inválido"]))
    assert resultado.iloc[0] == pytest.approx(1.25)
    assert resultado.iloc[1] == pytest.approx(-3.5)
    assert resultado.iloc[2:].isna().all()


def test_selecao_limiar_retorna_intervalo_valido():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.6, 0.9])
    limiar, desempenho = selecionar_limiar_f1(y, prob)
    assert 0 <= limiar <= 1
    assert desempenho["F1"] == pytest.approx(1.0)


def test_priorizacao_concentra_positivos():
    y = np.array([0, 1, 0, 1, 0])
    prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    resultado = avaliar_priorizacao(y, prob, fracao=0.4)
    assert resultado["Recall"] == pytest.approx(1.0)
    assert resultado["Precisao"] == pytest.approx(1.0)
    assert resultado["Lift"] == pytest.approx(2.5)

