# Classificação da gravidade de acidentes em rodovias federais

Projeto de aprendizado de máquina aplicado à engenharia de transportes. O objetivo é classificar acidentes registrados pela Polícia Rodoviária Federal como graves ou não graves, usando características disponíveis no registro inicial da ocorrência.

Um acidente é considerado grave quando possui pelo menos uma pessoa morta ou ferida gravemente. O modelo é apresentado como prova de conceito para priorização de atendimento e análise de segurança viária, não como sistema operacional da PRF.

## Estrutura

- `GUIA_APRESENTACAO_NOTEBOOK.md`: roteiro cronometrado e fala de apoio para cada célula;
- `output/pdf/guia_apresentacao_notebook.pdf`: versão em PDF do guia de apresentação;
- `relatorio/relatorio.tex`: manuscrito completo em LaTeX, com tabelas e figuras;
- `projeto_prf_classificacao.ipynb`: notebook principal;
- `projeto_prf_classificacao_executado.ipynb`: versão com resultados;
- `scripts/baixar_dados.py`: download reproduzível dos dados oficiais;
- `src/prf_ml.py`: regras centrais de leitura, alvo, atributos e métricas;
- `tests/`: testes contra vazamento e inconsistências de preparação;
- `requirements.txt`: dependências;
- `RELATORIO_BASE.md`: texto-base do relatório;
- `ROTEIRO_VIDEO.md`: roteiro para vídeo de até cinco minutos;
- `resultados/`: figuras e tabelas geradas.

## Fonte dos dados

São utilizados os arquivos de acidentes agrupados por ocorrência de 2017 a 2025, publicados no [Portal de Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf). O sistema BAT é utilizado pela PRF desde 2017, razão pela qual anos anteriores não são misturados à análise.

## Metodologia

- treino: 2017 a 2023;
- validação e escolha do modelo/limiar: 2024;
- teste final: 2025;
- alvo: `mortos > 0` ou `feridos_graves > 0`;
- modelos: baseline, regressão logística, Random Forest e HistGradientBoosting;
- métrica principal: PR-AUC;
- limiar equilibrado: escolhido em 2024 pelo maior F1;
- calibração sigmoide: ajustada em 2024 após a seleção do modelo;
- cenário operacional: priorização das 20% ocorrências com maior probabilidade.

Colunas com informações sobre vítimas, pessoas, classificação final e identificadores são excluídas. A causa declarada também é excluída porque pode resultar de investigação posterior. O tipo de acidente e o número de veículos são mantidos, assumindo uma aplicação depois da comunicação inicial da ocorrência.

## Execução

É necessário Python 3.10 ou superior e acesso à internet para o primeiro download.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\baixar_dados.py
python -m pytest
jupyter lab projeto_prf_classificacao.ipynb
```

Depois, execute **Kernel > Restart Kernel and Run All Cells**.

Para executar sem interface:

```powershell
jupyter nbconvert --to notebook --execute projeto_prf_classificacao.ipynb --output projeto_prf_classificacao_executado.ipynb --ExecutePreprocessor.timeout=1800
```

## Interpretação responsável

O banco contém apenas acidentes registrados, não todos os veículos que circularam em cada trecho. Portanto, contagens de acidentes não devem ser interpretadas como taxa de risco sem dados de exposição, como volume de tráfego. As importâncias do modelo indicam associação preditiva, não causalidade.

## Resultados da execução de referência

O HistGradientBoosting foi selecionado usando exclusivamente a validação de 2024. No teste de 2025, com 72.529 ocorrências, foram obtidos:

- PR-AUC: 0,525, contra prevalência/baseline de 0,283;
- IC95% da PR-AUC: 0,517 a 0,532;
- ROC-AUC: 0,710;
- Brier score calibrado: 0,176, contra 0,203 do baseline;
- precisão: 41,4%;
- recall: 65,6%;
- F1: 0,507;
- balanced accuracy: 64,5%.

Ao priorizar somente as 20% ocorrências com maior probabilidade, o modelo concentrou 38,9% dos acidentes graves. A precisão desse grupo foi 54,9%, equivalente a um lift de 1,94 em relação à prevalência de 2025.

O tipo do acidente foi a variável com maior importância por permutação, seguido por quantidade de veículos, BR, UF e traçado da via. Isso é compatível com a aplicação após a comunicação inicial e limita o uso preventivo do modelo.

No estudo de ablação, a retirada de `tipo_acidente` e `veiculos` reduziu a PR-AUC para 0,395, uma queda de 24,6%. Portanto, a documentação não apresenta o modelo completo como ferramenta preventiva anterior à ocorrência.
