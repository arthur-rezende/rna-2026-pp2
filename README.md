# PP2.1 — Perceptron de Rosenblatt (Redes Neurais Artificiais, 2026.2)

Projeto Prático 2.1 da disciplina de **Redes Neurais Artificiais** (EST/UEA): implementação,
em Python com orientação a objetos, do algoritmo de treinamento supervisionado do neurônio
**Perceptron de Rosenblatt** aplicado à classificação binária, em três partes. A implementação
usa exclusivamente NumPy para as operações matriciais; `scikit-learn` só entra no cálculo de
métricas, e `pandas`/`prettytable`/`matplotlib` dão suporte a tabelas e figuras.

## Equipe

| Integrante | Matrícula |
|---|---|
| Arthur Marshall | _2415310048_ |
| Bruno Brás | _2415310041_ |

**ID da equipe:** (soma do último dígito das matrículas) mod 4 = **1**.
Consequência: a Parte II usa `data/data120262.txt` e a semente aleatória da Parte III é 1.

## Estrutura do repositório

```
src/perceptron_pp/     implementação orientada a objetos
  ├─ perceptron.py     neurônio (bias via entrada -1, degrau com θ, ajustes/épocas, reta de decisão)
  ├─ metricas.py       matriz de confusão, acurácia/precisão/revocação/F1, tabelas da Parte II
  └─ visualizacao.py   fronteira de decisão (classe 0 = vermelho, classe 1 = azul), treino×teste, boxplots
notebooks/
  ├─ 01_parte1.ipynb   Parte I — problema linearmente separável (dataAll20262)
  ├─ 02_parte2.ipynb   Parte II — experimentação 6×10 (data120262, ID 1)
  └─ 03_parte3.ipynb   Parte III — validação holdout (dataHoldout20262)
data/                  datasets em formato binário NumPy (m, 1, 3)
resultados/            figuras geradas pelos notebooks
scripts/gerar_tabela.py tabela resumo da Parte II fora do Jupyter (apoio)
```

## Ambiente

```bash
conda env create -f environment.yml      # rna2026-pp2-perceptron (python 3.11)
conda activate rna2026-pp2-perceptron
jupyter lab
```

Alternativa com venv: `python -m venv .venv`, `pip install -e .` e abrir os notebooks com esse kernel.

## Como executar

Abrir cada notebook em `notebooks/` e executar **Kernel → Restart & Run All** — as células
devem ser entregues com os resultados da execução visíveis. Em `02_parte2.ipynb` o orçamento
de épocas é parâmetro (`max_epocas`); o relatório citado usa **500 épocas** por execução.

## Resumo dos resultados e discussões

**Parte I** (`dataAll20262`, η = 0.1, w ~ U(−0.5, +0.5), θ = 0): convergiu em **5 épocas** com
**79 ajustes**; fronteira resultante x2 ≈ 0.196·x1. A discussão completa (teorema de convergência,
escala dos pesos) está no notebook.

**Parte II** (`data120262`, ID 1; η ∈ {0.4, 0.1, 0.01} × I ∈ {(−100,+100), (−0.5,+0.5)}, 10 repetições):
o conjunto **não é linearmente separável**, a taxa de erro estaciona em ~1,9% por época mesmo com
20.000 épocas, e o teorema de Novikoff + a distância mínima entre classes opostas descartam a
separabilidade (ver diagnóstico no notebook). Por isso a métrica "épocas até a convergência" não é
atingível: a tabela registra a coluna **Convergiu = Não**, Min/Max Épocas = "—" e compara as 6
configurações **com orçamento fixo de 500 épocas** (velocidade de aprendizado). Discussão de
equivalência entre configurações no notebook.

**Parte III** (`dataHoldout20262`; split 70/30 com seed 1; normalização min-max ajustada só no treino;
100 épocas com apresentação aleatória dos exemplos): o problema não é linearmente separável e a solução
linear resultante prevê quase sempre a classe 0 — acurácia de teste ≈ **0.563**, precisão ≈ **1.0**,
revocação ≈ **0.078**, F1 ≈ **0.144**. A discussão sobre a qualidade da solução (e as justificativas
de projeto, como a normalização) está no notebook.
