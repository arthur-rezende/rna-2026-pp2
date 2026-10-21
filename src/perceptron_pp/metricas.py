from __future__ import annotations
 
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def matriz_confusao(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
 
 
def avaliar(y_true, y_pred) -> dict:
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "revocacao": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
 
 
def tabela_metricas(metricas: dict, titulo: str | None = None) -> PrettyTable:
    tabela = PrettyTable(["Métrica", "Valor"])
    if titulo:
        tabela.title = titulo
    nomes = {"acuracia": "Acurácia", "precisao": "Precisão",
             "revocacao": "Revocação", "f1": "F1-Score"}
    for chave, valor in metricas.items():
        tabela.add_row([nomes.get(chave, chave), f"{valor:.4f}"])
    return tabela
 
 
def resumo_execucoes(df: pd.DataFrame) -> pd.DataFrame:
    agregacoes = {
        "ajustes_media": ("ajustes", "mean"),
        "ajustes_desvio": ("ajustes", "std"),
        "epocas_min": ("epocas", "min"),
        "epocas_max": ("epocas", "max"),
    }
    # Em conjuntos não linearmente separáveis nenhuma execução converge; nesse caso o
    # número de épocas é apenas o teto adotado e não deve ser reportado como convergência.
    if "convergiu" in df.columns:
        agregacoes["convergiu"] = ("convergiu", "all")

    return (
        df.groupby(["eta", "intervalo"], sort=False)
        .agg(**agregacoes)
        .reset_index()
    )
 
 
def tabela_resumo(df: pd.DataFrame) -> PrettyTable:
    resumo = resumo_execucoes(df)
    tem_convergencia = "convergiu" in resumo.columns

    colunas = ["Taxa de Aprendizado", "Intervalo de Pesos",
               "Qtde Ajustes", "Min Épocas", "Max Épocas"]
    if tem_convergencia:
        colunas.append("Convergiu")
    tabela = PrettyTable(colunas)

    for _, linha in resumo.iterrows():
        convergiu = bool(linha["convergiu"]) if tem_convergencia else True
        # Sem convergência, as épocas observadas são apenas o teto do experimento.
        epocas_min = int(linha["epocas_min"]) if convergiu else "—"
        epocas_max = int(linha["epocas_max"]) if convergiu else "—"
        valores = [
            f"η = {linha['eta']}",
            str(linha["intervalo"]),
            f"{linha['ajustes_media']:.2f} ± {linha['ajustes_desvio']:.2f}",
            epocas_min,
            epocas_max,
        ]
        if tem_convergencia:
            valores.append("Sim" if convergiu else "Não")
        tabela.add_row(valores)

    return tabela