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
    return (
        df.groupby(["eta", "intervalo"], sort=False)
        .agg(ajustes_media=("ajustes", "mean"),
             ajustes_desvio=("ajustes", "std"),
             epocas_min=("epocas", "min"),
             epocas_max=("epocas", "max"))
        .reset_index()
    )
 
 
def tabela_resumo(df: pd.DataFrame) -> PrettyTable:
    resumo = resumo_execucoes(df)
    tabela = PrettyTable(["Taxa de Aprendizado", "Intervalo de Pesos",
                          "Qtde Ajustes", "Min Épocas", "Max Épocas"])
    for _, linha in resumo.iterrows():
        tabela.add_row([
            f"η = {linha['eta']}",
            str(linha["intervalo"]),
            f"{linha['ajustes_media']:.2f} ± {linha['ajustes_desvio']:.2f}",
            int(linha["epocas_min"]),
            int(linha["epocas_max"]),
        ])
    return tabela