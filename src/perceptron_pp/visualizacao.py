from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fronteira_decisao(
        X: np.ndarray,
        y: np.ndarray,
        modelo,
        titulo: str = "Fronteira de Decisão do Perceptron",
        ax: plt.Axes | None = None,
        mostrar: bool = True,
) -> plt.Axes:
    """Plota a dispersão dos dados (Classe 0: Vermelho, Classe 1: Azul) e a reta de decisão do Perceptron.

    Respeita a convenção de cores definida no projeto prático.
    """
    criou_figura = ax is None
    if criou_figura:
        fig, ax = plt.subplots(figsize=(8, 6))

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Separação das classes conforme requisitos (0: vermelho, 1: azul)
    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="red",
        label="Classe 0",
        alpha=0.7,
        edgecolors="k",
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="blue",
        label="Classe 1",
        alpha=0.7,
        edgecolors="k",
    )

    # Plotagem da reta de separação aprendida
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_vals = np.linspace(x1_min, x1_max, 200)

    try:
        x2_vals = modelo.reta_decisao(x1_vals)
        ax.plot(
            x1_vals,
            x2_vals,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Fronteira de Decisão",
        )
    except ZeroDivisionError:
        w0, w1, _ = modelo.w_
        x1_vert = (modelo.theta + w0) / w1
        ax.axvline(
            x=x1_vert,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Fronteira (Vertical)",
        )

    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    ax.set_ylim(x2_min, x2_max)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(titulo)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)

    if mostrar and criou_figura:
        plt.show()

    return ax


def plot_comparacao_treino_teste(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        modelo,
        titulo: str = "Validação Holdout - Treino vs Teste",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Gera dois gráficos lado a lado (Treino e Teste) exibindo a mesma reta de decisão obtida no treino."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    plot_fronteira_decisao(
        X_train,
        y_train,
        modelo,
        titulo="Conjunto de Treinamento (70%)",
        ax=ax1,
        mostrar=False,
    )
    plot_fronteira_decisao(
        X_test,
        y_test,
        modelo,
        titulo="Conjunto de Teste (30%)",
        ax=ax2,
        mostrar=False,
    )

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)

ROTULOS_METRICAS = {
    "ajustes": "Quantidade de ajustes no vetor de pesos",
    "epocas": "Épocas até a convergência",
}


def plot_boxplots_experimentos(
        df: pd.DataFrame,
        coluna_metricas: tuple[str, ...] = ("ajustes", "epocas"),
        titulo: str = "Parte II - Distribuição do Número de Ajustes por Configuração",
        escala_log: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    # Gera os boxplots das 6 configurações testadas na Parte II para comparação do número de ajustes.
    configs = list(df.groupby(["eta", "intervalo"], sort=False).groups.keys())
    rotulos = [f"n={eta}\nI={intervalo}" for eta, intervalo in configs]

    fig, axes = plt.subplots(1, len(coluna_metricas),
                             figsize=(7 * len(coluna_metricas), 5), squeeze=False)
    axes = list(axes.ravel())

    for ax, coluna in zip(axes, coluna_metricas):
            grupos = [df.loc[(df["eta"] == e) & (df["intervalo"] == i), coluna].to_numpy()
                    for e, i in configs]
            ax.boxplot(grupos, tick_labels=rotulos)
            if escala_log:
                ax.set_yscale("log")
            ax.set_title(ROTULOS_METRICAS.get(coluna, coluna))
            ax.set_xlabel("Configuração (η, intervalo de pesos)")
            ax.set_ylabel(ROTULOS_METRICAS.get(coluna, coluna)
                        + (" (escala log)" if escala_log else ""))
            ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    return fig, axes