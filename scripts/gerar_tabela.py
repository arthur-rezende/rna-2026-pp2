# ESSA PARTE EU ASSUMO QUE FIZ COM I.A SÓ PARA VISUALIZAR COMO SERIA FEITA A SEGUNDA PARTE DA
# ATIVIDADE, NÃO PRETENDO LEVAR PARA A CODEBASE FINAL



#!/usr/bin/env python3
"""Script para automação da Parte II (Experimentação).

Executa 10 repetições para cada uma das 6 configurações exigidas, calcula as estatísticas
e exibe a tabela formatada no terminal via PrettyTable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Garante que a pasta src/ esteja acessível no caminho de importação
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ_PROJETO / "src"))

from perceptron_pp.metricas import tabela_resumo
from perceptron_pp.perceptron import Perceptron


def carregar_dataset(caminho_arquivo: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega o array binário .txt no formato (m, 1, 3) e o remodela para (m, 3)."""
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    dados = np.load(caminho_arquivo)
    if dados.ndim == 3:
        dados = dados.reshape(-1, 3)

    X = dados[:, :2]
    y = dados[:, 2]
    return X, y


def rodar_experimentos(
        X: np.ndarray,
        y: np.ndarray,
        etas: list[float] = [0.4, 0.1, 0.01],
        intervalos: list[tuple[float, float]] = [(-100.0, 100.0), (-0.5, 0.5)],
        n_repeticoes: int = 10,
        seed_base: int = 42,
) -> pd.DataFrame:
    """Executa as N repetições para as 6 combinações de (eta, intervalo_pesos)."""
    resultados = []
    rng_master = np.random.default_rng(seed_base)

    for eta in etas:
        for inter in intervalos:
            for rep in range(n_repeticoes):
                seed_exec = int(rng_master.integers(0, 1_000_000))

                modelo = Perceptron(
                    eta=eta,
                    intervalo_pesos=inter,
                    theta=0.0,
                    rng=seed_exec,
                )
                modelo.fit(X, y, verbose=False)

                resultados.append({
                    "eta": eta,
                    "intervalo": str(inter),
                    "repeticao": rep + 1,
                    "ajustes": modelo.n_ajustes_,
                    "epocas": modelo.n_epocas_,
                    "convergiu": modelo.convergiu_,
                })

    return pd.DataFrame(resultados)


def main():
    parser = argparse.ArgumentParser(
        description="Executa a bateria de testes da Parte II do PP2.1."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataAll20262.txt",
        help="Caminho relativo do dataset (ex: data/data020262.txt)",
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=10,
        help="Número de repetições por configuração (padrão: 10)",
    )
    args = parser.parse_args()

    caminho_dados = RAIZ_PROJETO / args.data
    print(f"[+] Carregando dados de: {caminho_dados}")
    X, y = carregar_dataset(caminho_dados)

    print(f"[+] Executando {args.repeticoes} repetições para cada uma das 6 configurações...")
    df_resultados = rodar_experimentos(X, y, n_repeticoes=args.repeticoes)

    print("\n" + "=" * 65)
    print("TABELA RESUMO DOS EXPERIMENTOS (PARTE II)")
    print("=" * 65)
    tabela = tabela_resumo(df_resultados)
    print(tabela)


if __name__ == "__main__":
    main()