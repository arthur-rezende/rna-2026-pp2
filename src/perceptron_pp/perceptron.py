from __future__ import annotations

import numpy as np


class Perceptron:

    def __init__(self, eta=0.1, intervalo_pesos=(-0.1, 0.1), theta=0.0, rng=None):
        self.eta = eta # taxa de aprendizado
        self.intervalo_pesos = intervalo_pesos
        self.theta = theta # Limiar da função degrau
        self.rng = np.random.default_rng(rng) # 'seed'


    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.hstack([-np.ones((X.shape[0], 1)), X])


    def _ativacao(self, u):
        return (np.asarray(u) >= self.theta).astype(int)


    def _inicializar_pesos(self, n_pesos: int) -> np.ndarray:
        a, b = self.intervalo_pesos
        return self.rng.uniform(a, b, size=n_pesos)


    def fit(self, X, yd, max_epocas=None, embaralhar=False, verbose=True):
        # Treinamento do neurônio
        
        Xb = self._com_bias(X)
        yd = np.asarray(yd, dtype=int).ravel()
        if Xb.shape[0] != yd.shape[0]:
            raise ValueError("X e yd devem ter o mesmo número de amostras.")
 
        self.w_inicial_ = self._inicializar_pesos(Xb.shape[1])
        self.w_ = self.w_inicial_.copy()
        if verbose:
            print("Vetor inicial de pesos [w0, w1, w2]:", self.w_inicial_)
 
        self.n_ajustes_ = 0
        self.n_epocas_ = 0
        self.erros_por_epoca_ = []
        self.convergiu_ = False
 
        while max_epocas is None or self.n_epocas_ < max_epocas:
            self.n_epocas_ += 1
            ordem = self.rng.permutation(len(Xb)) if embaralhar else np.arange(len(Xb))
            erros = 0
 
            for i in ordem:
                y = self._ativacao(Xb[i] @ self.w_)
                erro = yd[i] - y
                if erro != 0:
                    self.w_ += self.eta * erro * Xb[i]
                    self.n_ajustes_ += 1
                    erros += 1
 
            self.erros_por_epoca_.append(erros)
            if erros == 0:
                self.convergiu_ = True
                break
 
        return self
 
    def predict(self, X) -> np.ndarray:
        self._verificar_treinado()
        return self._ativacao(self._com_bias(X) @ self.w_)
 
    def reta_decisao(self, x1):
        self._verificar_treinado()
        w0, w1, w2 = self.w_
        if np.isclose(w2, 0.0):
            raise ZeroDivisionError("w2 = 0: a fronteira é uma reta vertical.")
        return (self.theta + w0 - w1 * np.asarray(x1)) / w2
 
    def _verificar_treinado(self):
        if not hasattr(self, "w_"):
            raise RuntimeError("Chame fit() antes de usar o modelo.")
 
    def __repr__(self):
        return (f"Perceptron(eta={self.eta}, intervalo_pesos={self.intervalo_pesos}, "
                f"theta={self.theta})")