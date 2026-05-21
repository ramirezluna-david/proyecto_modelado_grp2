import numpy as np

from sklearn.base import BaseEstimator


def entrenar_modelo(modelo: BaseEstimator, X_train: np.array, y_train: np.array):
    """
    Entrena un modelo y devuelve la instancia entrenada.

    Parametros
    ----------
    modelo : BaseEstimator
        Modelo a entrenar.
    X_train : np.array
        Conjunto de datos de entrenamiento.
    y_train : np.array
        Etiquetas de entrenamiento.

    Returns
    -------
    BaseEstimator
        Modelo entrenado.
    """
    modelo.fit(X_train, y_train)
    return modelo
