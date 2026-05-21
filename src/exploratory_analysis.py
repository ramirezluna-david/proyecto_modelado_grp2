from __future__ import annotations

import pandas as pd

def calcular_correlacion_objetivo(
    dataframe: pd.DataFrame,
    target: str,
    *,
    metodo: str = "pearson",
    numeric_only: bool = True,
    correlaciones: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Calcula la correlacion con la variable objetivo y la ordena.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Conjunto de datos con variables numericas.
    target : str
        Nombre de la variable objetivo.
    metodo : str, default="pearson"
        Metodo de correlacion a usar.
    numeric_only : bool, default=True
        Si True, usa solo columnas numericas.
    correlaciones : pd.DataFrame | None
        Matriz de correlaciones precalculada (opcional).

    Returns
    -------
    pd.Series
        Correlaciones ordenadas respecto a la variable objetivo.
    """
    if correlaciones is None:
        correlaciones = dataframe.corr(method=metodo, numeric_only=numeric_only)
    return correlaciones[target].sort_values(ascending=False)

def seleccionar_mejores_predictores(
    dataframe: pd.DataFrame,
    target: str,
    umbral: float = 0.008,
    incluir_objetivo: bool = False,
) -> pd.DataFrame:
    """
    Selecciona variables predictoras usando la matriz de correlaciones de Pearson.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Conjunto de datos con variables numericas.
    target : str
        Nombre de la variable objetivo.
    umbral : float, default=0.008
        Umbral minimo de correlacion (valor absoluto).
    incluir_objetivo : bool, default=False
        Si True, mantiene la variable objetivo en el resultado.

    Returns
    -------
    pd.DataFrame
        Tabla con variables y correlaciones ordenadas.
    """
    correlaciones = dataframe.corr(numeric_only=True)
    correlacion_objetivo = correlaciones[target].sort_values(ascending=False)
    correlacion_abs = correlacion_objetivo.abs()
    mask = correlacion_abs >= umbral
    if not incluir_objetivo:
        mask = mask & (correlacion_abs < 1.0)
    seleccion = correlacion_objetivo[mask].sort_values(ascending=False)
    salida = seleccion.reset_index()
    salida.columns = ["variable", "correlacion"]
    salida["correlacion_abs"] = salida["correlacion"].abs()
    salida = salida.sort_values("correlacion_abs", ascending=False).reset_index(drop=True)
    salida.index = salida.index + 1
    salida.index.name = "ranking"
    return salida


def seleccionar_mejores_predictores_clasificacion(
    dataframe: pd.DataFrame,
    target: str,
    umbral: float = 0.008,
) -> pd.DataFrame:
    """
    Selecciona variables predictoras para clasificacion usando correlacion de Pearson.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Conjunto de datos con variables numericas.
    target : str
        Nombre de la variable objetivo.
    umbral : float, default=0.008
        Umbral minimo de correlacion (valor absoluto).

    Returns
    -------
    pd.DataFrame
        Tabla con variables y correlaciones ordenadas.
    """
    correlaciones = dataframe.corr(numeric_only=True)
    correlacion_objetivo = correlaciones[target].sort_values(ascending=False)
    correlacion_abs = correlacion_objetivo.abs()
    mask = (correlacion_abs >= umbral) & (correlacion_abs < 1.0)
    seleccion = correlacion_objetivo[mask].sort_values(ascending=False)
    salida = seleccion.reset_index()
    salida.columns = ["variable", "correlacion"]
    salida["correlacion_abs"] = salida["correlacion"].abs()
    salida = salida.sort_values("correlacion_abs", ascending=False).reset_index(drop=True)
    salida.index = salida.index + 1
    salida.index.name = "ranking"
    return salida
