import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

def evaluar_classifier(modelo: BaseEstimator, X_test: np.array, y_test: np.array):
  """
  Evalua un clasificador y retorna sus metricas.

  Parámetros
  ----------
  modelo : BaseEstimator
    Modelo a evaluar (debe exponer predict_proba y estar entrenado).
  X_test : np.array
    Conjunto de datos de prueba.
  y_test : np.array
    Etiquetas de prueba.
  Returns
  -------
  dict
    Diccionario con accuracy, f1 y roc_auc.

  """
  y_pred = modelo.predict(X_test)
  y_prob = modelo.predict_proba(X_test)[:,1]

  return {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
  }

def evaluar_regresor(modelo: BaseEstimator, X_test: np.array, y_test: np.array):
  """
  Evalua un modelo de regresion y retorna sus metricas.

  Parámetros
  ----------
  modelo : BaseEstimator
    Modelo a evaluar (debe estar entrenado).
  X_test : np.array
    Conjunto de datos de prueba.
  y_test : np.array
    Etiquetas de prueba.

  Returns
  -------
  dict
    Diccionario con mae y r2.
  """
  y_pred = modelo.predict(X_test)
    
  return {
      "MAE": mean_absolute_error(y_test, y_pred),
      "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
      "R2": r2_score(y_test, y_pred),
  }

def comparar_metricas(resultados: dict, sort_by=None, ascending=False):
  """
  Convierte un diccionario de metricas en un DataFrame y opcionalmente ordena.

  Parametros
  ----------
  resultados : dict
    Diccionario con nombre_modelo -> metricas.
  sort_by : str or None
    Columna por la que ordenar.
  ascending : bool
    Orden ascendente si es True.

  Returns
  -------
  DataFrame
    Tabla de metricas por modelo.
  """
  df = pd.DataFrame(resultados).T
  if sort_by and sort_by in df.columns:
    df = df.sort_values(by=sort_by, ascending=ascending)
  return df

def comparar_modelos_regresion(diccionario_modelos: dict, X_test: np.array, y_test: np.array, sort_by="R2"):
    """
    Toma un diccionario de modelos entrenados, los evalúa en el conjunto de prueba
    y retorna un DataFrame ordenado con la comparación de métricas.
    
    Parámetros
    ----------
    diccionario_modelos : dict
        Diccionario con nombres de modelos como llaves y los estimadores como valores.
    X_test : np.array
        Conjunto de datos de prueba.
    y_test : np.array
        Etiquetas de prueba.
    sort_by : str
        Métrica por la cual ordenar el DataFrame resultante.
    """
    resultados = []
    
    for nombre, modelo in diccionario_modelos.items():
        metricas = evaluar_regresor(modelo, X_test, y_test)
        metricas['Modelo'] = nombre
        resultados.append(metricas)
        
    df_resultados = pd.DataFrame(resultados).set_index('Modelo')
    
    # Ordenar adecuadamente (R2 descendente es mejor; MAE/RMSE ascendente es mejor)
    if sort_by in df_resultados.columns:
        ascending = False if sort_by == "R2" else True
        df_resultados = df_resultados.sort_values(by=sort_by, ascending=ascending)
        
    return df_resultados