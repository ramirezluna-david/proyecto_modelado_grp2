import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay

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

def comparar_modelos_clasificacion(diccionario_modelos: dict, X_test: np.array, y_test: np.array, sort_by="roc_auc"):
    """
    Toma un diccionario de modelos de clasificación entrenados, los evalúa en el 
    conjunto de prueba y retorna un DataFrame ordenado con la comparación de métricas.
    
    Parámetros
    ----------
    diccionario_modelos : dict
        Diccionario con nombres de modelos como llaves y los estimadores como valores.
    X_test : np.array
        Conjunto de datos de prueba.
    y_test : np.array
        Etiquetas de prueba.
    sort_by : str
        Métrica por la cual ordenar el DataFrame resultante (accuracy, f1, roc_auc).
    """
    resultados = []
    
    for nombre, modelo in diccionario_modelos.items():
        metricas = evaluar_classifier(modelo, X_test, y_test)
        metricas['Modelo'] = nombre
        resultados.append(metricas)
        
    df_resultados = pd.DataFrame(resultados).set_index('Modelo')
    
    # Ordenar descendentemente ya que en clasificación (Accuracy, F1, ROC AUC) mayor es mejor
    if sort_by in df_resultados.columns:
        df_resultados = df_resultados.sort_values(by=sort_by, ascending=False)
        
    return df_resultados

def graficar_comparacion_metricas(df_metricas, metricas, target_name, colores=None, y_lim=None, label_fmt='%.2f'):
    """
    Genera gráficos de barras comparativos para múltiples métricas de evaluación.
    
    Parámetros
    ----------
    df_metricas : pd.DataFrame
        DataFrame con los modelos como índice y las métricas como columnas.
    metricas : list
        Lista de strings con los nombres de las columnas de las métricas a graficar.
    target_name : str
        Nombre de la variable objetivo (para el título).
    colores : list, opcional
        Paleta de colores a utilizar en seaborn.
    y_lim : tuple, opcional
        Límites para el eje Y, ej: (0, 1) para métricas de clasificación.
    label_fmt : str, opcional
        Formato de los números sobre las barras, ej: '%.2f' o '%.4f'.
    """
    n_metricas = len(metricas)
    
    # Ajustar el ancho de la figura dinámicamente según la cantidad de métricas
    fig, axes = plt.subplots(1, n_metricas, figsize=(6 * n_metricas, 5))
    
    # Asegurar que 'axes' sea iterable incluso si solo hay 1 métrica
    if n_metricas == 1:
        axes = [axes]
        
    for i, metrica in enumerate(metricas):
        sns.barplot(
            x=df_metricas.index,
            y=df_metricas[metrica],
            hue=df_metricas.index,
            ax=axes[i],
            palette=colores,
            legend=False,
        )
        
        # Formateo de textos y etiquetas
        axes[i].set_title(f'{metrica.upper()} - {target_name}', fontsize=14)
        axes[i].set_ylabel(metrica.upper())
        axes[i].set_xlabel('')
        
        # Aplicar límites al eje Y si se especifican (útil para Accuracy, F1, etc.)
        if y_lim:
            axes[i].set_ylim(y_lim)
            
        # Añadir valores sobre las barras
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt=label_fmt, padding=3)

    plt.tight_layout()
    plt.show()

def graficar_matriz_confusion(modelo, X_test, y_test, clases=["No", "Sí"], titulo="Matriz de Confusión", cmap="Blues"):
    """
    Calcula y grafica la matriz de confusión a partir de un modelo entrenado.
    
    Parámetros
    ----------
    modelo : BaseEstimator
        Modelo de clasificación entrenado (ej. un pipeline).
    X_test : np.array o pd.DataFrame
        Conjunto de datos de prueba.
    y_test : np.array o pd.Series
        Etiquetas reales del conjunto de prueba.
    clases : list, opcional
        Etiquetas a mostrar en los ejes (ej. ["No", "Sí"] para Abandono).
    titulo : str, opcional
        Título principal del gráfico.
    cmap : str, opcional
        Mapa de colores de matplotlib (por defecto "Blues").
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # from_estimator realiza la predicción (modelo.predict) y grafica automáticamente
    ConfusionMatrixDisplay.from_estimator(
        estimator=modelo,
        X=X_test,
        y=y_test,
        display_labels=clases,
        cmap=cmap,
        ax=ax,
        colorbar=False # Oculta la barra de color lateral para un diseño más limpio
    )
    
    plt.title(titulo, fontweight="bold", fontsize=14)
    plt.xlabel("Predicción", fontweight="bold", fontsize=12)
    plt.ylabel("Valor real", fontweight="bold", fontsize=12)
    
    plt.tight_layout()
    plt.show()