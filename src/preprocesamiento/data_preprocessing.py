import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CorrelationFilter(BaseEstimator, TransformerMixin):
  """
  Eliminación de variables correlacionadas

  Parámetros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  columns_to_drop_ : array-like
    Nombres de las columnas a eliminar.
  threshold : float
    Umbral de correlación.
  Returns
  -------
  DataFrame
    Conjunto de datos sin variables correlacionadas.
  """
  def __init__(self, threshold=0.9):
    self.threshold = threshold
    self.columns_to_drop_ = None

  def fit(self, X, y=None):
    X_df = pd.DataFrame(X)

    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    self.columns_to_drop_ = [
        col for col in upper.columns if any(upper[col] > self.threshold)
    ]

    return self

  def transform(self, X):
    X_df = pd.DataFrame(X)
    X_filtered = X_df.drop(columns=self.columns_to_drop_, errors="ignore")
    return X_filtered.values

class DataFrameConverter(BaseEstimator, TransformerMixin):
  """
  Convierte un array en un DataFrame

  Parámetros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  feature_names_ : array-like
    Nombres de las columnas.
  Returns
  -------
  DataFrame
    Conjunto de datos con nombres de columnas.
  """
  def __init__(self, preprocessor):
    self.preprocessor = preprocessor
    self.feature_names_ = None

  def fit(self, X, y=None):
    # Obtener nombres después de fit del preprocessor
    self.feature_names_ = self.preprocessor.get_feature_names_out()
    return self

  def transform(self, X):
    return pd.DataFrame(X, columns=self.feature_names_)
  
class Winsorizer(BaseEstimator, TransformerMixin):
  """
  Winsorizacion de variables numericas.

  Parametros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  limits : tuple of float
    Limites inferior y superior para el recorte de colas.
  lower_bounds_ : Series
    Cuantiles inferiores por columna (calculados en fit).
  upper_bounds_ : Series
    Cuantiles superiores por columna (calculados en fit).
  columns_ : array-like
    Nombres de columnas usadas para transformar.
  Returns
  -------
  DataFrame
    Conjunto de datos con valores recortados.
  """
  def __init__(self, limits=(0.05, 0.05)):
    """
    Inicializa el transformador.

    Parametros
    ----------
    limits : tuple of float, default=(0.05, 0.05)
      Porcentaje de recorte en las colas inferior y superior.
    """
    self.limits = limits

  def fit(self, X, y=None):
      """
      Calcula y almacena los limites por columna.

      Parametros
      ----------
      X : array-like o DataFrame
        Conjunto de datos de entrada.
      y : Ignorado

      Returns
      -------
      self
        Instancia entrenada.
      """
      X_df = pd.DataFrame(X)
        # Calculamos y guardamos los límites matemáticos AQUÍ (Fase de aprendizaje)
      self.lower_bounds_ = X_df.quantile(self.limits[0])
      self.upper_bounds_ = X_df.quantile(1 - self.limits[1])
      self.columns_ = X_df.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
      return self

  def transform(self, X):
      """
      Aplica el recorte de valores segun los limites aprendidos.

      Parametros
      ----------
      X : array-like o DataFrame
        Conjunto de datos de entrada.

      Returns
      -------
      DataFrame
        Conjunto de datos con valores recortados.
      """
      X_df = pd.DataFrame(X, columns=self.columns_).copy()
      X_df = X_df.astype("float64")
        # Aplicamos los límites guardados
      for col in self.columns_:
        X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df

  def get_feature_names_out(self, input_features=None):
      """
      Devuelve nombres de caracteristicas.

      Parametros
      ----------
      input_features : array-like, optional
        Nombres de entrada a devolver.

      Returns
      -------
      ndarray
        Nombres de las columnas.
      """
      if input_features is None:
        return np.array(self.columns_)
      else:
        return np.array(input_features)
        
# Define función para eliminar duplicados
def tratar_duplicados(X : pd.DataFrame, drop = True):
  """
  Tratamiento de duplicados

  Parámetros
  ----------
  X : DataFrame
    Conjunto de datos.
  drop : bool
    Si se deben eliminar los duplicados.

  Retorna
  -------
  DataFrame
    Conjunto de datos sin duplicados.
  """
  return X.drop_duplicates() if drop else X

class FeatureEngineeringRegression(BaseEstimator, TransformerMixin):
  """
  Ingeniería de características

  Parámetros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  columns_ : array-like
    Nombres de las columnas a transformar.
  Returns
  -------
  DataFrame
    Conjunto de datos con nuevas características.
  """
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()

    # # razón de endeudamiento
    # X["ratio_endeudamiento"] = X["deuda_total"] / X["ingreso_mensual"]

    # porcentaje de gasto respecto al ingreso
    X['porcentaje_gasto'] = X['gasto_mensual'] / X['ingreso_mensual']

    return X