import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from preprocesamiento.data_preprocessing import CorrelationFilter
from preprocesamiento.data_preprocessing import DataFrameConverter
from preprocesamiento.data_preprocessing import FeatureEngineering
from preprocesamiento.data_preprocessing import Winsorizer
from preprocesamiento.data_preprocessing import tratar_duplicados

def build_preprocessor(
    numerical_features,
    categorical_nominales,
    categorical_ordinales,
    orden_tipo_plan,
    orden_uso_app,
):
    """
    Construye pipelines de transformacion y el preprocesador combinado.

    Parameters
    ----------
    numerical_features : list
        Columnas numericas.
    categorical_nominales : list
        Columnas categoricas nominales.
    categorical_ordinales : list
        Columnas categoricas ordinales.
    orden_tipo_plan : list
        Orden para la variable ordinal tipo_plan.
    orden_uso_app : list
        Orden para la variable ordinal uso_app.

    Returns
    -------
    tuple
        (pipeline_numerical_features, pipeline_nominales, pipeline_ordinales, preprocesador)
    """
    pipeline_numerical_features = Pipeline(steps=[
        ("winsorizer", Winsorizer(limits=(0.05, 0.05))),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    pipeline_nominales = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pipeline_ordinales = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[orden_tipo_plan, orden_uso_app])),
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ("num_limpios", pipeline_numerical_features, numerical_features),
            ("cat_nom", pipeline_nominales, categorical_nominales),
            ("cat_ord", pipeline_ordinales, categorical_ordinales),
        ],
        remainder="drop",
    )

    return pipeline_numerical_features, pipeline_nominales, pipeline_ordinales, preprocesador

def build_cleaning_pipeline(preprocesador):
    """
    Construye el pipeline de limpieza con duplicados, features y preprocesamiento.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador combinado.

    Returns
    -------
    Pipeline
        Pipeline de limpieza.
    """
    return Pipeline(
        steps=[
            ("duplicados", FunctionTransformer(tratar_duplicados, kw_args={"drop": True})),
            ("feature_engineering", FeatureEngineering()),
            ("preprocesamiento", preprocesador),
            ("conversion", DataFrameConverter(preprocesador)),
        ]
    )

def build_cleaning_pipeline_reg(preprocesador):
    """
    Construye el pipeline de limpieza con duplicados, features y preprocesamiento.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador combinado.

    Returns
    -------
    Pipeline
        Pipeline de limpieza.
    """
    return Pipeline(
        steps=[
            ("duplicados", FunctionTransformer(tratar_duplicados, kw_args={"drop": True})),
            # ("feature_engineering", FeatureEngineering()),
            ("preprocesamiento", preprocesador),
            ("conversion", DataFrameConverter(preprocesador)),
        ]
    )

def build_model_pipeline_regresion(modelo, threshold=0.9):
    """
    Construye un pipeline de regresion con filtro de colinealidad.

    Parameters
    ----------
    modelo : BaseEstimator
        Modelo de regresion.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline de regresion.
    """
    return Pipeline(
        steps=[
            ("colinealidad", CorrelationFilter(threshold=threshold)),
            ("modelo", modelo),
        ]
    )

def build_linear_regression_pipeline(threshold=0.9):
    """
    Pipeline para regresion lineal.

    Parameters
    ----------
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline de regresion lineal.
    """
    return build_model_pipeline_regresion(LinearRegression(), threshold=threshold)

def build_decision_tree_regressor_pipeline(
    max_depth=7,
    min_samples_leaf=15,
    random_state=42,
    threshold=0.9,
):
    """
    Pipeline para DecisionTreeRegressor.

    Parameters
    ----------
    max_depth : int, default=7
        Profundidad maxima del arbol.
    min_samples_leaf : int, default=15
        Minimo de muestras por hoja.
    random_state : int, default=42
        Semilla aleatoria.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline con DecisionTreeRegressor.
    """
    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    return build_model_pipeline_regresion(modelo, threshold=threshold)

def aplicar_pipeline_limpieza(pipeline_limpieza, var_indep, var_dep, target_col="deuda_total"):
    """
    Aplica el pipeline de limpieza y devuelve el dataset transformado con objetivo.

    Parameters
    ----------
    pipeline_limpieza : Pipeline
        Pipeline de limpieza entrenable.
    var_indep : DataFrame
        Variables independientes.
    var_dep : Series
        Variable dependiente.
    target_col : str, default="deuda_total"
        Nombre de la columna objetivo en el resultado.

    Returns
    -------
    DataFrame
        Dataset transformado con la columna objetivo.
    """
    data_transformada = pipeline_limpieza.fit_transform(var_indep)
    if isinstance(data_transformada, pd.DataFrame):
        df = data_transformada.copy()
    else:
        df = pd.DataFrame(data_transformada)

    if hasattr(df, "columns"):
        df.columns = df.columns.str.replace("num_limpios__", "", regex=False)
        df.columns = df.columns.str.replace("cat_nom__", "", regex=False)
        df.columns = df.columns.str.replace("cat_ord__", "", regex=False)

    df[target_col] = var_dep.to_numpy()
    return df


def _limpiar_prefijos_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina prefijos generados por el ColumnTransformer en los nombres de columnas.
    """
    if hasattr(df, "columns"):
        df = df.copy()
        df.columns = df.columns.str.replace("num_limpios__", "", regex=False)
        df.columns = df.columns.str.replace("cat_nom__", "", regex=False)
        df.columns = df.columns.str.replace("cat_ord__", "", regex=False)
    return df


def aplicar_pipeline_limpieza_train_test(pipeline_limpieza, X_train, X_test):
    """
    Ajusta el pipeline con train y transforma train/test sin leakage.

    Parameters
    ----------
    pipeline_limpieza : Pipeline
        Pipeline de limpieza entrenable.
    X_train : DataFrame
        Variables independientes de entrenamiento.
    X_test : DataFrame
        Variables independientes de prueba.

    Returns
    -------
    tuple
        (X_train_t, X_test_t) transformados y con columnas normalizadas.
    """
    X_train_t = pipeline_limpieza.fit_transform(X_train)
    X_test_t = pipeline_limpieza.transform(X_test)

    if not isinstance(X_train_t, pd.DataFrame):
        X_train_t = pd.DataFrame(X_train_t)
    if not isinstance(X_test_t, pd.DataFrame):
        X_test_t = pd.DataFrame(X_test_t)

    X_train_t = _limpiar_prefijos_columnas(X_train_t)
    X_test_t = _limpiar_prefijos_columnas(X_test_t)

    return X_train_t, X_test_t