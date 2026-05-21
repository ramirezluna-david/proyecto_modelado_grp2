from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from preprocesamiento.data_preprocessing import Winsorizer

def build_preprocessor(
    numerical_features,
    categorical_nominales,
    categorical_ordinales,
    orden_tipo_plan,
    orden_uso_app,
):
    """
    Construye pipelines de transformacion y el preprocesador combinado.

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