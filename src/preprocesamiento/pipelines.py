### Pipelines para cada tipo de conjunto

# Define pipeline para variables numéricas
pipeline_numerical_features = Pipeline(steps=[
    ('winsorizer', Winsorizer(limits=(0.05, 0.05))), # Aplica Winsorización para limitar outliers al 5%
    ('imputer', SimpleImputer(strategy='mean')), # Imputa valores faltantes con el promedio
    ('scaler', StandardScaler()) # Escala características numéricas
])

# Define pipeline para variables categóricas nominales
pipeline_nominales = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Imputa valores faltantes con la moda
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Aplica codificación OneHotEncoder para variables nominales
])

# Define pipeline para variables categóricas ordinales
pipeline_ordinales = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Imputa valores faltantes con la moda
    ('ordinal', OrdinalEncoder(categories=[orden_tipo_plan, orden_uso_app])) # Aplica codificación OrdinalEncoder para variables ordinales con orden definido
])

### Integración de pipelines de transformación
# Combina pipelines para aplicar transformaciones específicas a cada tipo de variable
preprocesador = ColumnTransformer(
    transformers=[
        ('num_limpios', pipeline_numerical_features, numerical_features),
        ('cat_nom', pipeline_nominales, categorical_nominales),
        ('cat_ord', pipeline_ordinales, categorical_ordinales),
    ],
    remainder='drop' # Elimina columnas no especificadas
)