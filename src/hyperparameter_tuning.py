# hyperparameter_tuning.py

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from model_evaluation import evaluar_classifier
from model_evaluation import evaluar_regressor

def optimizar_grid_search(pipeline, param_grid, X_train, y_train, cv=5, scoring='neg_mean_absolute_error'):
    """
    Realiza una búsqueda exhaustiva (GridSearchCV) de hiperparámetros.
    """
    print("Iniciando GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1, # Usa todos los núcleos del procesador
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n--- Resultados GridSearchCV ---")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score ({scoring}): {grid_search.best_score_:.4f}")
    
    # Retorna el mejor modelo ya entrenado y sus parámetros
    return grid_search.best_estimator_, grid_search.best_params_


def optimizar_random_search(pipeline, param_distributions, X_train, y_train, n_iter=10, cv=5, scoring='neg_mean_absolute_error', random_state=42):
    """
    Realiza una búsqueda aleatoria (RandomizedSearchCV) de hiperparámetros.
    """
    print(f"Iniciando RandomizedSearchCV con {n_iter} iteraciones...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\n--- Resultados RandomizedSearchCV ---")
    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"Mejor score ({scoring}): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_

def reporte_evaluacion(nombre, mejor_modelo, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"Evaluación Final: {nombre}")
    print(f"{'='*50}")
    
    metricas = evaluar_classifier(mejor_modelo, X_test, y_test)
    
    print(f"{'Accuracy':<20}: {metricas['accuracy']:.4f}")
    print(f"{'F1 Score:':<20}: {metricas['f1']:.4f}")
    print(f"{'ROC AUC Score':<20}: {metricas['roc_auc']:.4f}")

def reporte_evaluacion_reg(nombre, mejor_modelo, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"Evaluación Final: {nombre}")
    print(f"{'='*50}")
    
    metricas = evaluar_regressor(mejor_modelo, X_test, y_test)
    
    print(f"{'MAE':<20}: {metricas['MAE']:.4f}")
    print(f"{'MSE':<20}: {metricas['MSE']:.4f}")
    print(f"{'R2 Score':<20}: {metricas['R2']:.4f}")