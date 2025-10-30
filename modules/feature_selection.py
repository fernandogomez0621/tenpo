"""
Módulo para selección de características
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class FeatureSelector:
    """Clase para selección de características"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selectors = {}
    
    def select_features(self, X, y, methods, n_features=40):
        """
        Selecciona características usando múltiples métodos
        
        Args:
            X: Features
            y: Target
            methods: Lista de métodos a aplicar
            n_features: Número de features a seleccionar
            
        Returns:
            Diccionario con resultados por método
        """
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        results = {}
        
        if "F-Score" in methods:
            results["F-Score"] = self._fscore_selection(X_scaled, y, X.columns, n_features)
        
        if "Información Mutua" in methods:
            results["Información Mutua"] = self._mutual_info_selection(X_scaled, y, X.columns, n_features)
        
        if "Random Forest" in methods:
            results["Random Forest"] = self._random_forest_selection(X_scaled, y, X.columns, n_features)
        
        if "PCA" in methods:
            results["PCA"] = self._pca_selection(X_scaled, y, X.columns)
        
        return results
    
    def _fscore_selection(self, X, y, columns, n_features):
        """Selección por F-Score"""
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        scores_df = pd.DataFrame({
            'Feature': columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        selected_features = scores_df.head(n_features)['Feature'].tolist()
        
        return {
            'scores': scores_df,
            'selected_features': selected_features,
            'selector': selector
        }
    
    def _mutual_info_selection(self, X, y, columns, n_features):
        """Selección por Información Mutua"""
        selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        scores_df = pd.DataFrame({
            'Feature': columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        selected_features = scores_df.head(n_features)['Feature'].tolist()
        
        return {
            'scores': scores_df,
            'selected_features': selected_features,
            'selector': selector
        }
    
    def _random_forest_selection(self, X, y, columns, n_features):
        """Selección por importancia de Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        scores_df = pd.DataFrame({
            'Feature': columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        selected_features = scores_df.head(n_features)['Feature'].tolist()
        
        return {
            'scores': scores_df,
            'selected_features': selected_features,
            'model': rf
        }
    
    def _pca_selection(self, X, y, columns, variance_threshold=0.95):
        """Análisis de Componentes Principales"""
        pca = PCA(n_components=variance_threshold, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Componentes y varianza explicada
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=columns
        )
        
        return {
            'pca': pca,
            'n_components': pca.n_components_,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': components_df,
            'X_transformed': X_pca
        }
    
    def get_feature_importance_summary(self, results):
        """Obtiene resumen de importancia de features"""
        summary = {}
        
        for method, data in results.items():
            if method != "PCA" and 'scores' in data:
                summary[method] = data['scores'].head(20)
        
        return summary