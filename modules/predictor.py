"""
Módulo para hacer predicciones
"""

import pandas as pd
import numpy as np

class Predictor:
    """Clase para hacer predicciones con modelos entrenados"""
    
    def __init__(self):
        self.predictions = None
    
    def predict(self, model, df_test, selected_features, scaler, threshold=0.5):
        """
        Hace predicciones en conjunto de test
        
        Args:
            model: Modelo entrenado
            df_test: DataFrame de test con feature engineering aplicado
            selected_features: Lista de features seleccionadas
            scaler: Scaler entrenado
            threshold: Umbral de clasificación
            
        Returns:
            Diccionario con predicciones y probabilidades
        """
        # Verificar features faltantes
        missing_features = [f for f in selected_features if f not in df_test.columns]
        
        # Crear features faltantes con valor 0
        for feat in missing_features:
            df_test[feat] = 0
        
        # Seleccionar features
        X_test = df_test[selected_features]
        
        # Normalizar
        X_test_scaled = scaler.transform(X_test)
        
        # Hacer predicciones
        try:
            # Intentar predict_proba (sklearn models)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        except AttributeError:
            # Keras model
            y_proba = model.predict(X_test_scaled, verbose=0).flatten()
        
        # Aplicar threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        self.predictions = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'threshold': threshold
        }
        
        return self.predictions
    
    def predict_with_confidence(self, model, df_test, selected_features, scaler):
        """
        Hace predicciones con niveles de confianza
        
        Returns:
            DataFrame con predicciones y niveles de confianza
        """
        predictions = self.predict(model, df_test, selected_features, scaler)
        
        # Clasificar por nivel de confianza
        confidence_levels = []
        for prob in predictions['y_proba']:
            if prob >= 0.9:
                confidence_levels.append('Muy Alto')
            elif prob >= 0.7:
                confidence_levels.append('Alto')
            elif prob >= 0.5:
                confidence_levels.append('Medio')
            elif prob >= 0.3:
                confidence_levels.append('Bajo')
            else:
                confidence_levels.append('Muy Bajo')
        
        results_df = pd.DataFrame({
            'Prediccion': predictions['y_pred'],
            'Probabilidad': predictions['y_proba'],
            'Nivel_Confianza': confidence_levels
        })
        
        return results_df
    
    def get_high_risk_transactions(self, top_n=100):
        """Obtiene las transacciones de mayor riesgo"""
        if self.predictions is None:
            return None
        
        y_proba = self.predictions['y_proba']
        top_indices = np.argsort(y_proba)[-top_n:][::-1]
        
        return {
            'indices': top_indices,
            'probabilities': y_proba[top_indices]
        }
    
    def generate_prediction_summary(self):
        """Genera resumen de predicciones"""
        if self.predictions is None:
            return None
        
        y_pred = self.predictions['y_pred']
        y_proba = self.predictions['y_proba']
        
        return {
            'total': len(y_pred),
            'predicted_fraud': (y_pred == 1).sum(),
            'predicted_normal': (y_pred == 0).sum(),
            'fraud_percentage': (y_pred == 1).sum() / len(y_pred) * 100,
            'prob_min': y_proba.min(),
            'prob_max': y_proba.max(),
            'prob_mean': y_proba.mean(),
            'prob_median': np.median(y_proba),
            'prob_std': y_proba.std()
        }