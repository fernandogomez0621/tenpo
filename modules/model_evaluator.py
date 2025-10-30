"""
Módulo para evaluación de modelos
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

class ModelEvaluator:
    """Clase para evaluación de modelos"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true, y_pred, y_proba):
        """
        Evalúa un modelo con múltiples métricas
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_proba: Probabilidades
            
        Returns:
            Diccionario con métricas
        """
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Métricas derivadas
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Curvas
        fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_true, y_proba)
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_proba)
        
        return {
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'fpr': fpr,
            'fnr': fnr,
            'roc_curve': (fpr_curve, tpr_curve, thresholds_roc),
            'pr_curve': (precision_curve, recall_curve, thresholds_pr),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'ap_score': average_precision_score(y_true, y_proba)
        }
    
    def get_classification_report(self, y_true, y_pred):
        """Genera reporte de clasificación"""
        report = classification_report(
            y_true, y_pred, 
            target_names=['Normal', 'Fraude'],
            output_dict=True
        )
        return report
    
    def find_optimal_threshold(self, y_true, y_proba, metric='f1'):
        """
        Encuentra el threshold óptimo
        
        Args:
            y_true: Valores reales
            y_proba: Probabilidades
            metric: Métrica a optimizar ('f1', 'precision', 'recall')
            
        Returns:
            Threshold óptimo y métricas
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.linspace(0.1, 0.9, 81)
        scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred_thresh)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred_thresh)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred_thresh)
            else:
                score = f1_score(y_true, y_pred_thresh)
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_score': optimal_score,
            'thresholds': thresholds,
            'scores': scores
        }
    
    def compare_models(self, results_df):
        """
        Compara múltiples modelos
        
        Args:
            results_df: DataFrame con resultados de múltiples modelos
            
        Returns:
            Análisis comparativo
        """
        # Identificar mejor modelo por métrica
        best_by_metric = {}
        metrics = ['F1-Score', 'ROC-AUC', 'Precision', 'Recall', 'Accuracy']
        
        for metric in metrics:
            if metric in results_df.columns:
                best_idx = results_df[metric].idxmax()
                best_by_metric[metric] = {
                    'model': results_df.loc[best_idx, 'Modelo'],
                    'technique': results_df.loc[best_idx, 'Técnica'],
                    'score': results_df.loc[best_idx, metric]
                }
        
        # Estadísticas por modelo
        model_stats = results_df.groupby('Modelo')[metrics].agg(['mean', 'std', 'min', 'max'])
        
        return {
            'best_by_metric': best_by_metric,
            'model_statistics': model_stats,
            'overall_best': results_df.loc[results_df['F1-Score'].idxmax()]
        }
    
    def calculate_business_metrics(self, cm, fraud_cost=100, false_positive_cost=10):
        """
        Calcula métricas de negocio
        
        Args:
            cm: Matriz de confusión
            fraud_cost: Costo de un fraude no detectado
            false_positive_cost: Costo de un falso positivo
            
        Returns:
            Métricas de negocio
        """
        tn, fp, fn, tp = cm.ravel()
        
        # Costos totales
        fraud_loss = fn * fraud_cost
        false_alarm_cost = fp * false_positive_cost
        total_cost = fraud_loss + false_alarm_cost
        
        # Ahorros
        fraud_detected_savings = tp * fraud_cost
        net_savings = fraud_detected_savings - total_cost
        
        return {
            'fraud_loss': fraud_loss,
            'false_alarm_cost': false_alarm_cost,
            'total_cost': total_cost,
            'fraud_detected_savings': fraud_detected_savings,
            'net_savings': net_savings,
            'cost_per_transaction': total_cost / (tn + fp + fn + tp)
        }