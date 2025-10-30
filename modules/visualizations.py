"""
Módulo para visualizaciones
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Visualizer:
    """Clase para crear visualizaciones"""
    
    def __init__(self):
        self.colors = {
            'normal': '#2ecc71',
            'fraud': '#e74c3c',
            'primary': '#3498db',
            'secondary': '#9b59b6'
        }
    
    def plot_class_distribution(self, y):
        """Visualiza la distribución de clases"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        class_counts = y.value_counts()
        class_percentages = y.value_counts(normalize=True) * 100
        
        # Gráfico de barras
        axes[0].bar(['Normal', 'Fraude'], class_counts.values, 
                   color=[self.colors['normal'], self.colors['fraud']])
        axes[0].set_ylabel('Cantidad', fontsize=12)
        axes[0].set_title('Distribución de Clases', fontsize=14, fontweight='bold')
        axes[0].set_yscale('log')
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Gráfico de pie
        axes[1].pie(class_counts.values, labels=['Normal', 'Fraude'], 
                   autopct='%1.4f%%',
                   colors=[self.colors['normal'], self.colors['fraud']], 
                   startangle=90)
        axes[1].set_title('Proporción de Clases', fontsize=14, fontweight='bold')
        
        # Porcentaje visual
        axes[2].barh(['Fraude', 'Normal'], class_percentages.values, 
                    color=[self.colors['fraud'], self.colors['normal']])
        axes[2].set_xlabel('Porcentaje (%)', fontsize=12)
        axes[2].set_title('Porcentaje de Clases', fontsize=14, fontweight='bold')
        for i, v in enumerate(class_percentages.values):
            axes[2].text(v, i, f'{v:.4f}%', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_amount_time_analysis(self, df):
        """Visualiza análisis de Amount y Time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        if 'Amount' in df.columns and 'Class' in df.columns:
            # Amount por clase
            axes[0, 0].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, 
                           label='Normal', color=self.colors['normal'])
            axes[0, 0].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, 
                           label='Fraude', color=self.colors['fraud'])
            axes[0, 0].set_xlabel('Amount')
            axes[0, 0].set_ylabel('Frecuencia')
            axes[0, 0].set_title('Distribución de Amount por Clase')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
            
            # Amount (log scale)
            axes[0, 1].hist(np.log1p(df[df['Class']==0]['Amount']), bins=50, 
                           alpha=0.7, label='Normal', color=self.colors['normal'])
            axes[0, 1].hist(np.log1p(df[df['Class']==1]['Amount']), bins=50, 
                           alpha=0.7, label='Fraude', color=self.colors['fraud'])
            axes[0, 1].set_xlabel('log(Amount + 1)')
            axes[0, 1].set_ylabel('Frecuencia')
            axes[0, 1].set_title('Distribución de log(Amount) por Clase')
            axes[0, 1].legend()
        
        if 'Time' in df.columns and 'Class' in df.columns:
            # Time por clase
            axes[1, 0].hist(df[df['Class']==0]['Time'], bins=50, alpha=0.7, 
                           label='Normal', color=self.colors['normal'])
            axes[1, 0].hist(df[df['Class']==1]['Time'], bins=50, alpha=0.7, 
                           label='Fraude', color=self.colors['fraud'])
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].set_title('Distribución de Time por Clase')
            axes[1, 0].legend()
            
            # Boxplot de Amount
            axes[1, 1].boxplot([df[df['Class']==0]['Amount'],
                               df[df['Class']==1]['Amount']], 
                              labels=['Normal', 'Fraude'])
            axes[1, 1].set_ylabel('Amount')
            axes[1, 1].set_title('Boxplot de Amount por Clase')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df):
        """Visualiza matriz de correlación"""
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Matriz de Correlación', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, df, features):
        """Visualiza distribuciones de features"""
        n_cols = 3
        n_rows = (len(features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for idx, col in enumerate(features):
            if idx < len(axes) and col in df.columns:
                axes[idx].hist(df[df['Class']==0][col], bins=50, alpha=0.7, 
                              label='Normal', color=self.colors['normal'], density=True)
                axes[idx].hist(df[df['Class']==1][col], bins=50, alpha=0.7, 
                              label='Fraude', color=self.colors['fraud'], density=True)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Densidad')
                axes[idx].set_title(f'Distribución de {col}')
                axes[idx].legend()
        
        for idx in range(len(features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, scores_df):
        """Visualiza importancia de features"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        feature_col = scores_df.columns[0]
        score_col = scores_df.columns[1]
        
        top_20 = scores_df.head(20)
        ax.barh(range(20), top_20[score_col][::-1], color=self.colors['primary'])
        ax.set_yticks(range(20))
        ax.set_yticklabels(top_20[feature_col][::-1])
        ax.set_xlabel(score_col)
        ax.set_title(f'Top 20 Features - {score_col}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, results_df):
        """Visualiza comparación de modelos"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Agrupar por modelo y técnica
            pivot = results_df.pivot_table(
                values=metric, 
                index='Técnica', 
                columns='Modelo'
            )
            
            pivot.plot(kind='bar', ax=ax, rot=45)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} - Comparación de Modelos', 
                        fontsize=12, fontweight='bold')
            ax.legend(title='Modelo')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Visualiza matriz de confusión"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Matriz absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Normal', 'Fraude'],
                   yticklabels=['Normal', 'Fraude'],
                   cbar_kws={'label': 'Count'})
        axes[0].set_ylabel('Valor Real', fontsize=12)
        axes[0].set_xlabel('Predicción', fontsize=12)
        axes[0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        
        # Matriz normalizada
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                   xticklabels=['Normal', 'Fraude'],
                   yticklabels=['Normal', 'Fraude'],
                   cbar_kws={'label': 'Proporción'})
        axes[1].set_ylabel('Valor Real', fontsize=12)
        axes[1].set_xlabel('Predicción', fontsize=12)
        axes[1].set_title('Matriz de Confusión Normalizada', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true, y_proba):
        """Visualiza curva ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=3,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_proba):
        """Visualiza curva Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        from sklearn.metrics import average_precision_score
        ap_score = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, color=self.colors['normal'], lw=3,
                label=f'PR curve (AP = {ap_score:.4f})')
        baseline = (y_true == 1).sum() / len(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', label='Baseline')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig