"""
Módulo para análisis exploratorio de datos
"""

import pandas as pd
import numpy as np
from scipy import stats

class ExploratoryDataAnalysis:
    """Clase para análisis exploratorio de datos"""
    
    def __init__(self, df):
        self.df = df
    
    def analyze_class_distribution(self):
        """Analiza la distribución de la variable objetivo"""
        if 'Class' not in self.df.columns:
            return None
        
        class_counts = self.df['Class'].value_counts()
        class_pcts = self.df['Class'].value_counts(normalize=True) * 100
        
        return {
            'normal_count': class_counts.get(0, 0),
            'fraud_count': class_counts.get(1, 0),
            'normal_pct': class_pcts.get(0, 0),
            'fraud_pct': class_pcts.get(1, 0),
            'imbalance_ratio': class_counts.get(0, 0) / class_counts.get(1, 1)
        }
    
    def analyze_numerical_features(self):
        """Analiza características numéricas"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats_dict = {}
        for col in numeric_cols:
            if col != 'Class':
                stats_dict[col] = {
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'q25': self.df[col].quantile(0.25),
                    'q75': self.df[col].quantile(0.75),
                    'skew': self.df[col].skew(),
                    'kurtosis': self.df[col].kurtosis()
                }
        
        return stats_dict
    
    def analyze_by_class(self, feature):
        """Analiza una característica por clase"""
        if 'Class' not in self.df.columns or feature not in self.df.columns:
            return None
        
        normal_stats = self.df[self.df['Class'] == 0][feature].describe()
        fraud_stats = self.df[self.df['Class'] == 1][feature].describe()
        
        return {
            'normal': normal_stats.to_dict(),
            'fraud': fraud_stats.to_dict()
        }
    
    def detect_outliers(self, feature, method='iqr'):
        """Detecta outliers en una característica"""
        if feature not in self.df.columns:
            return None
        
        data = self.df[feature]
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
        else:
            return None
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'values': outliers.tolist()
        }
    
    def correlation_analysis(self, target='Class'):
        """Análisis de correlación con variable objetivo"""
        if target not in self.df.columns:
            return None
        
        correlations = self.df.corr()[target].abs().sort_values(ascending=False)
        
        return correlations.to_dict()
    
    def get_summary_statistics(self):
        """Retorna estadísticas resumen del dataset"""
        return {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'describe': self.df.describe().to_dict(),
            'null_counts': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }