"""
Módulo para ingeniería de características
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Clase para crear nuevas características"""
    
    def __init__(self):
        self.important_v = ['V14', 'V10', 'V12', 'V17', 'V11', 'V4', 'V3']
    
    def create_features(self, df):
        """
        Crea nuevas características a partir de las existentes
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame con nuevas características
        """
        df_fe = df.copy()
        
        # Transformaciones de Amount
        if 'Amount' in df_fe.columns:
            df_fe['Amount_log'] = np.log1p(df_fe['Amount'])
            df_fe['Amount_sqrt'] = np.sqrt(df_fe['Amount'])
            df_fe['Amount_squared'] = df_fe['Amount'] ** 2
            df_fe['Amount_cubed'] = df_fe['Amount'] ** 3
        
        # Transformaciones de Time
        if 'Time' in df_fe.columns:
            df_fe['Time_hour'] = (df_fe['Time'] / 3600) % 24
            df_fe['Time_day'] = (df_fe['Time'] / 86400).astype(int)
            df_fe['Time_sin'] = np.sin(2 * np.pi * df_fe['Time_hour'] / 24)
            df_fe['Time_cos'] = np.cos(2 * np.pi * df_fe['Time_hour'] / 24)
        
        # Interacciones entre V features importantes
        for i, v1 in enumerate(self.important_v):
            for v2 in self.important_v[i+1:]:
                if v1 in df_fe.columns and v2 in df_fe.columns:
                    df_fe[f'{v1}_{v2}_interaction'] = df_fe[v1] * df_fe[v2]
                    df_fe[f'{v1}_{v2}_ratio'] = df_fe[v1] / (df_fe[v2] + 1e-5)
        
        # Features estadísticas agregadas
        v_columns = [col for col in df_fe.columns if col.startswith('V') and len(col) <= 3]
        if len(v_columns) > 0:
            df_fe['V_mean'] = df_fe[v_columns].mean(axis=1)
            df_fe['V_std'] = df_fe[v_columns].std(axis=1)
            df_fe['V_min'] = df_fe[v_columns].min(axis=1)
            df_fe['V_max'] = df_fe[v_columns].max(axis=1)
            df_fe['V_median'] = df_fe[v_columns].median(axis=1)
            df_fe['V_skew'] = df_fe[v_columns].skew(axis=1)
            df_fe['V_kurtosis'] = df_fe[v_columns].kurtosis(axis=1)
        
        return df_fe
    
    def create_polynomial_features(self, df, features, degree=2):
        """Crea características polinomiales"""
        df_poly = df.copy()
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    df_poly[f'{feature}_pow{d}'] = df[feature] ** d
        
        return df_poly
    
    def create_interaction_features(self, df, feature_pairs):
        """Crea características de interacción"""
        df_inter = df.copy()
        
        for f1, f2 in feature_pairs:
            if f1 in df.columns and f2 in df.columns:
                df_inter[f'{f1}_{f2}_mult'] = df[f1] * df[f2]
                df_inter[f'{f1}_{f2}_div'] = df[f1] / (df[f2] + 1e-5)
                df_inter[f'{f1}_{f2}_add'] = df[f1] + df[f2]
                df_inter[f'{f1}_{f2}_sub'] = df[f1] - df[f2]
        
        return df_inter
    
    def create_binned_features(self, df, feature, n_bins=10):
        """Crea características discretizadas"""
        df_binned = df.copy()
        
        if feature in df.columns:
            df_binned[f'{feature}_binned'] = pd.qcut(
                df[feature], 
                q=n_bins, 
                labels=False, 
                duplicates='drop'
            )
        
        return df_binned