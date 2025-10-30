"""
Módulo para carga y validación de datos
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Clase para cargar y validar datos"""
    
    def __init__(self):
        self.df = None
    
    def load_data(self, filepath):
        """
        Carga datos desde archivo CSV
        
        Args:
            filepath: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            # Verificar que el archivo existe
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
            
            # Cargar datos
            df = pd.read_csv(filepath)
            
            # Validaciones básicas
            if df.empty:
                raise ValueError("El archivo está vacío")
            
            # Verificar columnas esperadas
            required_cols = ['Time', 'Amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columnas faltantes: {missing_cols}")
            
            self.df = df
            return df
            
        except Exception as e:
            raise Exception(f"Error al cargar datos: {str(e)}")
    
    def get_info(self):
        """Retorna información básica del dataset"""
        if self.df is None:
            return None
        
        return {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'null_counts': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
    
    def validate_data(self):
        """Valida la integridad de los datos"""
        if self.df is None:
            return False, "No hay datos cargados"
        
        issues = []
        
        # Verificar valores nulos
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
        
        # Verificar duplicados
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Filas duplicadas: {duplicates}")
        
        # Verificar tipos de datos
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(self.df.columns) - 1:
            issues.append("Algunas columnas no son numéricas")
        
        if issues:
            return False, issues
        else:
            return True, "Datos válidos"