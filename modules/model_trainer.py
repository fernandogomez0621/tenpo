"""
Módulo para entrenamiento de modelos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, accuracy_score, average_precision_score)
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# TensorFlow/Keras
try:
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

class ModelTrainer:
    """Clase para entrenamiento de modelos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_info = {}
    
    def train_models(self, X, y, model_types, balancing_techniques, test_size=0.3):
        """
        Entrena múltiples modelos con diferentes técnicas de balanceo
        
        Args:
            X: Features
            y: Target
            model_types: Lista de tipos de modelos
            balancing_techniques: Lista de técnicas de balanceo
            test_size: Tamaño del conjunto de test
            
        Returns:
            Diccionario con resultados
        """
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Crear datasets balanceados
        balanced_datasets = self._create_balanced_datasets(
            X_train_scaled, y_train, balancing_techniques
        )
        
        # Entrenar modelos
        all_results = []
        best_f1 = 0
        
        for model_type in model_types:
            for technique, (X_bal, y_bal) in balanced_datasets.items():
                print(f"Entrenando {model_type} con {technique}...")
                
                result = self._train_single_model(
                    model_type, technique, 
                    X_bal, y_bal, X_test_scaled, y_test
                )
                
                all_results.append(result)
                
                # Actualizar mejor modelo
                if result['F1-Score'] > best_f1:
                    best_f1 = result['F1-Score']
                    self.best_model = result['model']
                    self.best_model_info = {
                        'model_type': model_type,
                        'technique': technique,
                        'metrics': {k: v for k, v in result.items() if k != 'model'},
                        'y_test': y_test,
                        'y_pred': result['y_pred'],
                        'y_proba': result['y_proba']
                    }
        
        return {
            'all_results': all_results,
            'best_model': self.best_model_info,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
    
    def _create_balanced_datasets(self, X, y, techniques):
        """Crea datasets balanceados"""
        datasets = {}
        
        if "Sin Balanceo" in techniques:
            datasets["Sin Balanceo"] = (X, y)
        
        if "Undersampling" in techniques:
            rus = RandomUnderSampler(random_state=42)
            X_rus, y_rus = rus.fit_resample(X, y)
            datasets["Undersampling"] = (X_rus, y_rus)
        
        if "SMOTE" in techniques:
            smote = SMOTE(random_state=42)
            X_smote, y_smote = smote.fit_resample(X, y)
            datasets["SMOTE"] = (X_smote, y_smote)
        
        if "ADASYN" in techniques:
            adasyn = ADASYN(random_state=42)
            X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
            datasets["ADASYN"] = (X_adasyn, y_adasyn)
        
        if "Class Weights" in techniques:
            datasets["Class Weights"] = (X, y)
        
        return datasets
    
    def _train_single_model(self, model_type, technique, X_train, y_train, X_test, y_test):
        """Entrena un modelo individual"""
        
        if model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        elif model_type == "XGBoost":
            if technique == "Class Weights":
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='auc'
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='auc'
                )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        elif model_type == "Red Neuronal" and KERAS_AVAILABLE:
            model = self._create_neural_network(X_train.shape[1])
            
            if technique == "Class Weights":
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                class_weight = {0: 1.0, 1: scale_pos_weight}
            else:
                class_weight = None
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                                      restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                         patience=5, min_lr=1e-7)
            
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=256,
                validation_split=0.2,
                class_weight=class_weight,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            y_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_proba > 0.5).astype(int)
        else:
            # Fallback a Random Forest si Red Neuronal no está disponible
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'Modelo': model_type,
            'Técnica': technique,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'AP': average_precision_score(y_test, y_proba),
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def _create_neural_network(self, input_dim):
        """Crea arquitectura de red neuronal"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model