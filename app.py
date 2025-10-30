"""
APLICACIÓN DE DETECCIÓN DE FRAUDE EN TARJETAS DE CRÉDITO
Prueba Técnica - Científico de Datos - Tenpo
Octubre 2025 - Versión Optimizada con Caché
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import pickle
import os

# Configuración de página
st.set_page_config(
    page_title="Detección de Fraude - Tenpo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Importar módulos personalizados
from modules.data_loader import DataLoader
from modules.eda import ExploratoryDataAnalysis
from modules.feature_engineering import FeatureEngineer
from modules.feature_selection import FeatureSelector
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import ModelEvaluator
from modules.predictor import Predictor
from modules.visualizations import Visualizer

# Directorio para cache
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .cache-status {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CACHÉ Y PERSISTENCIA
# ============================================================================

def save_to_cache(data, filename):
    """Guarda datos en caché"""
    try:
        filepath = CACHE_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Error al guardar caché: {str(e)}")
        return False

def load_from_cache(filename):
    """Carga datos desde caché"""
    try:
        filepath = CACHE_DIR / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        st.error(f"Error al cargar caché: {str(e)}")
        return None

def check_cache_exists(filename):
    """Verifica si existe un archivo en caché"""
    return (CACHE_DIR / filename).exists()

def get_cache_info():
    """Obtiene información sobre archivos en caché"""
    cache_files = {
        'df_train.pkl': 'Datos de Entrenamiento',
        'df_test.pkl': 'Datos de Test',
        'df_fe.pkl': 'Feature Engineering',
        'feature_selection.pkl': 'Selección de Features',
        'training_results.pkl': 'Modelos Entrenados',
        'predictions.pkl': 'Predicciones'
    }
    
    info = {}
    for filename, description in cache_files.items():
        filepath = CACHE_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            info[description] = {
                'exists': True,
                'size': f"{size_mb:.2f} MB",
                'filename': filename
            }
        else:
            info[description] = {'exists': False}
    
    return info

def clear_cache():
    """Limpia todos los archivos de caché"""
    try:
        for file in CACHE_DIR.glob("*.pkl"):
            file.unlink()
        return True
    except Exception as e:
        st.error(f"Error al limpiar caché: {str(e)}")
        return False

# ============================================================================
# INICIALIZACIÓN AUTOMÁTICA
# ============================================================================

def auto_initialize():
    """Inicializa automáticamente los datos desde caché o archivos"""
    
    # Cargar datos de entrenamiento
    if 'df_train' not in st.session_state:
        df_train = load_from_cache('df_train.pkl')
        if df_train is not None:
            st.session_state.df_train = df_train
            st.session_state.data_loaded = True
        else:
            # Intentar cargar desde archivo CSV
            if Path("creditcard_train.csv").exists():
                try:
                    loader = DataLoader()
                    st.session_state.df_train = loader.load_data("creditcard_train.csv")
                    st.session_state.data_loaded = True
                    save_to_cache(st.session_state.df_train, 'df_train.pkl')
                except:
                    st.session_state.data_loaded = False
    
    # Cargar datos de test
    if 'df_test' not in st.session_state:
        df_test = load_from_cache('df_test.pkl')
        if df_test is not None:
            st.session_state.df_test = df_test
        else:
            if Path("creditcard_test.csv").exists():
                try:
                    loader = DataLoader()
                    st.session_state.df_test = loader.load_data("creditcard_test.csv")
                    save_to_cache(st.session_state.df_test, 'df_test.pkl')
                except:
                    pass
    
    # Cargar feature engineering
    if 'df_fe' not in st.session_state:
        df_fe = load_from_cache('df_fe.pkl')
        if df_fe is not None:
            st.session_state.df_fe = df_fe
    
    # Cargar feature selection
    if 'feature_selection_results' not in st.session_state:
        feature_selection = load_from_cache('feature_selection.pkl')
        if feature_selection is not None:
            st.session_state.feature_selection_results = feature_selection['results']
            st.session_state.selected_features = feature_selection['selected_features']
    
    # Cargar modelos entrenados
    if 'training_results' not in st.session_state:
        training_results = load_from_cache('training_results.pkl')
        if training_results is not None:
            st.session_state.training_results = training_results
            st.session_state.model_trained = True
            
            # Cargar trainer
            trainer = load_from_cache('trainer.pkl')
            if trainer is not None:
                st.session_state.trainer = trainer
    
    # Cargar predicciones
    if 'predictions' not in st.session_state:
        predictions = load_from_cache('predictions.pkl')
        if predictions is not None:
            st.session_state.predictions = predictions
            st.session_state.predictions_made = True
    
    # Estados por defecto
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación"""
    
    # Inicialización automática
    auto_initialize()
    
    # Header
    st.markdown('<div class="main-header">💳 Sistema de Detección de Fraude</div>', 
                unsafe_allow_html=True)
    st.markdown("### Prueba Técnica - Científico de Datos - Tenpo")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/250x80/1f77b4/ffffff?text=TENPO", 
                 use_container_width=True)
        st.markdown("## 📋 Navegación")
        
        page = st.radio(
            "Seleccione una sección:",
            [
                "🏠 Inicio",
                "📊 Análisis Exploratorio",
                "🔧 Ingeniería de Features",
                "🎯 Selección de Features",
                "🤖 Entrenamiento de Modelos",
                "📈 Evaluación de Modelos",
                "🔮 Predicciones",
                "📄 Reporte Final"
            ]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Gestión de Caché")
        
        # Mostrar estado del caché
        cache_info = get_cache_info()
        
        with st.expander("📦 Estado del Caché", expanded=False):
            for description, info in cache_info.items():
                if info['exists']:
                    st.markdown(f"✅ **{description}**: {info['size']}")
                else:
                    st.markdown(f"❌ **{description}**: No disponible")
        
        # Botones de gestión
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Recargar Datos", use_container_width=True):
                with st.spinner("Recargando datos..."):
                    try:
                        loader = DataLoader()
                        st.session_state.df_train = loader.load_data("creditcard_train.csv")
                        st.session_state.df_test = loader.load_data("creditcard_test.csv")
                        st.session_state.data_loaded = True
                        
                        save_to_cache(st.session_state.df_train, 'df_train.pkl')
                        save_to_cache(st.session_state.df_test, 'df_test.pkl')
                        
                        st.success("✅ Datos recargados!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Limpiar Caché", use_container_width=True):
                if clear_cache():
                    # Limpiar session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.success("✅ Caché limpiado!")
                    st.rerun()
        
        # Información del sistema
        st.markdown("---")
        st.markdown("### 📊 Estado del Sistema")
        
        status_items = [
            ("Datos", st.session_state.data_loaded),
            ("Feature Eng.", 'df_fe' in st.session_state),
            ("Feat. Selection", 'selected_features' in st.session_state),
            ("Modelo", st.session_state.model_trained),
            ("Predicciones", st.session_state.predictions_made)
        ]
        
        for item, status in status_items:
            icon = "✅" if status else "⏳"
            st.markdown(f"{icon} {item}")
    
    # Contenido principal según la página seleccionada
    if page == "🏠 Inicio":
        show_home_page()
    elif page == "📊 Análisis Exploratorio":
        show_eda_page()
    elif page == "🔧 Ingeniería de Features":
        show_feature_engineering_page()
    elif page == "🎯 Selección de Features":
        show_feature_selection_page()
    elif page == "🤖 Entrenamiento de Modelos":
        show_model_training_page()
    elif page == "📈 Evaluación de Modelos":
        show_model_evaluation_page()
    elif page == "🔮 Predicciones":
        show_predictions_page()
    elif page == "📄 Reporte Final":
        show_report_page()

# ============================================================================
# PÁGINAS (mantener las mismas pero con modificaciones para caché)
# ============================================================================

def show_home_page():
    """Página de inicio"""
    st.markdown("## 🏠 Bienvenido al Sistema de Detección de Fraude")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📝 Descripción del Proyecto
        
        Este sistema permite desarrollar y evaluar modelos de Machine Learning para 
        detectar transacciones fraudulentas en tarjetas de crédito, minimizando los 
        falsos positivos que generan cancelaciones del servicio.
        
        ### 🎯 Objetivos
        
        1. **Análisis Exploratorio**: Comprender el comportamiento de los datos
        2. **Ingeniería de Features**: Crear características relevantes
        3. **Selección de Features**: Identificar las variables más importantes
        4. **Modelado**: Entrenar y comparar diferentes algoritmos
        5. **Evaluación**: Analizar métricas y desempeño
        6. **Predicción**: Generar resultados en conjunto de test
        
        ### 📊 Metodología
        
        - **Feature Engineering**: Transformaciones, interacciones y agregaciones
        - **Selección de Features**: F-Score, Información Mutua, Random Forest, PCA
        - **Técnicas de Balanceo**: Undersampling, SMOTE, ADASYN, Class Weights
        - **Modelos**: Random Forest, XGBoost, Redes Neuronales
        - **Evaluación**: F1-Score, ROC-AUC, Precision-Recall
        
        ### ⚡ Características de Optimización
        
        - **Caché Automático**: Los datos y modelos se guardan automáticamente
        - **Carga Rápida**: No necesita reprocesar en cada sesión
        - **Persistencia**: Los resultados se mantienen entre sesiones
        """)
    
    with col2:
        st.markdown("### 📈 Estado del Proyecto")
        
        if st.session_state.data_loaded:
            st.success("✅ Datos cargados")
            if 'df_train' in st.session_state:
                st.info(f"📊 Train: {len(st.session_state.df_train):,} registros")
            if 'df_test' in st.session_state:
                st.info(f"📊 Test: {len(st.session_state.df_test):,} registros")
        else:
            st.warning("⚠️ Datos no cargados")
            st.info("🔄 Coloque los archivos CSV en el directorio")
        
        if 'df_fe' in st.session_state:
            st.success("✅ Feature Engineering aplicado")
        else:
            st.warning("⏳ Feature Engineering pendiente")
        
        if 'selected_features' in st.session_state:
            st.success("✅ Features seleccionadas")
            st.info(f"🎯 {len(st.session_state.selected_features)} features")
        else:
            st.warning("⏳ Selección pendiente")
        
        if st.session_state.model_trained:
            st.success("✅ Modelo entrenado")
        else:
            st.warning("⏳ Modelo no entrenado")
        
        if st.session_state.predictions_made:
            st.success("✅ Predicciones realizadas")
        else:
            st.warning("⏳ Predicciones pendientes")
    
    st.markdown("---")
    
    # Información de caché
    cache_info = get_cache_info()
    available_cache = sum(1 for info in cache_info.values() if info['exists'])
    
    if available_cache > 0:
        st.info(f"💾 **{available_cache}** archivos disponibles en caché - El sistema cargará automáticamente los datos guardados")

def show_feature_engineering_page():
    """Página de ingeniería de características"""
    st.markdown("## 🔧 Ingeniería de Características")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Por favor, asegúrese de que los datos estén cargados")
        return
    
    df_train = st.session_state.df_train
    engineer = FeatureEngineer()
    
    # Verificar si ya existe feature engineering
    if 'df_fe' in st.session_state:
        st.success("✅ Feature Engineering ya aplicado (cargado desde caché)")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Features Originales", df_train.shape[1] - 1)
        col2.metric("Features Creadas", st.session_state.df_fe.shape[1] - df_train.shape[1])
        col3.metric("Total Features", st.session_state.df_fe.shape[1] - 1)
        
        st.markdown("#### Muestra de nuevas features")
        new_cols = [col for col in st.session_state.df_fe.columns if col not in df_train.columns]
        st.dataframe(st.session_state.df_fe[new_cols[:15]].head(10), use_container_width=True)
        
        if st.button("🔄 Regenerar Feature Engineering", use_container_width=True):
            with st.spinner("Regenerando features..."):
                df_fe = engineer.create_features(df_train)
                st.session_state.df_fe = df_fe
                save_to_cache(df_fe, 'df_fe.pkl')
                st.success("✅ Feature Engineering regenerado!")
                st.rerun()
        
        return
    
    st.markdown("""
    ### 📝 Transformaciones Aplicadas
    
    Se aplicarán las siguientes transformaciones para crear nuevas características:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Transformaciones de Amount:**
        - Logaritmo: `log(Amount + 1)`
        - Raíz cuadrada: `sqrt(Amount)`
        - Cuadrado: `Amount²`
        - Cubo: `Amount³`
        
        **2. Transformaciones de Time:**
        - Hora del día: `(Time / 3600) % 24`
        - Día: `Time / 86400`
        - Transformación cíclica (sin, cos)
        """)
    
    with col2:
        st.markdown("""
        **3. Interacciones:**
        - Productos entre V-features importantes
        - Ratios entre V-features
        
        **4. Agregaciones estadísticas:**
        - Media, desviación, min, max
        - Mediana, asimetría, curtosis
        """)
    
    if st.button("🚀 Aplicar Feature Engineering", use_container_width=True):
        with st.spinner("Aplicando transformaciones..."):
            df_fe = engineer.create_features(df_train)
            st.session_state.df_fe = df_fe
            
            # Guardar en caché
            save_to_cache(df_fe, 'df_fe.pkl')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Features Originales", df_train.shape[1] - 1)
            col2.metric("Features Creadas", df_fe.shape[1] - df_train.shape[1])
            col3.metric("Total Features", df_fe.shape[1] - 1)
            
            st.success("✅ Feature Engineering completado y guardado en caché!")
            
            st.markdown("#### Muestra de nuevas features")
            new_cols = [col for col in df_fe.columns if col not in df_train.columns]
            st.dataframe(df_fe[new_cols[:15]].head(10), use_container_width=True)

def show_feature_selection_page():
    """Página de selección de características"""
    st.markdown("## 🎯 Selección de Características")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Por favor, cargue los datos primero")
        return
    
    if 'df_fe' not in st.session_state:
        st.warning("⚠️ Por favor, aplique Feature Engineering primero")
        return
    
    df_fe = st.session_state.df_fe
    selector = FeatureSelector()
    viz = Visualizer()
    
    # Verificar si ya existe selección
    if 'feature_selection_results' in st.session_state:
        st.success("✅ Selección de Features ya realizada (cargado desde caché)")
        
        results = st.session_state.feature_selection_results
        methods = list(results.keys())
        
        # Mostrar resultados por método
        tabs = st.tabs([f"📊 {method}" for method in methods])
        
        for i, method in enumerate(methods):
            with tabs[i]:
                if method in results:
                    st.markdown(f"### Top 20 Features - {method}")
                    st.dataframe(
                        results[method]['scores'].head(20),
                        use_container_width=True
                    )
                    
                    if method != "PCA":
                        st.markdown("#### Importancia de Features")
                        fig = viz.plot_feature_importance(
                            results[method]['scores'].head(20)
                        )
                        st.pyplot(fig)
        
        # Features seleccionadas finales
        st.markdown("---")
        st.markdown("### ✅ Features Seleccionadas")
        
        col1, col2 = st.columns(2)
        col1.metric("Total Features Seleccionadas", len(st.session_state.selected_features))
        X = df_fe.drop('Class', axis=1)
        col2.metric("Reducción", 
                   f"{(1 - len(st.session_state.selected_features)/X.shape[1])*100:.1f}%")
        
        if st.button("🔄 Regenerar Selección", use_container_width=True):
            # Limpiar para permitir regeneración
            del st.session_state.feature_selection_results
            del st.session_state.selected_features
            st.rerun()
        
        return
    
    st.markdown("""
    ### 📊 Metodologías de Selección
    
    Se aplicarán diferentes métodos para identificar las características más relevantes:
    """)
    
    methods = st.multiselect(
        "Seleccione los métodos a aplicar:",
        ["F-Score", "Información Mutua", "Random Forest", "PCA"],
        default=["F-Score", "Información Mutua", "Random Forest"]
    )
    
    n_features = st.slider("Número de features a seleccionar por método:", 20, 100, 40)
    
    if st.button("🎯 Ejecutar Selección de Features", use_container_width=True):
        with st.spinner("Seleccionando características..."):
            X = df_fe.drop('Class', axis=1)
            y = df_fe['Class']
            
            results = selector.select_features(X, y, methods, n_features)
            st.session_state.feature_selection_results = results
            
            # Mostrar resultados por método
            tabs = st.tabs([f"📊 {method}" for method in methods])
            
            for i, method in enumerate(methods):
                with tabs[i]:
                    if method in results:
                        st.markdown(f"### Top 20 Features - {method}")
                        st.dataframe(
                            results[method]['scores'].head(20),
                            use_container_width=True
                        )
                        
                        if method != "PCA":
                            st.markdown("#### Importancia de Features")
                            fig = viz.plot_feature_importance(
                                results[method]['scores'].head(20)
                            )
                            st.pyplot(fig)
            
            # Features seleccionadas finales
            st.markdown("---")
            st.markdown("### ✅ Features Seleccionadas (Unión de todos los métodos)")
            
            all_features = set()
            for method in methods:
                if method in results and method != "PCA":
                    all_features.update(results[method]['selected_features'])
            
            st.session_state.selected_features = list(all_features)
            
            # Guardar en caché
            cache_data = {
                'results': results,
                'selected_features': list(all_features)
            }
            save_to_cache(cache_data, 'feature_selection.pkl')
            
            col1, col2 = st.columns(2)
            col1.metric("Total Features Seleccionadas", len(all_features))
            col2.metric("Reducción", 
                       f"{(1 - len(all_features)/X.shape[1])*100:.1f}%")
            
            st.success("✅ Selección de características completada y guardada en caché!")

def show_model_training_page():
    """Página de entrenamiento de modelos"""
    st.markdown("## 🤖 Entrenamiento de Modelos")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Por favor, cargue los datos primero")
        return
    
    if 'selected_features' not in st.session_state:
        st.warning("⚠️ Por favor, complete la selección de features primero")
        return
    
    # Verificar si ya hay modelo entrenado
    if st.session_state.model_trained and 'training_results' in st.session_state:
        st.success("✅ Modelos ya entrenados (cargado desde caché)")
        
        results = st.session_state.training_results
        results_df = pd.DataFrame(results['all_results'])
        
        display_cols = [col for col in results_df.columns 
                       if col not in ['model', 'y_pred', 'y_proba']]
        st.dataframe(results_df[display_cols], use_container_width=True)
        
        # Mejor modelo
        best_idx = results_df['F1-Score'].idxmax()
        best_model = results_df.loc[best_idx]
        
        st.markdown("### 🏆 Mejor Modelo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modelo", best_model['Modelo'])
        col2.metric("Técnica", best_model['Técnica'])
        col3.metric("F1-Score", f"{best_model['F1-Score']:.4f}")
        col4.metric("ROC-AUC", f"{best_model['ROC-AUC']:.4f}")
        
        if st.button("🔄 Reentrenar Modelos", use_container_width=True):
            # Limpiar para permitir reentrenamiento
            del st.session_state.training_results
            del st.session_state.trainer
            st.session_state.model_trained = False
            st.rerun()
        
        return
    
    trainer = ModelTrainer()
    
    st.markdown("### ⚙️ Configuración de Entrenamiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Modelos a entrenar")
        models_to_train = st.multiselect(
            "Seleccione los modelos:",
            ["Random Forest", "XGBoost", "Red Neuronal"],
            default=["Random Forest", "XGBoost"]
        )
    
    with col2:
        st.markdown("#### Técnicas de balanceo")
        balancing_techniques = st.multiselect(
            "Seleccione las técnicas:",
            ["Sin Balanceo", "Undersampling", "SMOTE", "ADASYN", "Class Weights"],
            default=["SMOTE", "Class Weights"]
        )
    
    test_size = st.slider("Tamaño del conjunto de test (%):", 10, 40, 30) / 100
    
    if st.button("🚀 Entrenar Modelos", use_container_width=True):
        with st.spinner("Entrenando modelos... Esto puede tomar varios minutos"):
            df_fe = st.session_state.df_fe
            selected_features = st.session_state.selected_features
            
            X = df_fe[selected_features]
            y = df_fe['Class']
            
            results = trainer.train_models(
                X, y, 
                models_to_train, 
                balancing_techniques,
                test_size
            )
            
            st.session_state.training_results = results
            st.session_state.trainer = trainer
            st.session_state.model_trained = True
            
            # Guardar en caché
            save_to_cache(results, 'training_results.pkl')
            save_to_cache(trainer, 'trainer.pkl')
            
            st.success("✅ Entrenamiento completado y guardado en caché!")
            
            # Mostrar resumen
            st.markdown("### 📊 Resumen de Resultados")
            
            results_df = pd.DataFrame(results['all_results'])
            display_cols = [col for col in results_df.columns 
                           if col not in ['model', 'y_pred', 'y_proba']]
            st.dataframe(results_df[display_cols], use_container_width=True)
            
            # Mejor modelo
            best_idx = results_df['F1-Score'].idxmax()
            best_model = results_df.loc[best_idx]
            
            st.markdown("### 🏆 Mejor Modelo")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modelo", best_model['Modelo'])
            col2.metric("Técnica", best_model['Técnica'])
            col3.metric("F1-Score", f"{best_model['F1-Score']:.4f}")
            col4.metric("ROC-AUC", f"{best_model['ROC-AUC']:.4f}")

def show_eda_page():
    """Página de análisis exploratorio"""
    st.markdown("## 📊 Análisis Exploratorio de Datos")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Por favor, asegúrese de que los datos estén cargados")
        return
    
    df_train = st.session_state.df_train
    eda = ExploratoryDataAnalysis(df_train)
    viz = Visualizer()
    
    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Info Básica", 
        "⚖️ Balance de Clases", 
        "📈 Distribuciones",
        "🔗 Correlaciones"
    ])
    
    with tab1:
        st.markdown("### 📋 Información del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Registros", f"{len(df_train):,}")
        col2.metric("Features", f"{df_train.shape[1]-1}")
        col3.metric("Valores Nulos", f"{df_train.isnull().sum().sum()}")
        col4.metric("Duplicados", f"{df_train.duplicated().sum()}")
        
        st.markdown("#### Primeros registros")
        st.dataframe(df_train.head(10), use_container_width=True)
        
        st.markdown("#### Estadísticas descriptivas")
        st.dataframe(df_train.describe(), use_container_width=True)
    
    with tab2:
        st.markdown("### ⚖️ Análisis de Balance de Clases")
        
        class_info = eda.analyze_class_distribution()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Transacciones Normales", 
                   f"{class_info['normal_count']:,}",
                   f"{class_info['normal_pct']:.4f}%")
        col2.metric("Transacciones Fraudulentas", 
                   f"{class_info['fraud_count']:,}",
                   f"{class_info['fraud_pct']:.4f}%")
        col3.metric("Ratio de Desbalance", 
                   f"1:{class_info['imbalance_ratio']:.0f}",
                   delta="Altamente desbalanceado", delta_color="off")
        
        st.markdown("#### Visualización del desbalance")
        fig = viz.plot_class_distribution(df_train['Class'])
        st.pyplot(fig)
    
    with tab3:
        st.markdown("### 📈 Distribuciones de Variables")
        
        # Amount y Time
        st.markdown("#### Análisis de Amount y Time")
        fig = viz.plot_amount_time_analysis(df_train)
        st.pyplot(fig)
        
        # Top V features
        st.markdown("#### Top Features por Correlación")
        correlation = df_train.corr()['Class'].abs().sort_values(ascending=False)
        top_features = correlation.head(9).index.tolist()[1:]  # Excluir Class
        
        fig = viz.plot_feature_distributions(df_train, top_features)
        st.pyplot(fig)
    
    with tab4:
        st.markdown("### 🔗 Análisis de Correlaciones")
        
        correlation = df_train.corr()['Class'].abs().sort_values(ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Top 15 Correlaciones")
            st.dataframe(
                correlation.head(16).to_frame().rename(columns={'Class': 'Correlación'}),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Matriz de Correlación")
            top_features = correlation.head(16).index.tolist()
            fig = viz.plot_correlation_matrix(df_train[top_features])
            st.pyplot(fig)

def show_model_evaluation_page():
    """Página de evaluación de modelos"""
    st.markdown("## 📈 Evaluación de Modelos")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Por favor, entrene los modelos primero")
        return
    
    results = st.session_state.training_results
    evaluator = ModelEvaluator()
    viz = Visualizer()
    
    # Comparación de modelos
    st.markdown("### 📊 Comparación de Modelos")
    
    results_df = pd.DataFrame(results['all_results'])
    display_cols = [col for col in results_df.columns 
                   if col not in ['model', 'y_pred', 'y_proba']]
    
    # Gráfico de comparación
    fig = viz.plot_model_comparison(results_df[display_cols])
    st.pyplot(fig)
    
    # Mejor modelo
    st.markdown("---")
    st.markdown("### 🏆 Análisis del Mejor Modelo")
    
    best_model_data = results['best_model']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("F1-Score", f"{best_model_data['metrics']['F1-Score']:.4f}")
    col2.metric("ROC-AUC", f"{best_model_data['metrics']['ROC-AUC']:.4f}")
    col3.metric("Precision", f"{best_model_data['metrics']['Precision']:.4f}")
    col4.metric("Recall", f"{best_model_data['metrics']['Recall']:.4f}")
    col5.metric("Accuracy", f"{best_model_data['metrics']['Accuracy']:.4f}")
    
    # Matriz de confusión y curvas
    tab1, tab2, tab3 = st.tabs(["🔲 Matriz de Confusión", "📈 Curva ROC", "📊 Precision-Recall"])
    
    with tab1:
        fig = viz.plot_confusion_matrix(
            best_model_data['y_test'],
            best_model_data['y_pred']
        )
        st.pyplot(fig)
    
    with tab2:
        fig = viz.plot_roc_curve(
            best_model_data['y_test'],
            best_model_data['y_proba']
        )
        st.pyplot(fig)
    
    with tab3:
        fig = viz.plot_precision_recall_curve(
            best_model_data['y_test'],
            best_model_data['y_proba']
        )
        st.pyplot(fig)

def show_predictions_page():
    """Página de predicciones"""
    st.markdown("## 🔮 Generación de Predicciones")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Por favor, entrene los modelos primero")
        return
    
    # Verificar si ya hay predicciones
    if st.session_state.predictions_made and 'predictions' in st.session_state:
        st.success("✅ Predicciones ya generadas (cargado desde caché)")
        
        predictions = st.session_state.predictions
        
        # Estadísticas
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predicciones", len(predictions['y_pred']))
        col2.metric("Normal", 
                   f"{(predictions['y_pred']==0).sum():,}",
                   f"{(predictions['y_pred']==0).sum()/len(predictions['y_pred'])*100:.2f}%")
        col3.metric("Fraude", 
                   f"{(predictions['y_pred']==1).sum():,}",
                   f"{(predictions['y_pred']==1).sum()/len(predictions['y_pred'])*100:.2f}%")
        
        # Distribución de probabilidades
        st.markdown("### 📊 Distribución de Probabilidades")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(predictions['y_proba'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(predictions['threshold'], color='red', linestyle='--', linewidth=2,
                  label=f'Threshold: {predictions["threshold"]}')
        ax.set_xlabel('Probabilidad de Fraude')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Probabilidades Predichas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Descargar archivo
        if Path('creditcard_test_evaluate.csv').exists():
            with open('creditcard_test_evaluate.csv', 'rb') as f:
                st.download_button(
                    label="📥 Descargar Predicciones (CSV)",
                    data=f,
                    file_name='creditcard_test_evaluate.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        
        if st.button("🔄 Regenerar Predicciones", use_container_width=True):
            del st.session_state.predictions
            st.session_state.predictions_made = False
            st.rerun()
        
        return
    
    predictor = Predictor()
    
    st.markdown("""
    ### 📝 Generar Predicciones en Conjunto de Test
    
    Esta sección genera las predicciones finales en el archivo de test usando el mejor modelo entrenado.
    """)
    
    threshold = st.slider(
        "Threshold de clasificación:",
        0.1, 0.9, 0.5, 0.01,
        help="Probabilidad mínima para clasificar como fraude"
    )
    
    if st.button("🎯 Generar Predicciones", use_container_width=True):
        with st.spinner("Generando predicciones..."):
            df_test = st.session_state.df_test
            trainer = st.session_state.trainer
            selected_features = st.session_state.selected_features
            
            # Aplicar feature engineering al test
            engineer = FeatureEngineer()
            df_test_fe = engineer.create_features(df_test)
            
            # Hacer predicciones
            predictions = predictor.predict(
                trainer.best_model,
                df_test_fe,
                selected_features,
                trainer.scaler,
                threshold
            )
            
            st.session_state.predictions = predictions
            st.session_state.predictions_made = True
            
            # Guardar en caché
            save_to_cache(predictions, 'predictions.pkl')
            
            # Guardar archivo
            output_df = pd.DataFrame({'Class': predictions['y_pred']})
            output_df.to_csv('creditcard_test_evaluate.csv', index=False)
            
            st.success("✅ Predicciones generadas y guardadas!")
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predicciones", len(predictions['y_pred']))
            col2.metric("Normal", 
                       f"{(predictions['y_pred']==0).sum():,}",
                       f"{(predictions['y_pred']==0).sum()/len(predictions['y_pred'])*100:.2f}%")
            col3.metric("Fraude", 
                       f"{(predictions['y_pred']==1).sum():,}",
                       f"{(predictions['y_pred']==1).sum()/len(predictions['y_pred'])*100:.2f}%")
            
            # Distribución de probabilidades
            st.markdown("### 📊 Distribución de Probabilidades")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(predictions['y_proba'], bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold: {threshold}')
            ax.set_xlabel('Probabilidad de Fraude')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Probabilidades Predichas')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Descargar archivo
            with open('creditcard_test_evaluate.csv', 'rb') as f:
                st.download_button(
                    label="📥 Descargar Predicciones (CSV)",
                    data=f,
                    file_name='creditcard_test_evaluate.csv',
                    mime='text/csv',
                    use_container_width=True
                )

def show_report_page():
    """Página de reporte final"""
    st.markdown("## 📄 Reporte Final")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Complete el proceso de entrenamiento primero")
        return
    
    st.markdown("""
    ### 📊 Resumen Ejecutivo
    
    #### 🎯 Objetivo del Proyecto
    Desarrollar un modelo de Machine Learning para detectar transacciones fraudulentas 
    en tarjetas de crédito, minimizando los falsos positivos que generan cancelaciones del servicio.
    """)
    
    # Información del dataset
    st.markdown("#### 📈 Dataset")
    df_train = st.session_state.df_train
    class_counts = df_train['Class'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Registros", f"{len(df_train):,}")
    col2.metric("Features Originales", df_train.shape[1] - 1)
    col3.metric("Ratio Desbalance", f"1:{int(class_counts[0]/class_counts[1])}")
    
    # Metodología
    st.markdown("""
    #### 🔧 Metodología Aplicada
    
    **1. Análisis Exploratorio**
    - Análisis de distribuciones
    - Identificación de desbalance de clases
    - Análisis de correlaciones
    
    **2. Ingeniería de Características**
    - Transformaciones de Amount (log, sqrt, cuadrado, cubo)
    - Transformaciones cíclicas de Time
    - Interacciones entre features
    - Agregaciones estadísticas
    
    **3. Selección de Características**
    - F-Score (Análisis Univariado)
    - Información Mutua
    - Random Forest Importance
    - PCA (opcional)
    
    **4. Técnicas de Balanceo**
    - Random Undersampling
    - SMOTE
    - ADASYN
    - Class Weights / scale_pos_weight
    
    **5. Modelos Entrenados**
    - Random Forest
    - XGBoost
    - Redes Neuronales Profundas
    """)
    
    # Resultados
    if 'training_results' in st.session_state:
        st.markdown("#### 🏆 Resultados del Mejor Modelo")
        
        best_model = st.session_state.training_results['best_model']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Configuración:**")
            st.write(f"- Modelo: {best_model['model_type']}")
            st.write(f"- Técnica: {best_model['technique']}")
            if 'selected_features' in st.session_state:
                st.write(f"- Features: {len(st.session_state.selected_features)}")
        
        with col2:
            st.markdown("**Métricas:**")
            st.write(f"- F1-Score: {best_model['metrics']['F1-Score']:.4f}")
            st.write(f"- ROC-AUC: {best_model['metrics']['ROC-AUC']:.4f}")
            st.write(f"- Precision: {best_model['metrics']['Precision']:.4f}")
            st.write(f"- Recall: {best_model['metrics']['Recall']:.4f}")
    
    # Justificación de métricas
    st.markdown("""
    #### 📊 Justificación de Métricas
    
    **Métrica Principal Recomendada: F1-Score y ROC-AUC**
    
    **¿Por qué NO usar Accuracy?**
    - Dataset altamente desbalanceado (< 1% fraudes)
    - Un modelo que prediga siempre "Normal" tendría alta accuracy pero sería inútil
    - Accuracy no es apropiada para clases desbalanceadas
    
    **¿Por qué F1-Score?**
    - Balance óptimo entre Precision y Recall
    - Penaliza tanto falsos positivos como falsos negativos
    - Apropiada para clases desbalanceadas
    - Fácil de interpretar
    
    **¿Por qué ROC-AUC?**
    - Independiente del threshold
    - Mide capacidad de discriminación del modelo
    - Robusta ante desbalance de clases
    
    **Consideraciones de Negocio:**
    - **Falsos Positivos**: Bloqueos incorrectos → Cancelación del servicio
    - **Falsos Negativos**: Fraudes no detectados → Pérdidas económicas
    - Balance necesario entre ambos objetivos
    """)
    
    # Recomendaciones
    st.markdown("""
    #### 💡 Recomendaciones
    
    **Para Implementación:**
    1. Sistema de alertas graduales basado en probabilidad:
       - Prob > 0.9: Bloqueo automático
       - Prob 0.7-0.9: Revisión manual prioritaria
       - Prob 0.5-0.7: Monitoreo adicional
       - Prob < 0.5: Aprobar automáticamente
    
    2. Monitoreo continuo de métricas en producción
    
    3. Sistema de feedback para mejora continua
    
    **Para Mejora Continua:**
    1. Re-entrenar el modelo mensualmente con nuevos datos
    2. A/B testing con diferentes thresholds
    3. Incorporar feedback de clientes y analistas
    4. Analizar casos de falsos positivos/negativos
    """)
    
    # Archivos generados
    st.markdown("#### 📁 Archivos Generados")
    st.write("✅ creditcard_test_evaluate.csv - Predicciones en conjunto de test")
    st.write("✅ cache/ - Modelos y datos persistentes para carga rápida")
    st.write("✅ Visualizaciones y análisis en la aplicación")
    
    # Sistema de caché
    st.markdown("---")
    st.markdown("### 💾 Sistema de Caché y Persistencia")
    
    cache_info = get_cache_info()
    
    st.markdown("""
    **Ventajas del sistema implementado:**
    - ⚡ Carga automática de datos al iniciar
    - 💾 Los modelos entrenados se guardan automáticamente
    - 🔄 No necesita reprocesar datos en cada sesión
    - 📦 Gestión inteligente de memoria
    """)
    
    with st.expander("📊 Estado detallado del caché"):
        for description, info in cache_info.items():
            if info['exists']:
                st.success(f"✅ **{description}**: {info['size']}")
            else:
                st.info(f"⏳ **{description}**: No generado aún")
    
    # Contacto
    st.markdown("---")
    st.markdown("""
    ### 📧 Información del Proyecto
    **Desarrollado por:** Sistema de Detección de Fraude - Tenpo
    **Fecha:** Octubre 2025
    
    ---
    **Análisis completado exitosamente** ✅
    """)

if __name__ == "__main__":
    main()
