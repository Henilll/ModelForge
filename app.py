"""
ModelForge - Automated Machine Learning Platform
Installation: pip install streamlit pandas numpy scikit-learn xgboost lightgbm plotly seaborn matplotlib openpyxl reportlab
Run: streamlit run modelforge.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
import base64
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available. Install with: pip install reportlab")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost not available: {str(e)}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"LightGBM not available: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="ModelForge - AutoML Platform",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* --- Main background --- */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        background-attachment: fixed;
        min-height: 100vh;
        padding: 1rem;
    }

    /* --- Streamlit container fixes --- */
    .stApp {
        background: transparent;
    }
    
    .block-container {
        padding: 2.5rem !important;
        background: rgba(0, 0, 0, 0.85);
        border-radius: 24px;
        box-shadow: 0 25px 60px rgba(102, 51, 153, 0.3);
        backdrop-filter: blur(12px);
        max-width: 1200px;
        margin: 2rem auto;
        border: 1px solid rgba(147, 51, 234, 0.2);
    }

    /* --- Text colors --- */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, 
    .stSubheader, .stCaption,
    .stMarkdown p, .stMarkdown li,
    .stMarkContainer *,
    div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    /* --- Headers --- */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(139, 92, 246, 0.5);
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        background: linear-gradient(135deg, #a855f7 0%, #c084fc 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
    }

    .sub-header {
        text-align: center;
        color: #cbd5e1 !important;
        font-size: 1.4rem;
        margin-bottom: 3rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        line-height: 1.6;
    }

    /* --- Gradient Cards --- */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(217, 70, 239, 0.2) 100%) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        padding: 1.5rem !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.5) !important;
        border: 1px solid rgba(139, 92, 246, 0.5) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 0 2px 10px rgba(139, 92, 246, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* --- Custom Cards --- */
    .gradient-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(217, 70, 239, 0.15) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    .gradient-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.5);
    }

    /* --- Progress bars --- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #d946ef 100%) !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
        border: none !important;
    }

    /* --- Buttons --- */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 5px 20px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
        width: auto !important;
        min-width: 120px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.6) !important;
        background: linear-gradient(135deg, #9d6cff 0%, #e055ff 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* --- File uploader --- */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(217, 70, 239, 0.1) 100%) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        padding: 1.5rem !important;
        border: 2px dashed rgba(139, 92, 246, 0.5) !important;
    }
    
    .uploadedFile {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(217, 70, 239, 0.15) 100%) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        padding: 1.2rem !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #ffffff !important;
        margin: 0.5rem 0 !important;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem !important;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(217, 70, 239, 0.1) 100%) !important;
        backdrop-filter: blur(10px);
        padding: 1rem !important;
        border-radius: 16px !important;
        margin-bottom: 2rem !important;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: #cbd5e1 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
    }

    /* --- Dataframes & Tables --- */
    .dataframe {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        background: rgba(0, 0, 0, 0.5) !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        text-align: left !important;
        border: none !important;
    }
    
    .dataframe td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(139, 92, 246, 0.1) !important;
        color: #e2e8f0 !important;
        background: rgba(0, 0, 0, 0.3) !important;
    }
    
    .dataframe tr:hover {
        background-color: rgba(139, 92, 246, 0.1) !important;
    }

    /* --- Input fields --- */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 12px !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        background: rgba(0, 0, 0, 0.5) !important;
        color: white !important;
        min-height: 54px !important;
        height: auto !important;
        line-height: 1.5 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
        background: rgba(0, 0, 0, 0.7) !important;
    }

    /* --- Sliders --- */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.5) 0%, rgba(217, 70, 239, 0.5) 100%) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #8b5cf6 !important;
        border: 3px solid #8b5cf6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    
    .stSlider label {
        color: white !important;
    }

    /* --- Expanders --- */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(217, 70, 239, 0.1) 100%) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: white !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.5) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-top: none;
    }

    /* --- Alert Boxes --- */
    .stAlert {
        border-radius: 16px !important;
        border: none !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px);
    }

    /* Success */
    .stAlert [data-testid="stMarkdownContainer"]:has(> div > .st-emotion-cache-1gulkj5) {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(21, 128, 61, 0.2) 100%) !important;
        color: white !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
    }

    /* Info */
    .stAlert [data-testid="stMarkdownContainer"]:has(> div > .st-emotion-cache-1c7u8ml) {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(217, 70, 239, 0.2) 100%) !important;
        color: white !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }

    /* Warning */
    .stAlert [data-testid="stMarkdownContainer"]:has(> div > .st-emotion-cache-1b3z7r7) {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%) !important;
        color: white !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
    }

    /* Error */
    .stAlert [data-testid="stMarkdownContainer"]:has(> div > .st-emotion-cache-1j9i2x5) {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.2) 100%) !important;
        color: white !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }

    /* --- Charts --- */
    [data-testid="stChart"] {
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2) !important;
        padding: 1rem !important;
        background: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    /* --- Radio buttons & Checkboxes --- */
    .stRadio > div,
    .stCheckbox > div {
        padding: 1rem !important;
        background: rgba(139, 92, 246, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    
    .stRadio label,
    .stCheckbox label {
        color: white !important;
    }

    /* --- Code blocks --- */
    .stCodeBlock {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2) !important;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    /* --- Markdown text --- */
    .stMarkdown p {
        color: #cbd5e1 !important;
        line-height: 1.7 !important;
        margin-bottom: 1rem !important;
    }
    
    .stMarkdown ul, .stMarkdown ol {
        padding-left: 1.5rem !important;
        color: #cbd5e1 !important;
        margin-bottom: 1rem !important;
    }
    
    .stMarkdown a {
        color: #8b5cf6 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    
    .stMarkdown a:hover {
        color: #d946ef !important;
        text-decoration: underline !important;
    }

    /* --- Divider --- */
    .stHorizontalBlock hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, #8b5cf6 0%, #d946ef 100%) !important;
        margin: 2rem 0 !important;
        border-radius: 2px !important;
        opacity: 0.5;
    }

    /* --- Loading spinner --- */
    .stSpinner > div {
        border-color: #8b5cf6 transparent transparent transparent !important;
    }

    /* --- Tooltips --- */
    [data-baseweb="tooltip"] {
        border-radius: 8px !important;
        background: rgba(0, 0, 0, 0.9) !important;
        color: white !important;
        font-weight: 500 !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        backdrop-filter: blur(10px);
    }

    /* --- Container background for sections --- */
    .st-emotion-cache-1v0mbdj {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(217, 70, 239, 0.1) 100%) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px !important;
        padding: 2rem !important;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    /* --- Selection color --- */
    ::selection {
        background: rgba(139, 92, 246, 0.5) !important;
        color: white !important;
    }
    
    ::-moz-selection {
        background: rgba(139, 92, 246, 0.5) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

def load_dataset(file):
    """Load dataset from uploaded file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_column_types(df):
    """Detect numeric and categorical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

def detect_problem_type(target_series):
    """Detect if problem is classification or regression"""
    if target_series.dtype == 'object' or target_series.dtype.name == 'category':
        return 'classification'
    
    unique_values = target_series.nunique()
    total_values = len(target_series)
    
    if unique_values < 20 or unique_values / total_values < 0.05:
        return 'classification'
    else:
        return 'regression'

def preprocess_data(df, target_col, missing_strategy='mean', scale_data=True, encode_categorical=True):
    """Comprehensive data preprocessing"""
    df_processed = df.copy()
    
    # Separate features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Detect column types
    numeric_cols, categorical_cols = detect_column_types(X)
    
    # Handle missing values in features
    if missing_strategy in ['mean', 'median']:
        num_imputer = SimpleImputer(strategy=missing_strategy)
        if numeric_cols:
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical variables
    encoders = {}
    if encode_categorical and categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Handle missing values in target
    if y.isnull().any():
        y = y.fillna(y.mode()[0] if detect_problem_type(y) == 'classification' else y.mean())
    
    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    
    # Scale numeric features
    scaler = None
    if scale_data and numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler, encoders, target_encoder

def get_models(problem_type):
    """Get list of models based on problem type"""
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR()
        }
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    
    return models

def train_and_evaluate_models(X, y, problem_type):
    """Train multiple models and evaluate performance"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = get_models(problem_type)
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.markdown(f'<div class="info-card">🔄 Training {name}...</div>', unsafe_allow_html=True)
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'score': accuracy
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'score': r2
                }
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            continue
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.markdown('<div class="success-box">✅ Training Complete!</div>', unsafe_allow_html=True)
    progress_bar.empty()
    
    return results, X_test, y_test

def plot_dataset_overview(df):
    """Create visualizations for dataset overview"""
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Total Rows", f"{len(df):,}", delta=None)
    with col2:
        st.metric("📋 Total Columns", len(df.columns), delta=None)
    with col3:
        st.metric("⚠️ Missing Values", f"{df.isnull().sum().sum():,}", delta=None)
    with col4:
        st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB", delta=None)
    
    st.markdown("---")
    
    # Data preview
    st.markdown("#### 🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column statistics
    st.markdown("#### 📈 Column Statistics")
    col_stats = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_stats, use_container_width=True)
    
    # Numeric columns visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("#### 📊 Numeric Columns Distribution")
        
        selected_cols = st.multiselect(
            "Select columns to visualize:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        )
        
        if selected_cols:
            # Histograms
            fig = make_subplots(
                rows=(len(selected_cols) + 1) // 2,
                cols=2,
                subplot_titles=selected_cols
            )
            
            for idx, col in enumerate(selected_cols):
                row = idx // 2 + 1
                col_num = idx % 2 + 1
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False, marker_color='#667eea'),
                    row=row, col=col_num
                )
            
            fig.update_layout(height=300 * ((len(selected_cols) + 1) // 2), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            if len(selected_cols) > 1:
                st.markdown("#### 🔥 Correlation Heatmap")
                corr_matrix = df[selected_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=selected_cols,
                    y=selected_cols,
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)

def plot_model_performance(results, problem_type):
    """Visualize model performance comparison"""
    st.markdown("### 🏆 Model Performance Comparison")
    
    if problem_type == 'classification':
        perf_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [r['accuracy'] for r in results.values()],
            'Precision': [r['precision'] for r in results.values()],
            'Recall': [r['recall'] for r in results.values()],
            'F1 Score': [r['f1_score'] for r in results.values()],
            'CV Mean': [r['cv_mean'] for r in results.values()]
        })
        perf_df = perf_df.sort_values('Accuracy', ascending=False)
        
        # Bar chart
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors_list = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
        for idx, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric, 
                x=perf_df['Model'], 
                y=perf_df[metric],
                marker_color=colors_list[idx]
            ))
        
        fig.update_layout(
            title="Classification Metrics Comparison",
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(perf_df.style.highlight_max(axis=0), use_container_width=True)
        
    else:
        perf_df = pd.DataFrame({
            'Model': list(results.keys()),
            'R² Score': [r['r2_score'] for r in results.values()],
            'RMSE': [r['rmse'] for r in results.values()],
            'MAE': [r['mae'] for r in results.values()],
            'CV Mean': [r['cv_mean'] for r in results.values()]
        })
        perf_df = perf_df.sort_values('R² Score', ascending=False)
        
        fig = px.bar(
            perf_df,
            x='Model',
            y='R² Score',
            title="R² Score Comparison",
            color='R² Score',
            color_continuous_scale=[[0, '#667eea'], [1, '#764ba2']]
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='RMSE', x=perf_df['Model'], y=perf_df['RMSE'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='MAE', x=perf_df['Model'], y=perf_df['MAE'], marker_color='#764ba2'))
        fig.update_layout(
            title="Error Metrics Comparison",
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Error",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(perf_df.style.highlight_max(subset=['R² Score'], axis=0)
                    .highlight_min(subset=['RMSE', 'MAE'], axis=0), use_container_width=True)

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix for classification"""
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {i}" for i in range(len(cm))],
        y=[f"Class {i}" for i in range(len(cm))],
        color_continuous_scale=[[0, '#667eea'], [1, '#764ba2']],
        title=f"Confusion Matrix - {model_name}"
    )
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 15 Feature Importance - {model_name}",
            color='Importance',
            color_continuous_scale=[[0, '#667eea'], [1, '#764ba2']]
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def plot_prediction_vs_actual(y_test, y_pred, model_name):
    """Plot predictions vs actual values for regression"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, opacity=0.6, color='#667eea')
    ))
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='#764ba2', dash='dash', width=3)
    ))
    
    fig.update_layout(
        title=f"Predictions vs Actual Values - {model_name}",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def download_model(model, model_name):
    """Create download button for trained model"""
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    
    st.download_button(
        label=f"📥 Download {model_name}",
        data=buffer,
        file_name=f"{model_name.replace(' ', '_').lower()}_model.pkl",
        mime="application/octet-stream"
    )

class PDFFooter(canvas.Canvas):
    """Custom PDF class with footer on every page"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, start=1):
            self.__dict__.update(page)
            self.draw_footer(page_num, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_footer(self, page_num, page_count):
        self.saveState()
        self.setFont('Helvetica', 9)
        
        # Footer text
        footer_text = "ModelForge - By Henil"
        page_text = f"Page {page_num} of {page_count}"
        
        # Draw footer line
        self.setStrokeColorRGB(0.4, 0.49, 0.92)
        self.setLineWidth(2)
        self.line(50, 40, letter[0] - 50, 40)
        
        # Left side - ModelForge
        self.setFillColorRGB(0.4, 0.49, 0.92)
        self.drawString(50, 25, footer_text)
        
        # Center - Date
        date_text = datetime.now().strftime("%B %d, %Y")
        self.drawCentredString(letter[0] / 2, 25, date_text)
        
        # Right side - Page number
        self.drawRightString(letter[0] - 50, 25, page_text)
        
        self.restoreState()

def generate_pdf_report(results, problem_type, dataset_info, preprocessing_info):
    """Generate a modern PDF report"""
    if not REPORTLAB_AVAILABLE:
        st.error("ReportLab not installed. Install with: pip install reportlab")
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=60
    )
    
    # Custom styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=32,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10,
        fontName='Helvetica'
    )
    
    # Story to build PDF
    story = []
    
    # Cover Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("ModelForge", title_style))
    story.append(Paragraph("Automated Machine Learning Report", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    # story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", body_style))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    summary_text = f"""
    This report presents the results of automated machine learning analysis performed on your dataset.
    The analysis included data preprocessing, training multiple models, and comprehensive performance evaluation.
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Dataset Information
    story.append(Paragraph("Dataset Information", heading_style))
    dataset_data = [
        ['Metric', 'Value'],
        ['Total Rows', f"{dataset_info['rows']:,}"],
        ['Total Columns', str(dataset_info['columns'])],
        ['Missing Values', f"{dataset_info['missing']:,}"],
        ['Target Column', dataset_info['target']],
        ['Problem Type', dataset_info['problem_type'].upper()]
    ]
    
    dataset_table = Table(dataset_data, colWidths=[3*inch, 3*inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(dataset_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Preprocessing Details
    story.append(Paragraph("Preprocessing Configuration", heading_style))
    preprocess_data = [
        ['Setting', 'Value'],
        ['Missing Value Strategy', preprocessing_info['missing_strategy']],
        ['Feature Scaling', 'Yes' if preprocessing_info['scale_data'] else 'No'],
        ['Categorical Encoding', 'Yes' if preprocessing_info['encode_categorical'] else 'No']
    ]
    
    preprocess_table = Table(preprocess_data, colWidths=[3*inch, 3*inch])
    preprocess_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(preprocess_table)
    story.append(PageBreak())
    
    # Model Performance Results
    story.append(Paragraph("Model Performance Results", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    if problem_type == 'classification':
        perf_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Mean']]
        for model_name, model_result in results.items():
            perf_data.append([
                model_name,
                f"{model_result['accuracy']:.4f}",
                f"{model_result['precision']:.4f}",
                f"{model_result['recall']:.4f}",
                f"{model_result['f1_score']:.4f}",
                f"{model_result['cv_mean']:.4f}"
            ])
    else:
        perf_data = [['Model', 'R² Score', 'RMSE', 'MAE', 'CV Mean']]
        for model_name, model_result in results.items():
            perf_data.append([
                model_name,
                f"{model_result['r2_score']:.4f}",
                f"{model_result['rmse']:.4f}",
                f"{model_result['mae']:.4f}",
                f"{model_result['cv_mean']:.4f}"
            ])
    
    perf_table = Table(perf_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch] if problem_type == 'classification' 
                       else [2.5*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Best Model Highlight
    best_model_name = max(results, key=lambda x: results[x]['score'])
    best_score = results[best_model_name]['score']
    
    story.append(Paragraph("🏆 Best Performing Model", heading_style))
    best_text = f"""
    <b>Model:</b> {best_model_name}<br/>
    <b>Score:</b> {best_score:.4f}<br/><br/>
    This model demonstrated the highest performance among all trained models and is recommended for deployment.
    """
    story.append(Paragraph(best_text, body_style))
    story.append(PageBreak())
    
    # Detailed Model Analysis
    story.append(Paragraph("Detailed Model Analysis", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    for model_name, model_result in results.items():
        story.append(Paragraph(f"{model_name}", heading_style))
        
        if problem_type == 'classification':
            detail_text = f"""
            <b>Accuracy:</b> {model_result['accuracy']:.4f}<br/>
            <b>Precision:</b> {model_result['precision']:.4f}<br/>
            <b>Recall:</b> {model_result['recall']:.4f}<br/>
            <b>F1 Score:</b> {model_result['f1_score']:.4f}<br/>
            <b>Cross-Validation Mean:</b> {model_result['cv_mean']:.4f} (±{model_result['cv_std']:.4f})<br/>
            """
        else:
            detail_text = f"""
            <b>R² Score:</b> {model_result['r2_score']:.4f}<br/>
            <b>RMSE:</b> {model_result['rmse']:.4f}<br/>
            <b>MAE:</b> {model_result['mae']:.4f}<br/>
            <b>MSE:</b> {model_result['mse']:.4f}<br/>
            <b>Cross-Validation Mean:</b> {model_result['cv_mean']:.4f} (±{model_result['cv_std']:.4f})<br/>
            """
        
        story.append(Paragraph(detail_text, body_style))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Recommendations
    story.append(Paragraph("Recommendations & Next Steps", heading_style))
    recommendations = f"""
    Based on the analysis, here are the recommended next steps:<br/><br/>
    
    <b>1. Model Deployment:</b> Consider deploying the {best_model_name} model which achieved the best performance.<br/><br/>
    
    <b>2. Further Optimization:</b> Fine-tune hyperparameters using techniques like GridSearchCV or RandomizedSearchCV
    for potentially better performance.<br/><br/>
    
    <b>3. Feature Engineering:</b> Explore additional feature engineering techniques to improve model accuracy.<br/><br/>
    
    <b>4. Validation:</b> Validate the model on new, unseen data to ensure it generalizes well.<br/><br/>
    
    <b>5. Monitoring:</b> Implement monitoring to track model performance in production and detect drift.<br/>
    """
    story.append(Paragraph(recommendations, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    conclusion_text = f"""
    This automated machine learning analysis successfully trained and evaluated {len(results)} different models
    on your dataset. The {best_model_name} model emerged as the top performer with a score of {best_score:.4f}.
    All models have been properly validated using cross-validation to ensure robust performance estimates.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Build PDF with custom footer
    doc.build(story, canvasmaker=PDFFooter)
    buffer.seek(0)
    return buffer

def main():
    # Header with animation
    st.markdown('<h1 class="main-header">🔨 ModelForge</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Forge Your Models, Instantly with AI-Powered Precision</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")
        
        # Show available models info
        if not XGBOOST_AVAILABLE or not LIGHTGBM_AVAILABLE:
            with st.expander("ℹ️ Optional Libraries", expanded=False):
                if not XGBOOST_AVAILABLE:
                    st.warning("**XGBoost** not available.\n\nRun: `brew install libomp`\n\nthen: `pip install xgboost`")
                if not LIGHTGBM_AVAILABLE:
                    st.warning("**LightGBM** not available.\n\nRun: `pip install lightgbm`")
                st.info("✅ ModelForge works great with scikit-learn models!")
        
        # File upload
        st.markdown("#### 1️⃣ Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload your dataset in CSV, Excel, or JSON format"
        )
        
        if uploaded_file:
            df = load_dataset(uploaded_file)
            if df is not None:
                st.session_state.dataset = df
                st.markdown(f'<div class="success-box">✅ Loaded {len(df):,} rows × {len(df.columns)} columns</div>', 
                           unsafe_allow_html=True)
        
        # Target column selection
        if st.session_state.dataset is not None:
            st.markdown("---")
            st.markdown("#### 2️⃣ Select Target Column")
            target_col = st.selectbox(
                "Target column to predict:",
                options=st.session_state.dataset.columns.tolist()
            )
            
            if target_col:
                problem_type = detect_problem_type(st.session_state.dataset[target_col])
                st.markdown(f'<div class="info-card">📊 Detected: <b>{problem_type.upper()}</b> problem</div>', 
                           unsafe_allow_html=True)
                
                # Preprocessing options
                st.markdown("---")
                st.markdown("#### 3️⃣ Preprocessing Options")
                missing_strategy = st.selectbox(
                    "Missing value strategy:",
                    options=['mean', 'median', 'most_frequent'],
                    help="Strategy to handle missing values"
                )
                
                scale_data = st.checkbox("Scale numeric features", value=True)
                encode_categorical = st.checkbox("Encode categorical variables", value=True)
                
                # Train models button
                st.markdown("---")
                if st.button("🚀 Train Models", type="primary", use_container_width=True):
                    with st.spinner("🔄 Preprocessing and training models..."):
                        try:
                            # Preprocess data
                            X, y, scaler, encoders, target_encoder = preprocess_data(
                                st.session_state.dataset,
                                target_col,
                                missing_strategy,
                                scale_data,
                                encode_categorical
                            )
                            
                            st.session_state.preprocessed_data = {
                                'X': X,
                                'y': y,
                                'scaler': scaler,
                                'encoders': encoders,
                                'target_encoder': target_encoder,
                                'target_col': target_col,
                                'problem_type': problem_type,
                                'missing_strategy': missing_strategy,
                                'scale_data': scale_data,
                                'encode_categorical': encode_categorical
                            }
                            
                            # Train models
                            results, X_test, y_test = train_and_evaluate_models(X, y, problem_type)
                            
                            st.session_state.results = results
                            st.session_state.models_trained = True
                            
                            # Find best model
                            best_model_name = max(results, key=lambda x: results[x]['score'])
                            st.session_state.best_model = {
                                'name': best_model_name,
                                'model': results[best_model_name]['model'],
                                'score': results[best_model_name]['score']
                            }
                            
                            # st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
                
                # Reset button
                if st.session_state.models_trained:
                    if st.button("🔄 Reset All", use_container_width=True):
                        st.session_state.models_trained = False
                        st.session_state.results = {}
                        st.session_state.best_model = None
                        st.session_state.preprocessed_data = None
                        st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("### 👨‍💻 Created By")
        st.markdown("**Henil Bhavsar**")
        st.markdown("🔨 ModelForge v1.0")
    
    # Main content area
    if st.session_state.dataset is None:
        # Welcome screen
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 👋 Welcome to ModelForge!")
        st.markdown("</div>")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 1️⃣ Upload")
            st.markdown("Upload your dataset in CSV, Excel, or JSON format")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 2️⃣ Configure")
            st.markdown("Select target column and preprocessing options")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 3️⃣ Train")
            st.markdown("Automatically train and compare ML models")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Supported Problem Types")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📊 Classification
            - Binary Classification
            - Multi-class Classification
            - Models: Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Regression
            - Linear Regression
            - Non-linear Regression
            - Models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, XGBoost, LightGBM
            """)
        
    else:
        # Show tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Results", "📈 Visualizations", "💾 Export"])
        
        with tab1:
            plot_dataset_overview(st.session_state.dataset)
        
        with tab2:
            if st.session_state.models_trained:
                results = st.session_state.results
                problem_type = st.session_state.preprocessed_data['problem_type']
                
                # Best model highlight
                best = st.session_state.best_model
                st.markdown(f'<div class="success-box">🏆 <b>Best Model:</b> {best["name"]} | <b>Score:</b> {best["score"]:.4f}</div>', 
                           unsafe_allow_html=True)
                
                # Performance comparison
                plot_model_performance(results, problem_type)
                
                # Individual model details
                st.markdown("---")
                st.markdown("### 🔍 Detailed Model Analysis")
                selected_model = st.selectbox(
                    "Select a model to analyze:",
                    options=list(results.keys())
                )
                
                if selected_model:
                    model_result = results[selected_model]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    if problem_type == 'classification':
                        with col1:
                            st.metric("🎯 Accuracy", f"{model_result['accuracy']:.4f}")
                        with col2:
                            st.metric("🔍 Precision", f"{model_result['precision']:.4f}")
                        with col3:
                            st.metric("📊 Recall", f"{model_result['recall']:.4f}")
                        with col4:
                            st.metric("⚖️ F1 Score", f"{model_result['f1_score']:.4f}")
                        
                        # Confusion matrix
                        st.markdown("---")
                        plot_confusion_matrix(
                            model_result['y_test'],
                            model_result['y_pred'],
                            selected_model
                        )
                        
                        # Classification report
                        st.markdown("#### 📋 Classification Report")
                        report = classification_report(
                            model_result['y_test'],
                            model_result['y_pred'],
                            output_dict=True
                        )
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                    
                    else:
                        with col1:
                            st.metric("📈 R² Score", f"{model_result['r2_score']:.4f}")
                        with col2:
                            st.metric("📉 RMSE", f"{model_result['rmse']:.4f}")
                        with col3:
                            st.metric("📊 MAE", f"{model_result['mae']:.4f}")
                        with col4:
                            st.metric("🎯 MSE", f"{model_result['mse']:.4f}")
                        
                        # Prediction vs Actual plot
                        st.markdown("---")
                        plot_prediction_vs_actual(
                            model_result['y_test'],
                            model_result['y_pred'],
                            selected_model
                        )
                    
                    # Feature importance
                    st.markdown("---")
                    X = st.session_state.preprocessed_data['X']
                    plot_feature_importance(
                        model_result['model'],
                        X.columns.tolist(),
                        selected_model
                    )
            else:
                st.markdown('<div class="info-card">👈 Train models first to see results</div>', unsafe_allow_html=True)
        
        with tab3:
            if st.session_state.models_trained:
                st.markdown("### 📊 Advanced Visualizations")
                
                results = st.session_state.results
                problem_type = st.session_state.preprocessed_data['problem_type']
                
                # Cross-validation scores
                st.markdown("#### 📊 Cross-Validation Scores")
                cv_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'CV Mean': [r['cv_mean'] for r in results.values()],
                    'CV Std': [r['cv_std'] for r in results.values()]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=cv_df['Model'],
                    y=cv_df['CV Mean'],
                    error_y=dict(type='data', array=cv_df['CV Std']),
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    title="Cross-Validation Scores with Standard Deviation",
                    xaxis_title="Model",
                    yaxis_title="CV Score",
                    height=500,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve for classification
                if problem_type == 'classification':
                    st.markdown("---")
                    st.markdown("#### 📈 ROC Curves")
                    fig = go.Figure()
                    
                    for model_name, model_result in results.items():
                        try:
                            model = model_result['model']
                            y_test = model_result['y_test']
                            
                            if hasattr(model, 'predict_proba'):
                                if len(np.unique(y_test)) == 2:
                                    X_test = st.session_state.preprocessed_data['X'].iloc[:len(y_test)]
                                    y_prob = model.predict_proba(X_test)[:, 1]
                                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                                    roc_auc = auc(fpr, tpr)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=fpr,
                                        y=tpr,
                                        name=f'{model_name} (AUC = {roc_auc:.3f})',
                                        mode='lines',
                                        line=dict(width=3)
                                    ))
                        except:
                            pass
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name='Random Classifier',
                        mode='lines',
                        line=dict(dash='dash', color='gray', width=2)
                    ))
                    
                    fig.update_layout(
                        title="ROC Curves Comparison",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=500,
                        template="plotly_white",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot for regression
                if problem_type == 'regression':
                    st.markdown("---")
                    st.markdown("#### 📉 Residual Analysis")
                    selected_model_viz = st.selectbox(
                        "Select model for residual plot:",
                        options=list(results.keys()),
                        key='viz_model'
                    )
                    
                    if selected_model_viz:
                        model_result = results[selected_model_viz]
                        y_test = model_result['y_test']
                        y_pred = model_result['y_pred']
                        residuals = y_test - y_pred
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Residual Plot', 'Residual Distribution')
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=y_pred,
                                y=residuals,
                                mode='markers',
                                marker=dict(size=8, opacity=0.6, color='#667eea'),
                                name='Residuals'
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[y_pred.min(), y_pred.max()],
                                y=[0, 0],
                                mode='lines',
                                line=dict(color='#764ba2', dash='dash', width=2),
                                name='Zero Line'
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Histogram(x=residuals, name='Distribution', marker_color='#667eea'),
                            row=1, col=2
                        )
                        
                        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
                        fig.update_yaxes(title_text="Residuals", row=1, col=1)
                        fig.update_xaxes(title_text="Residuals", row=1, col=2)
                        fig.update_yaxes(title_text="Frequency", row=1, col=2)
                        
                        fig.update_layout(height=500, showlegend=False, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="info-card">👈 Train models first to see visualizations</div>', unsafe_allow_html=True)
        
        with tab4:
            if st.session_state.models_trained:
                st.markdown("### 💾 Export Models and Data")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🤖 Download Trained Models")
                    
                    best = st.session_state.best_model
                    download_model(best['model'], f"Best_Model_{best['name']}")
                    
                    st.markdown("---")
                    st.markdown("#### 📦 Download Individual Models")
                    
                    for model_name, model_result in st.session_state.results.items():
                        download_model(model_result['model'], model_name)
                
                with col2:
                    st.markdown("#### 📊 Download Predictions")
                    
                    pred_model = st.selectbox(
                        "Select model:",
                        options=list(st.session_state.results.keys())
                    )
                    
                    if pred_model:
                        model_result = st.session_state.results[pred_model]
                        
                        predictions_df = pd.DataFrame({
                            'Actual': model_result['y_test'],
                            'Predicted': model_result['y_pred']
                        })
                        
                        if st.session_state.preprocessed_data['problem_type'] == 'regression':
                            predictions_df['Residual'] = predictions_df['Actual'] - predictions_df['Predicted']
                            predictions_df['Absolute_Error'] = np.abs(predictions_df['Residual'])
                        
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name=f"{pred_model.replace(' ', '_').lower()}_predictions.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("**Preview:**")
                        st.dataframe(predictions_df.head(10), use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("#### 💽 Download Preprocessed Data")
                    
                    X = st.session_state.preprocessed_data['X']
                    y = st.session_state.preprocessed_data['y']
                    
                    preprocessed_df = X.copy()
                    preprocessed_df[st.session_state.preprocessed_data['target_col']] = y
                    
                    csv = preprocessed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Preprocessed Dataset",
                        data=csv,
                        file_name="preprocessed_data.csv",
                        mime="text/csv"
                    )
                
                st.markdown("---")
                st.markdown("### 📄 Generate Professional Report")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("""
                    Generate a comprehensive PDF report with:
                    - Executive summary
                    - Dataset information
                    - Preprocessing details
                    - Model performance comparison
                    - Detailed analysis
                    - Recommendations
                    - Professional formatting with ModelForge branding
                    """)
                
                with col2:
                    if REPORTLAB_AVAILABLE:
                        if st.button("🎨 Generate PDF Report", type="primary", use_container_width=True):
                            with st.spinner("📄 Creating professional PDF report..."):
                                dataset_info = {
                                    'rows': len(st.session_state.dataset),
                                    'columns': len(st.session_state.dataset.columns),
                                    'missing': st.session_state.dataset.isnull().sum().sum(),
                                    'target': st.session_state.preprocessed_data['target_col'],
                                    'problem_type': st.session_state.preprocessed_data['problem_type']
                                }
                                
                                preprocessing_info = {
                                    'missing_strategy': st.session_state.preprocessed_data['missing_strategy'],
                                    'scale_data': st.session_state.preprocessed_data['scale_data'],
                                    'encode_categorical': st.session_state.preprocessed_data['encode_categorical']
                                }
                                
                                pdf_buffer = generate_pdf_report(
                                    st.session_state.results,
                                    st.session_state.preprocessed_data['problem_type'],
                                    dataset_info,
                                    preprocessing_info
                                )
                                
                                if pdf_buffer:
                                    st.download_button(
                                        label="📥 Download PDF Report",
                                        data=pdf_buffer,
                                        file_name=f"ModelForge_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                                    st.success("✅ PDF report generated successfully!")
                    else:
                        st.error("❌ ReportLab not installed")
                        st.info("Install with: `pip install reportlab`")
                
                st.markdown("---")
                st.markdown("### 📝 Generate Markdown Report")
                
                if st.button("📋 Generate Markdown Report", use_container_width=True):
                    report = f"""# ModelForge - AutoML Report

**Generated on:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

## Executive Summary

This report presents the results of automated machine learning analysis performed on your dataset.
The analysis included data preprocessing, training multiple models, and comprehensive performance evaluation.

---

## Dataset Information

| Metric | Value |
|--------|-------|
| Total Rows | {len(st.session_state.dataset):,} |
| Total Columns | {len(st.session_state.dataset.columns)} |
| Missing Values | {st.session_state.dataset.isnull().sum().sum():,} |
| Target Column | {st.session_state.preprocessed_data['target_col']} |
| Problem Type | {st.session_state.preprocessed_data['problem_type'].upper()} |

---

## Preprocessing Configuration

| Setting | Value |
|---------|-------|
| Missing Value Strategy | {st.session_state.preprocessed_data['missing_strategy']} |
| Feature Scaling | {'Yes' if st.session_state.preprocessed_data['scale_data'] else 'No'} |
| Categorical Encoding | {'Yes' if st.session_state.preprocessed_data['encode_categorical'] else 'No'} |

---

## Model Performance Results

"""
                    problem_type = st.session_state.preprocessed_data['problem_type']
                    
                    if problem_type == 'classification':
                        report += "| Model | Accuracy | Precision | Recall | F1 Score | CV Mean |\n"
                        report += "|-------|----------|-----------|--------|----------|----------|\n"
                        for model_name, model_result in st.session_state.results.items():
                            report += f"| {model_name} | {model_result['accuracy']:.4f} | {model_result['precision']:.4f} | "
                            report += f"{model_result['recall']:.4f} | {model_result['f1_score']:.4f} | {model_result['cv_mean']:.4f} |\n"
                    else:
                        report += "| Model | R² Score | RMSE | MAE | CV Mean |\n"
                        report += "|-------|----------|------|-----|----------|\n"
                        for model_name, model_result in st.session_state.results.items():
                            report += f"| {model_name} | {model_result['r2_score']:.4f} | {model_result['rmse']:.4f} | "
                            report += f"{model_result['mae']:.4f} | {model_result['cv_mean']:.4f} |\n"
                    
                    best_model_name = max(st.session_state.results, key=lambda x: st.session_state.results[x]['score'])
                    best_score = st.session_state.results[best_model_name]['score']
                    
                    report += f"""
---

## 🏆 Best Performing Model

**Model:** {best_model_name}  
**Score:** {best_score:.4f}

This model demonstrated the highest performance among all trained models and is recommended for deployment.

---

## Detailed Model Analysis

"""
                    for model_name, model_result in st.session_state.results.items():
                        report += f"\n### {model_name}\n\n"
                        if problem_type == 'classification':
                            report += f"- **Accuracy:** {model_result['accuracy']:.4f}\n"
                            report += f"- **Precision:** {model_result['precision']:.4f}\n"
                            report += f"- **Recall:** {model_result['recall']:.4f}\n"
                            report += f"- **F1 Score:** {model_result['f1_score']:.4f}\n"
                        else:
                            report += f"- **R² Score:** {model_result['r2_score']:.4f}\n"
                            report += f"- **RMSE:** {model_result['rmse']:.4f}\n"
                            report += f"- **MAE:** {model_result['mae']:.4f}\n"
                        report += f"- **CV Mean:** {model_result['cv_mean']:.4f} (±{model_result['cv_std']:.4f})\n"
                    
                    report += f"""
---

## Recommendations & Next Steps

Based on the analysis, here are the recommended next steps:

1. **Model Deployment:** Consider deploying the {best_model_name} model which achieved the best performance.

2. **Further Optimization:** Fine-tune hyperparameters using techniques like GridSearchCV or RandomizedSearchCV 
   for potentially better performance.

3. **Feature Engineering:** Explore additional feature engineering techniques to improve model accuracy.

4. **Validation:** Validate the model on new, unseen data to ensure it generalizes well.

5. **Monitoring:** Implement monitoring to track model performance in production and detect drift.

---

## Conclusion

This automated machine learning analysis successfully trained and evaluated {len(st.session_state.results)} different models
on your dataset. The {best_model_name} model emerged as the top performer with a score of {best_score:.4f}.
All models have been properly validated using cross-validation to ensure robust performance estimates.

---

**ModelForge - By Henil**  
*Forge Your Models, Instantly*
"""
                    
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=report,
                        file_name=f"ModelForge_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
                    with st.expander("📄 Preview Report"):
                        st.markdown(report)
            else:
                st.markdown('<div class="info-card">👈 Train models first to export results</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
