# healthcare_complete.py - Application Streamlit Complète
# Intègre EDA + Prédiction de la qualité des soins

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from scipy import stats
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Healthcare Analytics & Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background-color: #e1f5fe;
        padding: 1rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Chargement des données avec cache"""
    try:
        df = pd.read_csv("Quality.csv")
        # Conversion des types
        if df['StartedOnCombination'].dtype == 'object':
            df['StartedOnCombination'] = df['StartedOnCombination'].map({'FALSE': 0, 'TRUE': 1})
        return df
    except FileNotFoundError:
        st.error("❌ Fichier Quality.csv non trouvé!")
        return None

@st.cache_resource
def load_model():
    """Chargement du modèle avec cache"""
    try:
        with open("model_A.pkl", "rb") as f:
            saved = pickle.load(f)
        return saved["model"], saved["features"]
    except FileNotFoundError:
        st.error("❌ Modèle model_A.pkl non trouvé! Veuillez d'abord entraîner le modèle.")
        return None, None

def overview_section(df):
    """Section Vue d'ensemble"""
    st.markdown('<h2 class="section-header">📊 Vue d\'ensemble du Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Nombre de patients", df.shape[0])
    with col2:
        st.metric("📊 Nombre de variables", df.shape[1])
    with col3:
        poor_care_pct = (df['PoorCare'].sum() / len(df)) * 100
        st.metric("⚠️ Taux PoorCare", f"{poor_care_pct:.1f}%")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("🔍 Valeurs manquantes", f"{missing_pct:.1f}%")
    
    # Aperçu des données
    st.subheader("👀 Aperçu des données")
    st.dataframe(df.head(10))
    
    # Informations sur les colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informations sur les colonnes")
        info_df = pd.DataFrame({
            'Type': df.dtypes.astype(str),
            'Non-null': df.count(),
            'Null': df.isnull().sum()
        })
        st.dataframe(info_df)
    
    with col2:
        st.subheader("🎯 Distribution de la variable cible")
        target_counts = df['PoorCare'].value_counts()
        
        fig = px.pie(
            values=target_counts.values,
            names=['Bonne qualité (0)', 'Mauvaise qualité (1)'],
            title="Distribution PoorCare",
            color_discrete_sequence=['lightgreen', 'lightcoral']
        )
        st.plotly_chart(fig, use_container_width=True)

def descriptive_stats_section(df):
    """Section Statistiques descriptives"""
    st.markdown('<h2 class="section-header">📈 Statistiques Descriptives</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'MemberID' in numeric_cols:
        numeric_cols.remove('MemberID')
    
    # Statistiques de base
    st.subheader("📊 Statistiques de base")
    desc_stats = df[numeric_cols].describe()
    st.dataframe(desc_stats, use_container_width=True)
    
    # Statistiques avancées
    st.subheader("🔬 Statistiques avancées")
    advanced_stats = pd.DataFrame({
        'Médiane': df[numeric_cols].median(),
        'Mode': df[numeric_cols].mode().iloc[0],
        'Asymétrie': df[numeric_cols].skew(),
        'Aplatissement': df[numeric_cols].kurtosis(),
        'Variance': df[numeric_cols].var()
    })
    st.dataframe(advanced_stats, use_container_width=True)
    
    # Distributions
    st.subheader("📊 Distributions des variables")
    
    selected_vars = st.multiselect(
        "Sélectionnez les variables à visualiser:",
        numeric_cols,
        default=numeric_cols[:3]
    )
    
    if selected_vars:
        for var in selected_vars:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df[var], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution de {var}')
                ax.set_xlabel(var)
                ax.set_ylabel('Fréquence')
                ax.grid(True, alpha=0.3)
                
                # Statistiques sur le graphique
                mean_val = df[var].mean()
                median_val = df[var].median()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'Médiane: {median_val:.2f}')
                ax.legend()
                
                st.pyplot(fig)
            
            with col2:
                # Boxplot
                fig = px.box(df, y=var, title=f'Boxplot de {var}')
                st.plotly_chart(fig, use_container_width=True)

def correlation_analysis_section(df):
    """Section Analyse des corrélations"""
    st.markdown('<h2 class="section-header">🔗 Analyse des Corrélations</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'MemberID' in numeric_cols:
        numeric_cols.remove('MemberID')
    
    # Matrice de corrélation
    corr_matrix = df[numeric_cols].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', ax=ax)
        ax.set_title('Matrice de corrélation')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Corrélations avec PoorCare")
        target_corr = corr_matrix['PoorCare'].drop('PoorCare').sort_values(key=abs, ascending=False)
        
        # Graphique en barres des corrélations
        fig = px.bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation='h',
            title="Corrélations avec PoorCare",
            color=target_corr.values,
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table des corrélations
        corr_df = pd.DataFrame({
            'Variable': target_corr.index,
            'Corrélation': target_corr.values,
            'Force': [
                "Très forte" if abs(x) > 0.7 else
                "Forte" if abs(x) > 0.5 else
                "Modérée" if abs(x) > 0.3 else
                "Faible" if abs(x) > 0.1 else
                "Très faible"
                for x in target_corr.values
            ]
        })
        st.dataframe(corr_df, use_container_width=True)

def outlier_analysis_section(df):
    """Section Analyse des outliers"""
    st.markdown('<h2 class="section-header">🔍 Analyse des Valeurs Aberrantes</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'MemberID' in numeric_cols:
        numeric_cols.remove('MemberID')
    if 'PoorCare' in numeric_cols:
        numeric_cols.remove('PoorCare')
    
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        n_outliers = len(outliers)
        outlier_percentage = (n_outliers / len(df)) * 100
        
        outlier_summary.append({
            'Variable': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Limite_inf': lower_bound,
            'Limite_sup': upper_bound,
            'Nb_outliers': n_outliers,
            'Pourcentage': outlier_percentage
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Résumé des outliers")
        st.dataframe(outlier_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Visualisation des outliers")
        selected_var = st.selectbox("Sélectionnez une variable:", numeric_cols)
        
        if selected_var:
            fig = px.box(df, y=selected_var, title=f'Boxplot de {selected_var}')
            st.plotly_chart(fig, use_container_width=True)

def bivariate_analysis_section(df):
    """Section Analyse bivariée"""
    st.markdown('<h2 class="section-header">📈 Analyse Bivariée (Variables vs PoorCare)</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'MemberID' in numeric_cols:
        numeric_cols.remove('MemberID')
    if 'PoorCare' in numeric_cols:
        numeric_cols.remove('PoorCare')
    
    # Tests statistiques
    st.subheader("🔬 Tests statistiques (Mann-Whitney U)")
    
    test_results = []
    for col in numeric_cols:
        group_0 = df[df['PoorCare'] == 0][col]
        group_1 = df[df['PoorCare'] == 1][col]
        
        statistic, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
        
        test_results.append({
            'Variable': col,
            'Médiane_PoorCare_0': group_0.median(),
            'Médiane_PoorCare_1': group_1.median(),
            'Statistique_U': statistic,
            'p_value': p_value,
            'Significatif': 'Oui' if p_value < 0.05 else 'Non'
        })
    
    test_df = pd.DataFrame(test_results)
    st.dataframe(test_df, use_container_width=True)
    
    # Visualisations comparatives
    st.subheader("📊 Comparaisons visuelles par classe")
    
    selected_vars_biv = st.multiselect(
        "Sélectionnez les variables à comparer:",
        numeric_cols,
        default=numeric_cols[:2]
    )
    
    if selected_vars_biv:
        for var in selected_vars_biv:
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot par classe
                fig = px.box(df, x='PoorCare', y=var, 
                           title=f'{var} par PoorCare',
                           color='PoorCare',
                           color_discrete_sequence=['lightgreen', 'lightcoral'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Violin plot
                fig = px.violin(df, x='PoorCare', y=var,
                              title=f'Distribution de {var} par PoorCare',
                              color='PoorCare',
                              color_discrete_sequence=['lightgreen', 'lightcoral'])
                st.plotly_chart(fig, use_container_width=True)

def prediction_section(df):
    """Section Prédiction"""
    st.markdown('<h2 class="section-header">🔮 Prédiction de la Qualité des Soins</h2>', unsafe_allow_html=True)
    
    # Charger le modèle
    model, feature_cols = load_model()
    
    if model is None:
        st.error("❌ Modèle non disponible. Veuillez d'abord entraîner le modèle avec logreg.py")
        return
    
    st.success("✅ Modèle chargé avec succès!")
    
    # Interface de prédiction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Paramètres d'entrée")
        
        # Inputs pour les features
        ERVisits = st.slider("ERVisits", 0, 20, 1)
        OfficeVisits = st.slider("OfficeVisits", 0, 60, 13)
        Narcotics = st.slider("Narcotics", 0, 100, 4)
        ProviderCount = st.slider("ProviderCount", 1, 120, 23)
        NumberClaims = st.slider("NumberClaims", 0, 400, 43)
        StartedOnCombination = st.selectbox("StartedOnCombination", [0, 1], index=1)
        
        st.subheader("⚙️ Paramètres du modèle")
        threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
        
        # Préparer les données d'entrée
        user_input = pd.DataFrame({
            "ERVisits": [ERVisits],
            "OfficeVisits": [OfficeVisits],
            "Narcotics": [Narcotics],
            "ProviderCount": [ProviderCount],
            "NumberClaims": [NumberClaims],
            "StartedOnCombination": [int(StartedOnCombination)],
        })[feature_cols]
        
        st.subheader("📊 Vos entrées")
        st.dataframe(user_input, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Résultats de prédiction")
        
        if st.button("🔮 Prédire", type="primary"):
            # Prédiction
            proba = float(model.predict_proba(user_input)[0, 1])
            y_pred = int(proba >= threshold)
            
            # Affichage des résultats
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric(
                    "Probabilité PoorCare",
                    f"{proba:.3f}",
                    f"Seuil: {threshold}"
                )
            
            with col2b:
                if y_pred == 1:
                    st.error(f"⚠️ Prédiction: RISQUE ÉLEVÉ")
                else:
                    st.success(f"✅ Prédiction: RISQUE FAIBLE")
            
            # Graphique de probabilité
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilité de PoorCare"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails techniques
            with st.expander("🔧 Détails techniques"):
                st.write("**Caractéristiques du modèle:**")
                st.write("- Algorithme: Régression Logistique")
                st.write("- Préprocessing: StandardScaler")
                st.write("- Solver: liblinear")
                st.write("- Class weight: balanced")
                st.write(f"- Seuil de décision: {threshold}")
                st.write(f"- Features utilisées: {', '.join(feature_cols)}")
        
        # Analyse comparative avec le dataset
        st.subheader("📊 Comparaison avec le dataset")
        
        # Percentiles des valeurs entrées
        percentiles_df = pd.DataFrame({
            'Variable': feature_cols,
            'Votre valeur': [
                ERVisits, OfficeVisits, Narcotics, 
                ProviderCount, NumberClaims, StartedOnCombination
            ],
            'Percentile': [
                stats.percentileofscore(df[col], val) 
                for col, val in zip(feature_cols, [
                    ERVisits, OfficeVisits, Narcotics,
                    ProviderCount, NumberClaims, StartedOnCombination
                ])
            ]
        })
        
        st.dataframe(percentiles_df, use_container_width=True)

def model_performance_section(df):
    """Section Analyse de performance du modèle"""
    st.markdown('<h2 class="section-header">🎯 Analyse de Performance du Modèle</h2>', unsafe_allow_html=True)
    
    # Charger le modèle
    model, feature_cols = load_model()
    
    if model is None:
        st.error("❌ Modèle non disponible. Veuillez d'abord entraîner le modèle avec logreg.py")
        return
    
    st.success("✅ Modèle chargé avec succès!")
    
    # Préparer les données
    X = df[feature_cols]
    y = df['PoorCare']
    
    # Division train/test (même que dans logreg.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Seuil personnalisable
    st.subheader("⚙️ Configuration du seuil")
    threshold = st.slider("Seuil de décision pour l'analyse", 0.0, 1.0, 0.5, 0.01)
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    
    # Métriques de base
    st.subheader("📊 Métriques de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_pred_custom)
        st.metric("🎯 Accuracy", f"{accuracy:.3f}")
    
    with col2:
        precision = precision_score(y_test, y_pred_custom, zero_division=0)
        st.metric("🔍 Precision", f"{precision:.3f}")
    
    with col3:
        recall = recall_score(y_test, y_pred_custom, zero_division=0)
        st.metric("📡 Recall", f"{recall:.3f}")
    
    with col4:
        f1 = f1_score(y_test, y_pred_custom, zero_division=0)
        st.metric("⚖️ F1-Score", f"{f1:.3f}")
    
    # Métriques avancées
    st.subheader("🔬 Métriques Avancées")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.metric("📈 ROC-AUC", f"{roc_auc:.3f}")
    
    with col2:
        avg_precision = average_precision_score(y_test, y_pred_proba)
        st.metric("📊 AP Score", f"{avg_precision:.3f}")
    
    with col3:
        balanced_acc = balanced_accuracy_score(y_test, y_pred_custom)
        st.metric("⚖️ Balanced Acc", f"{balanced_acc:.3f}")
    
    with col4:
        mcc = matthews_corrcoef(y_test, y_pred_custom)
        st.metric("🔗 MCC", f"{mcc:.3f}")
    
    # Matrice de confusion
    st.subheader("🎭 Matrice de Confusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm = confusion_matrix(y_test, y_pred_custom)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Prédite: Bonne', 'Prédite: Mauvaise'],
                    yticklabels=['Réelle: Bonne', 'Réelle: Mauvaise'])
        ax.set_title(f'Matrice de Confusion (Seuil: {threshold})')
        ax.set_ylabel('Valeurs Réelles')
        ax.set_xlabel('Valeurs Prédites')
        st.pyplot(fig)
    
    with col2:
        # Calculs détaillés de la matrice de confusion
        tn, fp, fn, tp = cm.ravel()
        
        st.write("**📋 Détails de la Matrice:**")
        st.write(f"• Vrais Négatifs (TN): {tn}")
        st.write(f"• Faux Positifs (FP): {fp}")
        st.write(f"• Faux Négatifs (FN): {fn}")
        st.write(f"• Vrais Positifs (TP): {tp}")
        
        st.write("**📈 Calculs dérivés:**")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        st.write(f"• Spécificité: {specificity:.3f}")
        st.write(f"• Sensibilité: {sensitivity:.3f}")
        st.write(f"• VPP (PPV): {ppv:.3f}")
        st.write(f"• VPN (NPV): {npv:.3f}")
    
    # Courbes ROC et Precision-Recall
    st.subheader("📈 Courbes de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Courbe ROC
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        # Point pour le seuil actuel
        if len(roc_thresholds) > 0:
            threshold_idx = np.argmin(np.abs(roc_thresholds - threshold))
            fig.add_trace(go.Scatter(
                x=[fpr[threshold_idx]], y=[tpr[threshold_idx]],
                mode='markers',
                name=f'Seuil {threshold}',
                marker=dict(color='orange', size=10)
            ))
        
        fig.update_layout(
            title='Courbe ROC',
            xaxis_title='Taux de Faux Positifs',
            yaxis_title='Taux de Vrais Positifs',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Courbe Precision-Recall
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Ligne de base (proportion de positifs)
        baseline = y_test.mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color="red",
                      annotation_text=f"Baseline ({baseline:.3f})")
        
        fig.update_layout(
            title='Courbe Précision-Rappel',
            xaxis_title='Rappel',
            yaxis_title='Précision',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des scores de probabilité
    st.subheader("📊 Distribution des Scores de Probabilité")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogrammes séparés par classe
    y_pred_proba_0 = y_pred_proba[y_test == 0]
    y_pred_proba_1 = y_pred_proba[y_test == 1]
    
    ax.hist(y_pred_proba_0, bins=20, alpha=0.7, label='PoorCare = 0', color='lightgreen', density=True)
    ax.hist(y_pred_proba_1, bins=20, alpha=0.7, label='PoorCare = 1', color='lightcoral', density=True)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil = {threshold}')
    
    ax.set_xlabel('Probabilité Prédite')
    ax.set_ylabel('Densité')
    ax.set_title('Distribution des Probabilités par Classe Réelle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Validation croisée
    st.subheader("🔄 Validation Croisée")
    
    with st.spinner("Exécution de la validation croisée..."):
        cv_folds = st.selectbox("Nombre de folds:", [3, 5, 10], index=1)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Différentes métriques
        cv_scores = {
            'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
            'precision': cross_val_score(model, X, y, cv=skf, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=skf, scoring='recall'),
            'f1': cross_val_score(model, X, y, cv=skf, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        }
        
        cv_df = pd.DataFrame({
            'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Moyenne': [scores.mean() for scores in cv_scores.values()],
            'Écart-type': [scores.std() for scores in cv_scores.values()],
            'Min': [scores.min() for scores in cv_scores.values()],
            'Max': [scores.max() for scores in cv_scores.values()]
        })
        
        st.dataframe(cv_df)
        
        # Graphique des scores CV
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(cv_scores.keys())
        means = [scores.mean() for scores in cv_scores.values()]
        stds = [scores.std() for scores in cv_scores.values()]
        
        bars = ax.bar(metrics, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        ax.set_ylabel('Score')
        ax.set_title('Scores de Validation Croisée avec Intervalles de Confiance')
        ax.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # Rapport de classification détaillé
    st.subheader("📋 Rapport de Classification Détaillé")
    
    report = classification_report(y_test, y_pred_custom, output_dict=True)
    
    # Convertir en DataFrame pour un meilleur affichage
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(3)
    
    st.dataframe(report_df)
    
    # Recommandations d'amélioration
    st.subheader("💡 Recommandations d'Amélioration")
    
    recommendations = []
    
    if accuracy < 0.8:
        recommendations.append("• Accuracy faible - Considérer d'autres algorithmes ou feature engineering")
    
    if precision < 0.7:
        recommendations.append("• Précision faible - Réduire les faux positifs, ajuster le seuil vers le haut")
    
    if recall < 0.7:
        recommendations.append("• Rappel faible - Réduire les faux négatifs, ajuster le seuil vers le bas")
    
    if roc_auc < 0.8:
        recommendations.append("• ROC-AUC faible - Le modèle a du mal à distinguer les classes")
    
    if mcc < 0.5:
        recommendations.append("• MCC faible - Performance globale médiocre, revoir l'approche")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Performance globale satisfaisante!")
        recommendations.append("• Considérer la validation sur de nouvelles données")
        recommendations.append("• Surveiller la dérive du modèle en production")
    
    for rec in recommendations:
        st.write(rec)

def main():
    """Fonction principale de l'application"""
    # En-tête
    st.markdown('<h1 class="main-header">🏥 Healthcare Analytics & Prediction Platform</h1>', unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar pour navigation
    st.sidebar.title("🧭 Navigation")
    sections = [
        "📊 Vue d'ensemble",
        "📈 Statistiques descriptives", 
        "🔗 Analyse des corrélations",
        "🔍 Analyse des outliers",
        "📈 Analyse bivariée",
        "🎯 Performance du modèle",
        "🔮 Prédiction"
    ]
    
    selected_section = st.sidebar.selectbox("Choisissez une section:", sections)
    
    # Affichage de la section sélectionnée
    if selected_section == "📊 Vue d'ensemble":
        overview_section(df)
    elif selected_section == "📈 Statistiques descriptives":
        descriptive_stats_section(df)
    elif selected_section == "🔗 Analyse des corrélations":
        correlation_analysis_section(df)
    elif selected_section == "🔍 Analyse des outliers":
        outlier_analysis_section(df)
    elif selected_section == "📈 Analyse bivariée":
        bivariate_analysis_section(df)
    elif selected_section == "🎯 Performance du modèle":
        model_performance_section(df)
    elif selected_section == "🔮 Prédiction":
        prediction_section(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Informations dataset:**")
    st.sidebar.write(f"• Patients: {df.shape[0]}")
    st.sidebar.write(f"• Variables: {df.shape[1]}")
    st.sidebar.write(f"• PoorCare: {df['PoorCare'].sum()}/{len(df)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Outils disponibles:**")
    st.sidebar.write("• Analyse exploratoire complète")
    st.sidebar.write("• Prédiction en temps réel")
    st.sidebar.write("• Visualisations interactives")
    st.sidebar.write("• Tests statistiques")

if __name__ == "__main__":
    main()
