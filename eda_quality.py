# EDA_Quality.py - Analyse Exploratoire des Données Détaillée
# Analyse complète du dataset Quality.csv pour prédiction de la qualité des soins

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration pour les graphiques
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("🏥 ANALYSE EXPLORATOIRE DES DONNÉES - QUALITY HEALTHCARE DATASET")
print("="*80)

def load_and_basic_info():
    """Chargement des données et informations de base"""
    print("\n📊 1. CHARGEMENT ET INFORMATIONS GÉNÉRALES")
    print("-" * 50)
    
    # Chargement
    df = pd.read_csv("Quality.csv")
    
    print(f"📈 Dimensions du dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"📊 Taille en mémoire: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\n🔍 Aperçu des premières lignes:")
    print(df.head())
    
    print("\n🔍 Aperçu des dernières lignes:")
    print(df.tail())
    
    print("\n📋 Informations sur les colonnes:")
    print(df.info())
    
    print("\n📊 Types de données:")
    print(df.dtypes)
    
    return df

def missing_values_analysis(df):
    """Analyse des valeurs manquantes"""
    print("\n🔍 2. ANALYSE DES VALEURS MANQUANTES")
    print("-" * 50)
    
    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Colonnes': df.columns,
        'Valeurs manquantes': missing_data,
        'Pourcentage (%)': missing_percent
    }).sort_values('Valeurs manquantes', ascending=False)
    
    print(missing_df)
    
    if missing_data.sum() == 0:
        print("✅ Aucune valeur manquante détectée!")
    else:
        print(f"⚠️  Total de valeurs manquantes: {missing_data.sum()}")
        
        # Visualisation des valeurs manquantes
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Carte de chaleur des valeurs manquantes')
        plt.tight_layout()
        plt.show()
    
    return missing_df

def data_types_and_conversion(df):
    """Analyse et conversion des types de données"""
    print("\n🔄 3. ANALYSE ET CONVERSION DES TYPES DE DONNÉES")
    print("-" * 50)
    
    print("Types de données avant conversion:")
    print(df.dtypes)
    
    # Conversion de StartedOnCombination
    if df['StartedOnCombination'].dtype == 'object':
        print("\n🔄 Conversion de StartedOnCombination (FALSE/TRUE → 0/1)")
        df['StartedOnCombination'] = df['StartedOnCombination'].map({'FALSE': 0, 'TRUE': 1})
        print("✅ Conversion réussie")
    
    print("\nTypes de données après conversion:")
    print(df.dtypes)
    
    # Vérification des valeurs uniques pour variables catégorielles
    categorical_cols = ['StartedOnCombination', 'PoorCare']
    for col in categorical_cols:
        print(f"\n🏷️  Valeurs uniques pour {col}:")
        print(f"   Valeurs: {sorted(df[col].unique())}")
        print(f"   Comptes: {df[col].value_counts().to_dict()}")
    
    return df

def descriptive_statistics(df):
    """Statistiques descriptives détaillées"""
    print("\n📊 4. STATISTIQUES DESCRIPTIVES DÉTAILLÉES")
    print("-" * 50)
    
    print("📈 Statistiques pour toutes les variables numériques:")
    desc_stats = df.describe()
    print(desc_stats)
    
    # Statistiques supplémentaires
    print("\n📊 Statistiques supplémentaires:")
    additional_stats = pd.DataFrame({
        'Médiane': df.select_dtypes(include=[np.number]).median(),
        'Mode': df.select_dtypes(include=[np.number]).mode().iloc[0],
        'Variance': df.select_dtypes(include=[np.number]).var(),
        'Écart-type': df.select_dtypes(include=[np.number]).std(),
        'Asymétrie (Skewness)': df.select_dtypes(include=[np.number]).skew(),
        'Aplatissement (Kurtosis)': df.select_dtypes(include=[np.number]).kurtosis(),
        'Q1 (25%)': df.select_dtypes(include=[np.number]).quantile(0.25),
        'Q3 (75%)': df.select_dtypes(include=[np.number]).quantile(0.75),
        'IQR': df.select_dtypes(include=[np.number]).quantile(0.75) - df.select_dtypes(include=[np.number]).quantile(0.25)
    })
    print(additional_stats)
    
    return desc_stats, additional_stats

def target_variable_analysis(df):
    """Analyse détaillée de la variable cible"""
    print("\n🎯 5. ANALYSE DE LA VARIABLE CIBLE (PoorCare)")
    print("-" * 50)
    
    target_counts = df['PoorCare'].value_counts()
    target_props = df['PoorCare'].value_counts(normalize=True)
    
    print("🔢 Distribution de PoorCare:")
    for idx, (count, prop) in enumerate(zip(target_counts, target_props)):
        label = "Bonne qualité (0)" if idx == 0 else "Mauvaise qualité (1)"
        print(f"   {label}: {count} ({prop:.2%})")
    
    # Test d'équilibrage des classes
    ratio = target_counts.min() / target_counts.max()
    print(f"\n⚖️  Ratio d'équilibrage: {ratio:.3f}")
    
    if ratio < 0.3:
        print("⚠️  Dataset déséquilibré détecté!")
    elif ratio < 0.5:
        print("⚠️  Léger déséquilibre des classes")
    else:
        print("✅ Classes relativement équilibrées")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique en barres
    target_counts.plot(kind='bar', ax=axes[0], color=['lightgreen', 'lightcoral'])
    axes[0].set_title('Distribution de PoorCare (Effectifs)')
    axes[0].set_xlabel('PoorCare')
    axes[0].set_ylabel('Nombre de patients')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Graphique en secteurs
    target_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                      colors=['lightgreen', 'lightcoral'])
    axes[1].set_title('Distribution de PoorCare (Pourcentages)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    return target_counts, target_props

def feature_distributions(df):
    """Analyse des distributions des variables explicatives"""
    print("\n📊 6. ANALYSE DES DISTRIBUTIONS DES VARIABLES")
    print("-" * 50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('PoorCare')  # Exclure la variable cible
    
    # Graphiques de distribution
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Histogramme avec courbe de densité
            axes[i].hist(df[col], bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Ajout de la courbe de densité
            df[col].plot.density(ax=axes[i], color='red', linewidth=2)
            
            axes[i].set_title(f'Distribution de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Densité')
            axes[i].grid(True, alpha=0.3)
            
            # Statistiques sur le graphique
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Médiane: {median_val:.2f}')
            axes[i].legend()
    
    # Supprimer les sous-graphiques vides
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Tests de normalité
    print("\n🔬 Tests de normalité (Shapiro-Wilk):")
    print("(p < 0.05 = distribution non normale)")
    for col in numeric_cols:
        if len(df[col]) <= 5000:  # Shapiro-Wilk limité à 5000 observations
            stat, p_value = stats.shapiro(df[col])
            normality = "Normale" if p_value > 0.05 else "Non normale"
            print(f"   {col}: statistique={stat:.4f}, p-value={p_value:.6f} → {normality}")

def outlier_analysis(df):
    """Analyse des valeurs aberrantes"""
    print("\n🔍 7. ANALYSE DES VALEURS ABERRANTES")
    print("-" * 50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('PoorCare')
    
    outlier_summary = []
    
    # Méthode IQR pour chaque variable
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
        
        print(f"📊 {col}:")
        print(f"   Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"   Outliers: {n_outliers} ({outlier_percentage:.2f}%)")
        if n_outliers > 0:
            print(f"   Valeurs: {sorted(outliers.tolist())}")
        print()
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    # Visualisation avec boxplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            bp = axes[i].boxplot(df[col], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            axes[i].set_title(f'Boxplot de {col}')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
    
    # Supprimer les sous-graphiques vides
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return outlier_df

def correlation_analysis(df):
    """Analyse des corrélations"""
    print("\n🔗 8. ANALYSE DES CORRÉLATIONS")
    print("-" * 50)
    
    # Matrice de corrélation
    correlation_matrix = df.corr()
    
    print("📊 Matrice de corrélation (Pearson):")
    print(correlation_matrix.round(3))
    
    # Corrélations avec la variable cible
    print("\n🎯 Corrélations avec PoorCare:")
    target_corr = correlation_matrix['PoorCare'].drop('PoorCare').sort_values(key=abs, ascending=False)
    
    for var, corr in target_corr.items():
        strength = ""
        if abs(corr) > 0.7:
            strength = "Très forte"
        elif abs(corr) > 0.5:
            strength = "Forte"
        elif abs(corr) > 0.3:
            strength = "Modérée"
        elif abs(corr) > 0.1:
            strength = "Faible"
        else:
            strength = "Très faible"
        
        direction = "positive" if corr > 0 else "négative"
        print(f"   {var}: {corr:.3f} ({strength} {direction})")
    
    # Heatmap des corrélations
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    plt.title('Matrice de corrélation (triangle inférieur)')
    plt.tight_layout()
    plt.show()
    
    # Test de significativité des corrélations
    print("\n🔬 Tests de significativité des corrélations avec PoorCare:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('PoorCare')
    
    for col in numeric_cols:
        corr_coef, p_value = pearsonr(df[col], df['PoorCare'])
        significance = "Significative" if p_value < 0.05 else "Non significative"
        print(f"   {col}: r={corr_coef:.3f}, p={p_value:.6f} → {significance}")
    
    return correlation_matrix

def bivariate_analysis(df):
    """Analyse bivariée détaillée"""
    print("\n📈 9. ANALYSE BIVARIÉE (VARIABLES vs PoorCare)")
    print("-" * 50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('PoorCare')
    
    # Comparaison des distributions par classe
    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row, col_idx = i // n_cols, i % n_cols
        
        # Boxplot par classe
        df.boxplot(column=col, by='PoorCare', ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} par PoorCare')
        axes[row, col_idx].set_xlabel('PoorCare')
        axes[row, col_idx].set_ylabel(col)
    
    # Supprimer les sous-graphiques vides
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = i // n_cols, i % n_cols
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    plt.show()
    
    # Tests statistiques
    print("\n🔬 Tests statistiques (Mann-Whitney U):")
    print("(Comparaison des distributions entre PoorCare=0 et PoorCare=1)")
    
    for col in numeric_cols:
        group_0 = df[df['PoorCare'] == 0][col]
        group_1 = df[df['PoorCare'] == 1][col]
        
        statistic, p_value = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
        significance = "Significative" if p_value < 0.05 else "Non significative"
        
        median_0 = group_0.median()
        median_1 = group_1.median()
        
        print(f"   {col}:")
        print(f"      Médiane (PoorCare=0): {median_0:.2f}")
        print(f"      Médiane (PoorCare=1): {median_1:.2f}")
        print(f"      Statistique U: {statistic:.2f}")
        print(f"      p-value: {p_value:.6f} → {significance}")
        print()

def advanced_visualizations(df):
    """Visualisations avancées"""
    print("\n🎨 10. VISUALISATIONS AVANCÉES")
    print("-" * 50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('PoorCare')
    
    # 1. Pairplot avec distinction par classe
    print("📊 Création du pairplot...")
    plt.figure(figsize=(15, 12))
    
    # Sélectionner un sous-ensemble pour la lisibilité
    selected_cols = numeric_cols[:4] + ['PoorCare']  # 4 premières variables + target
    
    sns.pairplot(df[selected_cols], hue='PoorCare', diag_kind='hist', 
                 plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
    plt.suptitle('Pairplot des principales variables', y=1.02)
    plt.show()
    
    # 2. Heatmap des moyennes par classe
    print("📊 Création de la heatmap des moyennes...")
    
    means_by_class = df.groupby('PoorCare')[numeric_cols].mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(means_by_class.T, annot=True, cmap='RdYlBu_r', 
                fmt='.2f', cbar_kws={'label': 'Valeur moyenne'})
    plt.title('Moyennes des variables par classe PoorCare')
    plt.xlabel('PoorCare')
    plt.ylabel('Variables')
    plt.tight_layout()
    plt.show()
    
    # 3. Distribution des variables par classe (violin plots)
    print("📊 Création des violin plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.violinplot(data=df, x='PoorCare', y=col, ax=axes[i])
            axes[i].set_title(f'Distribution de {col} par PoorCare')
            axes[i].grid(True, alpha=0.3)
    
    # Supprimer les sous-graphiques vides
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def generate_report_summary(df, correlation_matrix, outlier_df):
    """Génération du résumé du rapport"""
    print("\n📋 11. RÉSUMÉ EXÉCUTIF DE L'ANALYSE")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('PoorCare')
    
    print("🔍 POINTS CLÉS IDENTIFIÉS:")
    print("-" * 30)
    
    # 1. Qualité des données
    missing_count = df.isnull().sum().sum()
    print(f"✅ Qualité des données: {missing_count} valeurs manquantes")
    
    # 2. Équilibrage des classes
    target_counts = df['PoorCare'].value_counts()
    ratio = target_counts.min() / target_counts.max()
    balance_status = "Équilibré" if ratio > 0.5 else "Déséquilibré" if ratio < 0.3 else "Légèrement déséquilibré"
    print(f"⚖️  Équilibrage des classes: {balance_status} (ratio: {ratio:.3f})")
    
    # 3. Variables les plus corrélées avec la cible
    target_corr = correlation_matrix['PoorCare'].drop('PoorCare')
    strongest_corr = target_corr.abs().idxmax()
    strongest_corr_value = target_corr[strongest_corr]
    print(f"🎯 Variable la plus corrélée: {strongest_corr} (r={strongest_corr_value:.3f})")
    
    # 4. Outliers
    total_outliers = outlier_df['Nb_outliers'].sum()
    outlier_percentage = (total_outliers / (len(df) * len(numeric_cols))) * 100
    print(f"📊 Outliers détectés: {total_outliers} ({outlier_percentage:.2f}% du dataset)")
    
    # 5. Variables avec distributions non normales
    non_normal_vars = []
    for col in numeric_cols:
        if len(df[col]) <= 5000:
            stat, p_value = stats.shapiro(df[col])
            if p_value < 0.05:
                non_normal_vars.append(col)
    
    print(f"📈 Variables non normales: {len(non_normal_vars)}/{len(numeric_cols)}")
    
    print("\n💡 RECOMMANDATIONS:")
    print("-" * 20)
    
    if ratio < 0.5:
        print("• Considérer des techniques de rééquilibrage (SMOTE, sous-échantillonnage)")
    
    if total_outliers > 0:
        print("• Examiner et traiter les valeurs aberrantes si nécessaire")
    
    if len(non_normal_vars) > len(numeric_cols) / 2:
        print("• Envisager des transformations (log, racine carrée) pour normaliser")
    
    if abs(strongest_corr_value) < 0.3:
        print("• Corrélations faibles : explorer feature engineering ou variables additionnelles")
    
    print("• Valider les résultats avec validation croisée")
    print("• Considérer des modèles non-linéaires pour capturer des relations complexes")
    
    print("\n" + "=" * 80)

def main():
    """Fonction principale d'exécution de l'EDA"""
    try:
        # 1. Chargement et informations de base
        df = load_and_basic_info()
        
        # 2. Analyse des valeurs manquantes
        missing_df = missing_values_analysis(df)
        
        # 3. Conversion des types de données
        df = data_types_and_conversion(df)
        
        # 4. Statistiques descriptives
        desc_stats, additional_stats = descriptive_statistics(df)
        
        # 5. Analyse de la variable cible
        target_counts, target_props = target_variable_analysis(df)
        
        # 6. Distributions des variables
        feature_distributions(df)
        
        # 7. Analyse des outliers
        outlier_df = outlier_analysis(df)
        
        # 8. Analyse des corrélations
        correlation_matrix = correlation_analysis(df)
        
        # 9. Analyse bivariée
        bivariate_analysis(df)
        
        # 10. Visualisations avancées
        advanced_visualizations(df)
        
        # 11. Résumé du rapport
        generate_report_summary(df, correlation_matrix, outlier_df)
        
        print("\n🎉 ANALYSE EXPLORATOIRE TERMINÉE AVEC SUCCÈS!")
        print("📊 Toutes les visualisations et analyses ont été générées.")
        
        return df, correlation_matrix, outlier_df
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # Installation des packages requis si nécessaire
    try:
        import plotly
    except ImportError:
        print("📦 Installation de plotly...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'plotly'])
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.figure_factory as ff
    
    # Exécution de l'analyse
    df, correlation_matrix, outlier_df = main()