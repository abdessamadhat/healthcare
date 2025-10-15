# 🏥 Healthcare Analytics & Prediction Platform

Une application Streamlit complète pour l'analyse exploratoire des données (EDA) et la prédiction de la qualité des soins de santé utilisant l'apprentissage automatique.

## 📋 Description

Cette plateforme offre une interface web interactive pour analyser les données de qualité des soins de santé et prédire les risques de mauvaise qualité de soins (PoorCare) en utilisant un modèle de régression logistique.

## ✨ Fonctionnalités

### 🔍 Analyse Exploratoire des Données (EDA)
- **Vue d'ensemble** : Métriques clés et aperçu des données
- **Statistiques descriptives** : Analyse complète avec visualisations interactives
- **Analyse des corrélations** : Matrice de corrélation et relations entre variables
- **Détection d'outliers** : Identification des valeurs aberrantes avec méthode IQR
- **Analyse bivariée** : Comparaisons statistiques par classe cible

### 🎯 Performance du Modèle
- **Métriques de performance** : Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Matrice de confusion** interactive avec métriques dérivées
- **Courbes ROC et Précision-Rappel** avec seuil personnalisable
- **Validation croisée** stratifiée avec intervalles de confiance
- **Distribution des probabilités** par classe
- **Recommandations d'amélioration** automatiques

### 🔮 Prédiction en Temps Réel
- Interface intuitive pour saisir les paramètres patient
- Seuil de décision personnalisable
- Gauge de probabilité animée
- Comparaison avec les percentiles du dataset
- Détails techniques du modèle

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy
```

### Entraînement du modèle
```bash
python logreg.py
```

### Lancement de l'application
```bash
streamlit run healthcare_complete.py
```

L'application sera accessible à : http://localhost:8501

## 📊 Dataset

Le dataset `Quality.csv` contient les variables suivantes :

- **ERVisits** : Nombre de visites aux urgences
- **OfficeVisits** : Nombre de visites au cabinet médical  
- **Narcotics** : Consommation de narcotiques
- **ProviderCount** : Nombre de prestataires de soins
- **NumberClaims** : Nombre de réclamations
- **StartedOnCombination** : Traitement combiné (0/1)
- **PoorCare** : Variable cible - Qualité des soins (0=Bonne, 1=Mauvaise)

## 🏗️ Structure du Projet

```
├── healthcare_complete.py   # Application Streamlit principale
├── logreg.py               # Script d'entraînement du modèle
├── eda_quality.py          # Script EDA standalone  
├── Quality.csv             # Dataset
├── model_A.pkl             # Modèle entraîné
├── README.md               # Documentation
└── .gitignore              # Fichiers à ignorer
```

## 🔧 Technologies Utilisées

- **Streamlit** : Interface web interactive
- **scikit-learn** : Modèle de machine learning
- **Pandas/NumPy** : Manipulation des données
- **Matplotlib/Seaborn** : Visualisations statiques
- **Plotly** : Visualisations interactives
- **SciPy** : Tests statistiques

## 📈 Performances du Modèle

Le modèle de régression logistique utilise :
- **Préprocessing** : StandardScaler pour normalisation
- **Algorithme** : Régression Logistique avec solver liblinear
- **Équilibrage** : Class weight balanced
- **Validation** : Train/test split stratifié (75/25)

## 🎨 Captures d'Écran

### Interface Principale
![Interface](screenshots/interface.png)

### Analyse EDA
![EDA](screenshots/eda.png)

### Performance du Modèle
![Performance](screenshots/performance.png)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📧 Contact

**Développeur** : HATIM
**Email** : abdessamadhatim2004@gmail.com

## 🙏 Remerciements

- Dataset de qualité des soins de santé
- Communauté Streamlit
- Équipe scikit-learn
- Contributeurs open source

---

⭐ **N'hésitez pas à mettre une étoile si ce projet vous aide !**