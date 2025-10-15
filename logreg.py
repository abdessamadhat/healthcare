# Logreg.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
 
# 1) Charger les données
data = pd.read_csv("Quality.csv")
 
# 2) Features / target
feature_cols = [
    "ERVisits",
    "OfficeVisits",
    "Narcotics",
    "ProviderCount",
    "NumberClaims",
    "StartedOnCombination",
]
data["StartedOnCombination"] = data["StartedOnCombination"].astype(int)
X = data[feature_cols]
y = data["PoorCare"].astype(int)
 
# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)
 
# 4) Pipeline (scaling + logreg)
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            solver="liblinear", max_iter=3000, class_weight="balanced"
        )),
    ]
)
 
# 5) Fit
pipe.fit(X_train, y_train)
 
# 6) Sauvegarder modèle + ordre des features
to_save = {"model": pipe, "features": feature_cols}
with open("model_A.pkl", "wb") as f:
    pickle.dump(to_save, f)
print("Modèle entraîné et sauvegardé dans model_A.pkl")