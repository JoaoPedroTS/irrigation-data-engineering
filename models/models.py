from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_models() -> dict:
    return {
        "Logistic Regression" : LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(kernel="rbf", random_state=42, class_weight="balanced")
    }