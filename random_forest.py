from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import get_preprocessed_data


def run_random_forest(X_train, X_test, y_train, y_test, n_estimators=50, max_depth=8):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'accuracy': f"{accuracy_score(y_test, preds) * 100:.2f}%",
        'precision': f"{precision_score(y_test, preds, zero_division=0) * 100:.2f}%",
        'recall': f"{recall_score(y_test, preds, zero_division=0) * 100:.2f}%",
        'f1': f"{f1_score(y_test, preds, zero_division=0) * 100:.2f}%"
    }


if __name__ == "__main__":
    _, X_train, X_test, _, _, yc_train, yc_test, _, _ = get_preprocessed_data()
    metrics = run_random_forest(X_train, X_test, yc_train, yc_test)
    print("Random Forest Metrics:", metrics)
