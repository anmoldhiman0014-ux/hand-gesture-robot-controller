"""
train_model.py - Train SVM Gesture Classifier
Run after collect_data.py to train and save the model.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os

GESTURES = {0: "STOP", 1: "FORWARD", 2: "BACKWARD", 3: "LEFT", 4: "RIGHT", 5: "ROTATE"}
DATA_FILE = "gesture_data.csv"
MODEL_DIR = "models"

def train():
    print("\n🧠 GESTURE MODEL TRAINER")
    print("=" * 40)

    if not os.path.exists(DATA_FILE):
        print("❌ No data found! Run collect_data.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df)} samples")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Class distribution
    print("\n📊 Class Distribution:")
    for label, name in GESTURES.items():
        count = np.sum(y == label)
        print(f"   {name}: {count} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n⚙️  Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_s, y_train)

    accuracy = svm.score(X_test_s, y_test)
    cv_scores = cross_val_score(svm, scaler.transform(X), y, cv=5)

    print(f"\n✅ Test Accuracy:  {accuracy*100:.2f}%")
    print(f"✅ CV Accuracy:    {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    y_pred = svm.predict(X_test_s)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[GESTURES[i] for i in range(6)]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([GESTURES[i] for i in range(6)], rotation=45, color='white')
    ax.set_yticklabels([GESTURES[i] for i in range(6)], color='white')
    ax.set_title('Gesture Classifier — Confusion Matrix', color='#00d4ff', fontsize=14, pad=15)
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('True', color='white')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] < cm.max()/2 else 'black', fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()

    joblib.dump(svm, os.path.join(MODEL_DIR, 'gesture_svm.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    print(f"\n💾 Model saved → {MODEL_DIR}/gesture_svm.pkl")
    print(f"💾 Scaler saved → {MODEL_DIR}/scaler.pkl")
    print(f"📊 Plot saved  → {MODEL_DIR}/confusion_matrix.png")
    print("\n🚀 Ready! Run main.py to start the robot controller.")

if __name__ == "__main__":
    train()
