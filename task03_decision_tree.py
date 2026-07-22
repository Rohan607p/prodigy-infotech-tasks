import pandas as 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
n = 1000

age        = np.random.randint(18, 70, n)
job        = np.random.choice(['admin','blue-collar','entrepreneur','housemaid',
                                'management','retired','self-employed','services',
                                'student','technician','unemployed'], n)
balance    = np.random.randint(-500, 10000, n)
duration   = np.random.randint(0, 600, n)
campaign   = np.random.randint(1, 10, n)
previous   = np.random.randint(0, 5, n)
education  = np.random.choice(['primary','secondary','tertiary'], n)
marital    = np.random.choice(['married','single','divorced'], n)


prob = (
    0.1
    + 0.003 * (age - 18)
    + 0.00002 * np.clip(balance, 0, 10000)
    + 0.0005 * duration
    - 0.02 * campaign
)
prob = np.clip(prob, 0.05, 0.95)
subscribed = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'age': age, 'job': job, 'marital': marital, 'education': education,
    'balance': balance, 'duration': duration, 'campaign': campaign,
    'previous': previous, 'subscribed': subscribed
})

print("=" * 50)
print("BANK MARKETING DATASET")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nSubscription Rate: {df['subscribed'].mean()*100:.1f}%")
print(f"\nSample:\n{df.head()}")


le = LabelEncoder()
for col in ['job', 'marital', 'education']:
    df[col] = le.fit_transform(df[col])

X = df.drop('subscribed', axis=1)
y = df['subscribed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


dt = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=20)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)
print(f"Accuracy: {acc*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No','Yes'])}")


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Task 03 – Decision Tree Classifier (Bank Marketing)', fontsize=16, fontweight='bold')


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No','Yes'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\nAccuracy: {acc*100:.1f}%', fontweight='bold')


importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_fi = ['#4C72B0' if v < importances.max()*0.5 else '#E74C3C' for v in importances]
importances.plot(kind='barh', ax=axes[1], color=colors_fi, edgecolor='white')
axes[1].set_title('Feature Importances', fontweight='bold')
axes[1].set_xlabel('Importance Score')
axes[1].grid(axis='x', alpha=0.3)


plot_tree(dt, feature_names=X.columns, class_names=['No','Yes'],
          filled=True, rounded=True, max_depth=3, ax=axes[2], fontsize=7)
axes[2].set_title('Decision Tree (Top 3 Levels)', fontweight='bold')

plt.tight_layout()
plt.savefig('task03_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTask 03 complete! Output saved as task03_output.png")
