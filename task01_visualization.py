import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
n = 500

ages = np.concatenate([
    np.random.normal(25, 5, 150),
    np.random.normal(40, 8, 200),
    np.random.normal(60, 7, 150)
]).astype(int)
ages = np.clip(ages, 18, 80)

genders = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])

df = pd.DataFrame({'Age': ages[:n], 'Gender': genders})

print("Dataset Sample:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nAge Stats:\n{df['Age'].describe()}")
print(f"\nGender Counts:\n{df['Gender'].value_counts()}")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Task 01 – Population Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)


gender_counts = df['Gender'].value_counts()
colors = ['#4C72B0', '#DD8452', '#55A868']
axes[0].bar(gender_counts.index, gender_counts.values, color=colors, edgecolor='white', linewidth=1.2)
axes[0].set_title('Gender Distribution (Bar Chart)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Gender', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
for i, (val) in enumerate(gender_counts.values):
    axes[0].text(i, val + 3, str(val), ha='center', fontweight='bold', fontsize=11)
axes[0].set_ylim(0, max(gender_counts.values) * 1.15)
axes[0].grid(axis='y', alpha=0.3)


axes[1].hist(df['Age'], bins=20, color='#4C72B0', edgecolor='white', linewidth=0.8, alpha=0.85)
axes[1].set_title('Age Distribution (Histogram)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Age', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['Age'].mean():.1f}")
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)


for gender, color in zip(['Male', 'Female', 'Other'], colors):
    subset = df[df['Gender'] == gender]['Age']
    axes[2].hist(subset, bins=15, alpha=0.6, label=gender, color=color, edgecolor='white')
axes[2].set_title('Age Distribution by Gender', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Age', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('task01_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTask 01 complete! Output saved as task01_output.png")
