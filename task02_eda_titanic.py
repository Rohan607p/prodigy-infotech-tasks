import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = sns.load_dataset('titanic')
print("=" * 50)
print("TITANIC DATASET - INITIAL INFO")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nData Types:\n{df.dtypes}")


print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)


df['age'].fillna(df['age'].median(), inplace=True)

df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

df.drop(columns=['deck'], inplace=True)

print(f"Missing values after cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nSurvival Rate: {df['survived'].mean()*100:.1f}%")
print(f"\nBasic Stats:\n{df[['age','fare','survived']].describe().round(2)}")


fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Task 02 – Titanic Dataset EDA', fontsize=16, fontweight='bold')


survival_map = {0: 'Did not survive', 1: 'Survived'}
df['survived_label'] = df['survived'].map(survival_map)
counts = df['survived_label'].value_counts()
colors = ['#E74C3C', '#2ECC71']
axes[0,0].bar(counts.index, counts.values, color=colors, edgecolor='white')
axes[0,0].set_title('Survival Count', fontweight='bold')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v+5, str(v), ha='center', fontweight='bold')
axes[0,0].grid(axis='y', alpha=0.3)

survival_gender = df.groupby('sex')['survived'].mean() * 100
axes[0,1].bar(survival_gender.index, survival_gender.values,
              color=['#3498DB', '#E91E8C'], edgecolor='white')
axes[0,1].set_title('Survival Rate by Gender (%)', fontweight='bold')
axes[0,1].set_ylabel('Survival Rate (%)')
for i, v in enumerate(survival_gender.values):
    axes[0,1].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0,1].set_ylim(0, 100)
axes[0,1].grid(axis='y', alpha=0.3)


survival_class = df.groupby('pclass')['survived'].mean() * 100
axes[0,2].bar([f'Class {c}' for c in survival_class.index],
              survival_class.values, color=['#F39C12','#8E44AD','#16A085'], edgecolor='white')
axes[0,2].set_title('Survival Rate by Passenger Class (%)', fontweight='bold')
axes[0,2].set_ylabel('Survival Rate (%)')
for i, v in enumerate(survival_class.values):
    axes[0,2].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0,2].set_ylim(0, 100)
axes[0,2].grid(axis='y', alpha=0.3)


axes[1,0].hist(df[df['survived']==0]['age'], bins=20, alpha=0.7,
               label='Did not survive', color='#E74C3C', edgecolor='white')
axes[1,0].hist(df[df['survived']==1]['age'], bins=20, alpha=0.7,
               label='Survived', color='#2ECC71', edgecolor='white')
axes[1,0].set_title('Age Distribution by Survival', fontweight='bold')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Count')
axes[1,0].legend()
axes[1,0].grid(axis='y', alpha=0.3)


axes[1,1].hist(df['fare'].clip(upper=300), bins=30, color='#4C72B0',
               edgecolor='white', alpha=0.85)
axes[1,1].set_title('Fare Distribution', fontweight='bold')
axes[1,1].set_xlabel('Fare (£)')
axes[1,1].set_ylabel('Count')
axes[1,1].axvline(df['fare'].median(), color='red', linestyle='--',
                  label=f"Median: £{df['fare'].median():.1f}")
axes[1,1].legend()
axes[1,1].grid(axis='y', alpha=0.3)


num_cols = df[['survived','pclass','age','sibsp','parch','fare']].corr()
sns.heatmap(num_cols, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1,2], linewidths=0.5, center=0)
axes[1,2].set_title('Correlation Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig('task02_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTask 02 complete! Output saved as task02_output.png")
