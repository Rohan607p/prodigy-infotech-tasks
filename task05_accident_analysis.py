import pandas as pd
import numpy as np
import matplotlib.pyplot as 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 2000

states = ['CA','TX','FL','NY','PA','OH','IL','GA','NC','MI','VA','NJ','WA','AZ','MA','TN','IN','MD','MO','CO']
state_weights = [0.15,0.12,0.09,0.08,0.06,0.05,0.05,0.04,0.04,0.04,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.02,0.02]

weather_conditions = ['Clear','Cloudy','Rain','Snow','Fog','Thunderstorm','Hail','Windy']
weather_weights = [0.35,0.25,0.20,0.07,0.06,0.04,0.02,0.01]

road_conditions = ['Dry','Wet','Icy','Snow-covered','Muddy']
road_weights = [0.45,0.30,0.12,0.09,0.04]

severity = np.random.choice([1,2,3,4], n, p=[0.10,0.50,0.30,0.10])

raw_p = np.array([0.02,0.01,0.01,0.01,0.02,0.03,0.06,0.08,0.06,0.05,0.05,0.05,0.05,0.05,0.05,0.06,0.07,0.08,0.06,0.05,0.04,0.03,0.03,0.02])
raw_p = raw_p / raw_p.sum()
hours = np.random.choice(range(24), n, p=raw_p)

months = np.random.choice(range(1,13), n)
state_col = np.random.choice(states, n, p=state_weights)
weather = np.random.choice(weather_conditions, n, p=weather_weights)
road = np.random.choice(road_conditions, n, p=road_weights)
visibility = np.random.uniform(0, 10, n).round(1)
temp_f = np.random.normal(60, 20, n).round(1)
lat = np.random.uniform(25, 48, n).round(4)
lng = np.random.uniform(-125, -67, n).round(4)

month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
time_of_day = pd.cut(hours, bins=[-1,5,11,17,20,23],
    labels=['Night(0-5)','Morning(6-11)','Afternoon(12-17)','Evening(18-20)','Night(21-23)'])

df = pd.DataFrame({
    'Severity': severity, 'State': state_col, 'Weather_Condition': weather,
    'Road_Condition': road, 'Hour': hours, 'Month': months,
    'Month_Name': [month_names[m] for m in months], 'Time_of_Day': time_of_day,
    'Visibility': visibility, 'Temperature_F': temp_f, 'Start_Lat': lat, 'Start_Lng': lng
})

print("="*50)
print("US TRAFFIC ACCIDENT DATASET SUMMARY")
print("="*50)
print(f"Total Records   : {len(df)}")
print(f"States covered  : {df['State'].nunique()}")
print(f"Avg Severity    : {df['Severity'].mean():.2f} / 4")
print(f"\nTop 5 States:\n{df['State'].value_counts().head()}")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Task 05 – US Traffic Accident Pattern Analysis', fontsize=16, fontweight='bold')

hour_counts = df.groupby('Hour').size()
colors_hour = ['#E74C3C' if h in [7,8,9,16,17,18] else '#4C72B0' for h in hour_counts.index]
axes[0,0].bar(hour_counts.index, hour_counts.values, color=colors_hour, edgecolor='white')
axes[0,0].set_title('Accidents by Hour of Day\n(Red = Rush Hours)', fontweight='bold')
axes[0,0].set_xlabel('Hour (24h)')
axes[0,0].set_ylabel('Number of Accidents')
axes[0,0].set_xticks(range(0,24,2))
axes[0,0].grid(axis='y', alpha=0.3)

weather_counts = df['Weather_Condition'].value_counts()
colors_w = ['#2ECC71','#3498DB','#E74C3C','#9B59B6','#F39C12','#1ABC9C','#E91E63','#FF5722']
axes[0,1].barh(weather_counts.index, weather_counts.values, color=colors_w[:len(weather_counts)], edgecolor='white')
axes[0,1].set_title('Accidents by Weather Condition', fontweight='bold')
axes[0,1].set_xlabel('Number of Accidents')
axes[0,1].grid(axis='x', alpha=0.3)

sev_counts = df['Severity'].value_counts().sort_index()
sev_colors = ['#2ECC71','#F39C12','#E67E22','#E74C3C']
sev_labels = [f'Severity {i}\n({c})' for i,c in zip(sev_counts.index, sev_counts.values)]
axes[0,2].pie(sev_counts.values, labels=sev_labels, colors=sev_colors, autopct='%1.1f%%', startangle=90, pctdistance=0.75)
axes[0,2].set_title('Accident Severity Distribution', fontweight='bold')

month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly = df.groupby('Month_Name').size().reindex(month_order)
axes[1,0].plot(month_order, monthly.values, marker='o', color='#4C72B0', linewidth=2.5, markersize=8, markerfacecolor='#E74C3C')
axes[1,0].fill_between(range(12), monthly.values, alpha=0.15, color='#4C72B0')
axes[1,0].set_title('Monthly Accident Trend', fontweight='bold')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('Number of Accidents')
axes[1,0].set_xticks(range(12))
axes[1,0].set_xticklabels(month_order, rotation=45)
axes[1,0].grid(alpha=0.3)

road_sev = df.groupby(['Road_Condition','Severity']).size().unstack(fill_value=0)
road_sev.plot(kind='bar', ax=axes[1,1], color=['#2ECC71','#F39C12','#E67E22','#E74C3C'], edgecolor='white', rot=25)
axes[1,1].set_title('Road Condition vs Severity', fontweight='bold')
axes[1,1].set_ylabel('Count')
axes[1,1].legend(title='Severity', fontsize=9)
axes[1,1].grid(axis='y', alpha=0.3)

top_states = df['State'].value_counts().head(10).index
state_weather = pd.crosstab(df[df['State'].isin(top_states)]['State'], df[df['State'].isin(top_states)]['Weather_Condition'])
sns.heatmap(state_weather, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,2], linewidths=0.5)
axes[1,2].set_title('Accident Hotspot: State × Weather', fontweight='bold')

plt.tight_layout()
plt.savefig('task05_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTask 05 complete! Output saved as task05_output.png")
print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)
print(f"Peak accident hour     : {hour_counts.idxmax()}:00")
print(f"Most dangerous weather : {df['Weather_Condition'].value_counts().idxmax()}")
print(f"Most accidents in      : {df['State'].value_counts().idxmax()}")
print(f"Worst road condition   : {df[df['Severity']==4]['Road_Condition'].value_counts().idxmax()}")
