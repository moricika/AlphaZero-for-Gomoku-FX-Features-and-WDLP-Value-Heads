import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

data = {
    'Config': ['A: Baseline', 'B: +FX Only', 'C: +WDLP Only', 'D: +FX+WDLP', 
               'E: +Backfill', 'F: Full (Ours)'],
    'Elo': [150, 280, 250, 420, 480, 519],
    'Delta_Elo': [0, 130, 100, 270, 330, 369],
    'Win_Rate': [50, 62, 58, 72, 78, 82]
}

df = pd.DataFrame(data)

# Figure 1: Elo Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Config'], df['Elo'], color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b'])
plt.ylabel('Final Elo Rating', fontsize=14)
plt.title('Ablation Study: Impact on Playing Strength (Elo)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 8, f'{int(height)}', 
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('Ablation_Elo_Progression.svg', format='svg', bbox_inches='tight')
plt.show()

# Figure 2: Delta Elo + Win Rate
fig, ax1 = plt.subplots(figsize=(11, 6))
x = range(len(df))

ax1.bar([i - 0.2 for i in x], df['Delta_Elo'], 0.4, label='Δ Elo Gain', color='#1f77b4')
ax1.set_ylabel('Elo Gain over Baseline', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Config'], rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(x, df['Win_Rate'], marker='o', linewidth=3, color='#d62728', label='Win Rate (%)')
ax2.set_ylabel('Win Rate (%)', fontsize=14)

plt.title('Ablation Study: Component Contributions\n(Δ Elo Gain and Win Rate)', fontsize=16, pad=20)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Ablation_DeltaElo_WinRate.svg', format='svg', bbox_inches='tight')
plt.show()