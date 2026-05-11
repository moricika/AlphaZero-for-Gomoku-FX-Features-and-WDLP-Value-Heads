import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 600})

train = pd.read_csv('13x13_fxwdlp_training.csv')
eval_df = pd.read_csv('13x13_fxwdlp_evaluation.csv')

fig = plt.figure(figsize=(16, 12))

# Self-play
ax1 = plt.subplot(3, 3, 1)
ax1.plot(train['training_steps'], train['num_games_received'], color='tab:blue')
ax1.set_title('Self-play')
ax1.set_ylabel('Number of games')

ax2 = plt.subplot(3, 3, 4)
ax2.plot(train['training_steps'], train['num_transitions_received'] / train['num_games_received'], color='tab:blue')
ax2.set_ylabel('Avg steps per game')

# Training
ax3 = plt.subplot(3, 3, 2)
ax3.plot(train['training_steps'], train['policy_loss'], color='tab:blue')
ax3.set_title('Training')
ax3.set_ylabel('Policy Loss')

ax4 = plt.subplot(3, 3, 5)
ax4.plot(train['training_steps'], train['value_loss'], color='tab:orange', label='Value Loss')
ax4.plot(train['training_steps'], train['wdl_loss'], color='tab:green', label='WDL Loss')
ax4.set_ylabel('Loss')
ax4.legend()

ax5 = plt.subplot(3, 3, 8)
ax5.plot(train['training_steps'], train['plies_loss'], color='tab:purple')
ax5.set_ylabel('Plies Loss')
ax5.set_xlabel('Training steps')

# Evaluation
ax6 = plt.subplot(3, 3, 3)
ax6.plot(eval_df['training_steps'], eval_df['game_length'], color='tab:blue')
ax6.set_title('Evaluation')
ax6.set_ylabel('Avg game length')

ax7 = plt.subplot(3, 3, 6)
ax7.plot(eval_df['training_steps'], eval_df['black_elo_rating'], color='tab:blue', label='Black Elo')
ax7.plot(eval_df['training_steps'], eval_df['white_elo_rating'], color='tab:orange', label='White Elo')
ax7.set_ylabel('Elo ratings')
ax7.legend()

plt.suptitle('13×13 FXWDLP Training Progress', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('13x13_FXWDLP_Progress.png', dpi=600, bbox_inches='tight')
plt.savefig('13x13_FXWDLP_Progress.pdf', bbox_inches='tight')
print("13×13 FXWDLP progress plot saved")
plt.show()