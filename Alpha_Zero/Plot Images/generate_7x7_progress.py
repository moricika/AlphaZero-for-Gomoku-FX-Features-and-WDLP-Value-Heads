import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 20, 'figure.dpi': 600})

train = pd.read_csv('training.csv')
eval_df = pd.read_csv('evaluation.csv')

# 1. Number of Games (Self-play)
plt.figure(figsize=(8, 5))
plt.plot(train['training_steps'], train['num_games_received'], color='tab:blue', linewidth=2.5)
plt.title('Self-play: Number of Games')
plt.xlabel('Training Steps')
plt.ylabel('Number of Games')
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Selfplay_NumGames.png', dpi=600, bbox_inches='tight')
plt.close()

# 2. Average Steps per Game
plt.figure(figsize=(8, 5))
plt.plot(train['training_steps'], train['num_transitions_received'] / train['num_games_received'], 
         color='tab:blue', linewidth=2.5)
plt.title('Self-play: Average Steps per Game')
plt.xlabel('Training Steps')
plt.ylabel('Avg Steps per Game')
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Selfplay_AvgSteps.png', dpi=600, bbox_inches='tight')
plt.close()

# 3. Policy Loss
plt.figure(figsize=(8, 5))
plt.plot(train['training_steps'], train['policy_loss'], color='tab:blue', linewidth=2.5)
plt.title('Training: Policy Loss')
plt.xlabel('Training Steps')
plt.ylabel('Policy Loss')
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Training_PolicyLoss.png', dpi=600, bbox_inches='tight')
plt.close()

# 4. Value Loss vs WDL Loss
plt.figure(figsize=(8, 5))
plt.plot(train['training_steps'], train['value_loss'], color='tab:orange', linewidth=2.5, label='Value Loss')
plt.plot(train['training_steps'], train['wdl_loss'], color='tab:green', linewidth=2.5, label='WDL Loss')
plt.title('Training: Value Loss vs WDL Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Training_Value_vs_WDL.png', dpi=600, bbox_inches='tight')
plt.close()

# 5. Plies Loss
plt.figure(figsize=(8, 5))
plt.plot(train['training_steps'], train['plies_loss'], color='tab:purple', linewidth=2.5)
plt.title('Training: Plies Loss')
plt.xlabel('Training Steps')
plt.ylabel('Plies Loss')
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Training_PliesLoss.png', dpi=600, bbox_inches='tight')
plt.close()

# 6. Average Game Length (Evaluation)
plt.figure(figsize=(8, 5))
plt.plot(eval_df['training_steps'], eval_df['game_length'], color='tab:blue', linewidth=2.5)
plt.title('Evaluation: Average Game Length')
plt.xlabel('Training Steps')
plt.ylabel('Avg Game Length')
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Eval_GameLength.png', dpi=600, bbox_inches='tight')
plt.close()

# 7. Elo Ratings (Evaluation)
plt.figure(figsize=(8, 5))
plt.plot(eval_df['training_steps'], eval_df['black_elo_rating'], color='tab:blue', linewidth=2.5, label='Black Elo')
plt.plot(eval_df['training_steps'], eval_df['white_elo_rating'], color='tab:orange', linewidth=2.5, label='White Elo')
plt.title('Evaluation: Elo Rating Progression')
plt.xlabel('Training Steps')
plt.ylabel('Elo Rating')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('7x7_Eval_Elo.png', dpi=600, bbox_inches='tight')
plt.close()

print("✅ All 7 separate figures saved successfully at 600 DPI!")