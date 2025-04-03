import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

# STEP 1: Initialize traders
def initialize_traders(num_insiders=10, num_normals=10, num_noise=10):
    sectors = ['Tech', 'Finance', 'Healthcare']
    risk_levels = ['Conservative', 'Balanced', 'Aggressive']
    frequencies = ['Low', 'Medium', 'High']
    sizes = ['Small', 'Medium', 'Large']

    def create_trader(tid, ttype):
        return {
            'id': tid,
            'type': ttype,
            'sector': random.choice(sectors),
            'position_size': random.choice(sizes),
            'frequency': random.choice(frequencies),
            'risk': random.choice(risk_levels),
        }

    traders = []
    for i in range(num_insiders):
        traders.append(create_trader(f'INS_{i}', 'insider'))
    for i in range(num_normals):
        traders.append(create_trader(f'NORM_{i}', 'normal'))
    for i in range(num_noise):
        traders.append(create_trader(f'NOISE_{i}', 'noise'))

    return pd.DataFrame(traders)

# STEP 2: Encode preferences
def encode_preferences(df):
    sector_map = {'Tech': 1, 'Finance': 2, 'Healthcare': 3}
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    freq_map = {'Low': 1, 'Medium': 2, 'High': 3}
    risk_map = {'Conservative': 1, 'Balanced': 2, 'Aggressive': 3}

    df['sector_enc'] = df['sector'].map(sector_map)
    df['size_enc'] = df['position_size'].map(size_map)
    df['freq_enc'] = df['frequency'].map(freq_map)
    df['risk_enc'] = df['risk'].map(risk_map)
    return df

# STEP 3: Simulate trades (with optional restriction list)
def simulate_trading(df, days=30, restrict_ids=None):
    news_days = random.sample(range(5, days - 5), 5)
    trades = []

    freq_prob = {'Low': 0.1, 'Medium': 0.4, 'High': 0.8}
    size_range = {
        'Small': (100, 1000),
        'Medium': (1000, 3000),
        'Large': (3000, 5000)
    }

    for _, trader in df.iterrows():
        is_restricted = restrict_ids is not None and trader['id'] in restrict_ids
        freq = 0.05 if is_restricted else freq_prob[trader['frequency']]
        vol_min, vol_max = (100, 500) if is_restricted else size_range[trader['position_size']]

        for day in range(days):
            trade = None
            if trader['type'] == 'insider' and any(abs(day - nd) <= 1 for nd in news_days):
                trade = {'day': day, 'volume': random.randint(vol_min, vol_max), 'news_aligned': True}
            elif trader['type'] == 'normal':
                if random.random() < freq:
                    trade = {'day': day, 'volume': random.randint(vol_min, vol_max), 'news_aligned': False}
            elif trader['type'] == 'noise':
                if random.random() < 0.3:
                    trade = {'day': day, 'volume': random.randint(100, 5000), 'news_aligned': False}

            if trade:
                trade['trader_id'] = trader['id']
                trade['type'] = trader['type']
                trades.append(trade)

    return pd.DataFrame(trades), news_days

# STEP 4: Score traders
def calculate_scores(trades, news_days):
    trader_scores = {}
    for trader_id, group in trades.groupby('trader_id'):
        M_time = np.mean([np.exp(-0.5 * min([abs(t - n) for n in news_days])) for t in group['day']])
        M_volume = (group['volume'].max() - group['volume'].mean()) / (group['volume'].std() + 1e-5)
        M_pattern = sum(group['news_aligned']) / len(group)
        P_insider = 0.4 * M_time + 0.3 * M_volume + 0.3 * M_pattern
        trader_scores[trader_id] = {
            'M_time': M_time,
            'M_volume': M_volume,
            'M_pattern': M_pattern,
            'P_insider': P_insider
        }
    return pd.DataFrame.from_dict(trader_scores, orient='index').reset_index().rename(columns={'index': 'trader_id'})

# STEP 5: Evaluate model performance
def evaluate_detection(traders, scores, threshold=0.7):
    df = traders.merge(scores, left_on='id', right_on='trader_id')
    df['predicted_insider'] = df['P_insider'] > threshold
    df['actual_insider'] = df['type'] == 'insider'

    TP = sum(df['predicted_insider'] & df['actual_insider'])
    FP = sum(df['predicted_insider'] & ~df['actual_insider'])
    TN = sum(~df['predicted_insider'] & ~df['actual_insider'])
    FN = sum(~df['predicted_insider'] & df['actual_insider'])

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    return df, {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'accuracy': accuracy, 'precision': precision, 'recall': recall
    }

# STEP 6: Run both simulations and compare
if __name__ == "__main__":
    NUM_INSIDERS = 10

    # === First Run (Unrestricted) ===
    traders = initialize_traders(num_insiders=NUM_INSIDERS)
    traders = encode_preferences(traders)
    trades_1, news_1 = simulate_trading(traders)
    scores_1 = calculate_scores(trades_1, news_1)
    results_1, metrics_1 = evaluate_detection(traders, scores_1)

    print("\nTop 10 Traders Before Restrictions:")
    top_10 = results_1.sort_values(by='P_insider', ascending=False).head(10)
    print(top_10[['id', 'type', 'P_insider']])

    # === Second Run (Restricted for top 10) ===
    restricted_ids = top_10['id'].tolist()
    trades_2, news_2 = simulate_trading(traders, restrict_ids=restricted_ids)
    scores_2 = calculate_scores(trades_2, news_2)
    results_2, metrics_2 = evaluate_detection(traders, scores_2)

    # === Output Results ===
    print("\nMetrics Before Restriction:")
    for k, v in metrics_1.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nMetrics After Restricting Top 10 Traders:")
    for k, v in metrics_2.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print(metrics_1)
print(metrics_2)