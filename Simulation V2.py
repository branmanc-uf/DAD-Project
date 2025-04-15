# Implementing Utility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def initialize_strategic_traders(num_insiders, num_normals, num_noise):
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
            'utility': 0,
            'flagged_history': [],
            'status': 'free',
            'suspended_rounds': 0
        }

    traders = [create_trader(f'INS_{i}', 'insider') for i in range(num_insiders)]
    traders += [create_trader(f'NORM_{i}', 'normal') for i in range(num_normals)]
    traders += [create_trader(f'NOISE_{i}', 'noise') for i in range(num_noise)]
    return pd.DataFrame(traders)

def encode_preferences(df):
    maps = {
        'sector': {'Tech': 1, 'Finance': 2, 'Healthcare': 3},
        'position_size': {'Small': 1, 'Medium': 2, 'Large': 3},
        'frequency': {'Low': 1, 'Medium': 2, 'High': 3},
        'risk': {'Conservative': 1, 'Balanced': 2, 'Aggressive': 3}
    }
    for col, mapping in maps.items():
        df[f'{col}_enc'] = df[col].map(mapping)
    return df

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
        if trader['status'] == 'suspended':
            continue

        is_restricted = restrict_ids is not None and trader['id'] in restrict_ids
        if trader['type'] == 'insider' and trader['utility'] < -100:
            freq = 0.01
        else:
            freq = 0.05 if is_restricted else freq_prob[trader['frequency']]

        vol_min, vol_max = (100, 500) if is_restricted else size_range[trader['position_size']]

        for day in range(days):
            trade = None
            if trader['type'] == 'insider' and any(abs(day - nd) <= 5 for nd in news_days):
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

def calculate_scores(trades, news_days):
    trader_scores = {}
    for trader_id, group in trades.groupby('trader_id'):
        M_time = np.mean([1 / (1 + abs(t - n)) for t in group['day'] for n in news_days])
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

def evaluate_detection(traders, scores, threshold=0.7):
    df = traders.merge(scores, left_on='id', right_on='trader_id', how='left').fillna(0)
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

def run_iterative_simulation(rounds, num_insiders, num_normals, num_noise, flag_top_k):
    traders = initialize_strategic_traders(num_insiders, num_normals, num_noise)
    portfolio_map = {'Conservative': 200_000, 'Balanced': 100_000, 'Aggressive': 50_000}
    traders['portfolio_value'] = traders['risk'].map(portfolio_map)
    traders = encode_preferences(traders)

    all_rounds_metrics = []
    tracking_rows = []

    for r in range(1, rounds + 1):
        print(f"\nROUND {r}")

        restricted_ids = traders[traders['status'] == 'flagged']['id'].tolist()
        trades, news = simulate_trading(traders, restrict_ids=restricted_ids)
        scores = calculate_scores(trades, news)
        results, metrics = evaluate_detection(traders, scores)

        top_k_ids = results.sort_values(by='P_insider', ascending=False).head(flag_top_k)['id'].tolist()

        for i, row in traders.iterrows():
            trader_id = row['id']
            t_type = row['type']
            risk = row['risk']
            was_flagged = trader_id in top_k_ids
            prev_flagged = row['status'] == 'flagged'
            status = 'flagged' if was_flagged else 'previously_flagged' if prev_flagged else 'free'
            P_insider = float(scores[scores['trader_id'] == trader_id]['P_insider']) if trader_id in scores['trader_id'].values else 0

            utility_change = 0
            profit = 0
            traded_group = trades[trades['trader_id'] == trader_id]
            traded_near_news = any(traded_group['news_aligned']) if not traded_group.empty else False

            if t_type == 'insider':
                if was_flagged and traded_near_news:
                    utility_change = -50  # stronger penalty
                elif not was_flagged and traded_near_news:
                    utility_change = +10
            elif t_type == 'normal':
                utility_change = -10 if was_flagged else +5

            for _, trade in traded_group.iterrows():
                volume = trade['volume']
                if trade['news_aligned'] and t_type == 'insider':
                    profit += volume * (0.15 if not was_flagged else -0.25)
                else:
                    profit += volume * random.uniform(-0.05, 0.10)

       
            traders.at[i, 'portfolio_value'] += profit
            traders.at[i, 'utility'] += utility_change
            total_score = traders.at[i, 'utility'] + 0.01 * traders.at[i, 'portfolio_value']
            traders.at[i, 'total_score'] = total_score

            if was_flagged:
                traders.at[i, 'flagged_history'].append(r)

            if row['status'] == 'suspended':
                status = 'suspended'

            traders.at[i, 'status'] = status

            tracking_rows.append({
                'round': r,
                'trader_id': trader_id,
                'type': t_type,
                'status': status,
                'was_flagged': was_flagged,
                'P_insider': P_insider,
                'utility': traders.at[i, 'utility'],
                'utility_change': utility_change,
                'portfolio_value': traders.at[i, 'portfolio_value'],
                'profit_this_round': profit,
                'risk': traders.at[i, 'risk'],
                'total_score': total_score,
                'was_suspended': status == 'suspended'
            })

        print(f"\nRound {r} Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
        all_rounds_metrics.append({'round': r, **metrics})

        for i, row in traders.iterrows():
            if row['utility'] < -150 and row['status'] != 'suspended':
                traders.at[i, 'status'] = 'suspended'
                traders.at[i, 'suspended_rounds'] = 2
            elif row['status'] == 'suspended':
                if row['suspended_rounds'] > 1:
                    traders.at[i, 'suspended_rounds'] -= 1
                else:
                    traders.at[i, 'status'] = 'free'
                    traders.at[i, 'suspended_rounds'] = 0

        for i, row in traders.iterrows():
            hist = row['flagged_history']
            if len(hist) == 0:
                continue
            if row['type'] in ['insider', 'normal'] and hist[-1] == r:
                traders.at[i, 'frequency'] = 'Low'
                traders.at[i, 'position_size'] = 'Small'
            elif row['type'] == 'insider' and hist[-1] < r - 2:
                traders.at[i, 'frequency'] = 'High'
                traders.at[i, 'position_size'] = 'Large'

    return pd.DataFrame(all_rounds_metrics), pd.DataFrame(tracking_rows), traders

def plot_suspensions(tracking_df):
    suspension = tracking_df.groupby('round')['was_suspended'].sum().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=suspension, x='round', y='was_suspended', marker='o')
    plt.title('Number of Suspended Traders Per Round')
    plt.xlabel('Round')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_flagged_vs_clean_profit(tracking_df):
    tracking_df['flagged_label'] = tracking_df['was_flagged'].map({True: 'Flagged', False: 'Clean'})
    plt.figure(figsize=(10, 6))
    sns.histplot(data=tracking_df, x='profit_this_round', hue='flagged_label', bins=50, kde=True, element='step', stat='density')
    plt.title('Profit Distribution: Flagged vs Clean Traders')
    plt.xlabel('Profit This Round')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_risk_adjusted_return(tracking_df):
    tracking_df = tracking_df.sort_values(by=['trader_id', 'round'])
    tracking_df['pct_change'] = tracking_df.groupby('trader_id')['portfolio_value'].pct_change().fillna(0)
    risk_adj = tracking_df.groupby(['round', 'type'])['pct_change'].agg(['mean', 'std']).reset_index()
    risk_adj['sharpe'] = risk_adj['mean'] / (risk_adj['std'] + 1e-6)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=risk_adj, x='round', y='sharpe', hue='type', marker='o')
    plt.title('Risk-Adjusted Return (Sharpe Ratio) Over Time')
    plt.xlabel('Round')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_utility_over_time(tracking_df):
    plt.figure(figsize=(12, 8))
    avg_util = tracking_df.groupby(['round', 'type'])['utility'].mean().reset_index()
    sns.lineplot(data=avg_util, x='round', y='utility', hue='type', marker='o')
    plt.title('Average Utility Over Time by Trader Type')
    plt.xlabel('Round')
    plt.ylabel('Average Utility')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_model_performance_over_time(metrics_df):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['round'], metrics_df['accuracy'], label='Accuracy', marker='o')
    plt.plot(metrics_df['round'], metrics_df['precision'], label='Precision', marker='o')
    plt.plot(metrics_df['round'], metrics_df['recall'], label='Recall', marker='o')
    plt.title('Model Performance Over Time')
    plt.xlabel('Round')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_portfolio_by_risk(tracking_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    avg_by_risk = tracking_df.groupby(['round', 'risk'])['portfolio_value'].mean().reset_index()
    sns.lineplot(data=avg_by_risk, x='round', y='portfolio_value', hue='risk', marker='o')
    plt.title('Average Portfolio Value Over Time by Risk Profile')
    plt.xlabel('Round')
    plt.ylabel('Average Portfolio Value ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pct_growth_by_risk(tracking_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    tracking_df_sorted = tracking_df.sort_values(by=['trader_id', 'round'])
    tracking_df_sorted['pct_change'] = tracking_df_sorted.groupby('trader_id')['portfolio_value'].pct_change().fillna(0)

    avg_pct_by_risk = tracking_df_sorted.groupby(['round', 'risk'])['pct_change'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=avg_pct_by_risk, x='round', y='pct_change', hue='risk', marker='o')
    plt.title('Average % Portfolio Change Per Round by Risk Profile')
    plt.xlabel('Round')
    plt.ylabel('Average % Change')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_portfolio_over_time(tracking_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    avg_portfolio = tracking_df.groupby(['round', 'type'])['portfolio_value'].mean().reset_index()
    sns.lineplot(data=avg_portfolio, x='round', y='portfolio_value', hue='type', marker='o')
    plt.title('Average Portfolio Value Over Time by Trader Type')
    plt.xlabel('Round')
    plt.ylabel('Average Portfolio Value ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_total_score_over_time(tracking_df):
    plt.figure(figsize=(12, 8))
    avg_score = tracking_df.groupby(['round', 'type'])['total_score'].mean().reset_index()
    sns.lineplot(data=avg_score, x='round', y='total_score', hue='type', marker='o')
    plt.title('Average Combined Score (Utility + Scaled Profit) Over Time by Trader Type')
    plt.xlabel('Round')
    plt.ylabel('Average Total Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    metrics_df, tracking_df, final_traders = run_iterative_simulation(
        rounds=24, num_insiders=100, num_normals=100, num_noise=100, flag_top_k=100)
    plot_utility_over_time(tracking_df)
    plot_model_performance_over_time(metrics_df)
    plot_portfolio_over_time(tracking_df)
    plot_portfolio_by_risk(tracking_df)
    plot_pct_growth_by_risk(tracking_df)
    plot_total_score_over_time(tracking_df)
    plot_suspensions(tracking_df)
    plot_flagged_vs_clean_profit(tracking_df)
    plot_risk_adjusted_return(tracking_df)

    tracking_df.to_csv("tracking_utility.csv", index=False)
    final_traders.to_csv("final_trader_profiles.csv", index=False)
    print("\nSimulation complete. Results saved to CSV.")
