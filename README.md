# Insider Trading Simulation

This repository contains a Python-based simulation of strategic insider trading behavior in financial markets, evaluating detection mechanisms and trader performance over multiple rounds. The simulation models three trader types—insiders, normal traders, and noise traders—across configurable parameters, then analyzes their trading patterns relative to news events to flag potential insiders.

## Features

- **Trader Initialization**: Randomly generate insider, normal, and noise traders with attributes such as sector, position size, trading frequency, and risk tolerance.
- **Preference Encoding**: Convert categorical trader attributes into numerical encodings for downstream analysis.
- **Trading Simulation**: Simulate trading activity over a specified number of days, incorporating news-aligned insider trades and restricted trading behavior for flagged/suspended traders.
- **Scoring & Detection**: Compute insider probability scores (`P_insider`) using time, volume, and pattern metrics, then evaluate detection performance (accuracy, precision, recall) against a threshold.
- **Iterative Adaptation**: Run multiple rounds of simulation, dynamically adjusting flagging thresholds and suspension logic based on past performance and trader utility.
- **Visualization**: Multiple plotting functions to visualize model performance, utility trends, portfolio growth, suspension counts, profit distributions, and risk-adjusted returns over time.
- **Outputs**: Save simulation results, including round metrics and detailed trader tracking, to CSV files.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/insider-trading-simulation.git
   cd insider-trading-simulation
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Requirements*: numpy, pandas, matplotlib, seaborn

## Usage

1. **Run the simulation**:

   ```bash
   python main.py
   ```

   By default, this runs `run_iterative_simulation` with 24 rounds and 100 traders of each type. Modify parameters at the bottom of `main.py` as needed.

2. **View plots**: The script generates a series of plots for:

   - Average utility over time by trader type
   - Model performance metrics (accuracy, precision, recall)
   - Average portfolio value and % growth by risk profile
   - Suspension counts per round
   - Profit distribution: flagged vs. clean traders
   - Risk-adjusted returns (Sharpe ratios)

3. **Check outputs**:

   - `tracking_utility.csv`: Detailed per-round tracking of trader status, utility, portfolio value, and profits.
   - `final_trader_profiles.csv`: Final state of all trader profiles after simulation completion.

## Project Structure

```
insider-trading-simulation/
├── main.py                # Entry point, runs simulation and plotting
├── requirements.txt       # Python dependencies
├── tracking_utility.csv   # Generated: per-round tracking data
├── final_trader_profiles.csv  # Generated: final trader profiles
└── README.md              # Project overview and instructions
```

## Core Functions

- **initialize\_strategic\_traders(num\_insiders, num\_normals, num\_noise)**
- **encode\_preferences(df)**
- **simulate\_trading(df, days, restrict\_ids)**
- **calculate\_scores(trades, news\_days)**
- **evaluate\_detection(traders, scores, threshold)**
- **run\_iterative\_simulation(rounds, num\_insiders, num\_normals, num\_noise, initial\_flag\_k)**
- **plot\_...**: Various plotting functions for visualization

Refer to inline docstrings for detailed parameter and return value descriptions.

## Configuration

- Adjust `rounds`, `num_insiders`, `num_normals`, `num_noise`, and `initial_flag_k` in `main.py` to customize simulation scale.
- Modify suspension logic, flagging rules, and utility/penalty parameters directly in the code to experiment with different enforcement strategies.

## Contributing

Contributions welcome! Please open issues or pull requests to suggest new features, fix bugs, or improve documentation.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

