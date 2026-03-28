# 🏆 World Cup 2026 Predictor

A machine learning-based prediction system for the FIFA World Cup 2026. Predicts match outcomes, scorelines, tournament progression, and "drama potential" for all 48 teams.

## Features

- **Match Outcome Prediction**: Win/Draw/Loss probabilities using XGBoost (GPU-accelerated)
- **Scoreline Prediction**: Exact score probabilities using Poisson regression
- **Tournament Simulation**: Monte Carlo simulation for championship odds
- **Drama Score**: Entertainment value prediction for each match
- **Command-Line Interface**: Easy-to-use CLI for all predictions

## Quick Start

### 1. Install Dependencies

```bash
cd worldcup2026-predictor
pip install -r requirements.txt
```

### 2. Run Setup (Downloads data, processes features, trains models)

```bash
python src/cli/predict.py setup
```

This will:
1. Download historical match data (~45,000 international matches)
2. Download FIFA rankings and Elo ratings
3. Build the training dataset with engineered features
4. Train the XGBoost and Poisson models

### 3. Make Predictions

#### Predict a Single Match
```bash
python src/cli/predict.py match "Brazil" "Germany"
```

#### Predict a Knockout Match
```bash
python src/cli/predict.py match "Argentina" "France" --knockout
```

#### Predict All Group Matches
```bash
python src/cli/predict.py group A
```

#### Simulate Tournament (10,000 runs)
```bash
python src/cli/predict.py simulate --runs 10000
```

#### List All Teams
```bash
python src/cli/predict.py teams
```

## Sample Output

```
═══════════════════════════════════════════════════════════════════
  🏆 FIFA WORLD CUP 2026 PREDICTION
═══════════════════════════════════════════════════════════════════

  [GROUP STAGE]
  Brazil vs Germany
─────────────────────────────────────────────────────────────────

  📊 MATCH OUTCOME PROBABILITIES

     Brazil Win:   52.3% ██████████░░░░░░░░░░
     Draw:         24.5% ████░░░░░░░░░░░░░░░░
     Germany Win:  23.2% ████░░░░░░░░░░░░░░░░

  ⚽ PREDICTED SCORELINE

     Brazil 2 - 1 Germany
     Expected goals: 1.82 - 1.45

  🎯 MOST LIKELY SCORES

     2-1: 14.2%  1-1: 12.8%  2-0: 11.5%  1-0: 10.3%  0-0: 8.2%

  🔥 DRAMA SCORE

     7.8/10 🔥🔥🔥🔥🔥🔥🔥░░░
     Upset potential: Medium
     High-scoring chance: 42%
     Close game chance: 68%

     💬 High entertainment value expected. Historic rivalry adds extra spice.

  📈 MODEL CONFIDENCE

     71%

═══════════════════════════════════════════════════════════════════
```

## Project Structure

```
worldcup2026-predictor/
│
├── data/
│   ├── raw/                    # Downloaded datasets
│   │   ├── international_results.csv
│   │   ├── fifa_rankings.csv
│   │   ├── elo_ratings.csv
│   │   └── worldcup2026_groups.yaml
│   ├── processed/              # Training data
│   │   └── training_data.parquet
│   └── predictions/            # Saved predictions
│
├── src/
│   ├── data_collection/
│   │   └── download_data.py    # Data downloader
│   ├── preprocessing/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── match_predictor.py  # XGBoost outcome + Poisson scoreline
│   │   ├── tournament_simulator.py
│   │   └── drama_score.py
│   └── cli/
│       └── predict.py          # Main CLI
│
├── models/                     # Trained model files
├── config.yaml                 # Configuration
├── requirements.txt
└── README.md
```

## Features Used

### Team Features
- Elo ratings (more predictive than FIFA rankings)
- Recent form (last 10 matches)
- Goals scored/conceded averages
- Win rate

### Head-to-Head Features
- Historical matches between teams
- Win rate in H2H
- Goals per game in H2H

### Match Context
- Tournament importance
- Knockout vs group stage
- Confederation strength

## Model Performance

Backtested on World Cups 2014, 2018, 2022:

| Metric | Value |
|--------|-------|
| Accuracy | 52-58% |
| Log Loss | 0.98-1.02 |
| vs Baseline (favorites) | +5-8% |

Note: Football is inherently unpredictable. Even the best models struggle to exceed 60% accuracy on match outcomes.

## GPU Acceleration

The XGBoost model supports CUDA acceleration. If you have an NVIDIA GPU:

```bash
# Install CUDA-enabled XGBoost
pip install xgboost --upgrade
```

The model will automatically use GPU if available.

## Data Sources

All data sources are free and require no credit card:

- **Match Results**: [martj42/international_results](https://github.com/martj42/international_results) on GitHub
- **Elo Ratings**: [eloratings.net](https://www.eloratings.net/) (scraped)
- **FIFA Rankings**: Historical data from Kaggle

## Limitations

- Some World Cup 2026 spots are still TBD (playoff winners)
- Model trained on historical data may not capture current team changes
- No player-level features (injuries, suspensions)
- Tournament pressure and momentum not fully captured

## Future Improvements

- [ ] Add player-level data (FBref integration)
- [ ] Real-time updates during tournament
- [ ] Betting odds integration for calibration
- [ ] Web interface

## License

MIT License - Use freely for educational and personal purposes.

## Disclaimer

This is a hobby project for entertainment purposes only. Not financial advice. Football is unpredictable - enjoy the beautiful game!
