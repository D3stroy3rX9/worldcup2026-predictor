"""
Match Outcome Prediction Model
Predicts Win/Draw/Loss probabilities for matches
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

console = Console()


@dataclass
class PredictionResult:
    """Structured prediction output."""
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str
    confidence: float


class MatchOutcomeModel:
    """
    XGBoost-based model for predicting match outcomes.
    Outputs: Home Win / Draw / Away Win probabilities
    """
    
    # Features to use for prediction (excluding metadata and targets)
    FEATURE_COLUMNS = [
        'home_elo', 'away_elo', 'elo_diff', 'elo_sum',
        'home_form_points_avg', 'away_form_points_avg', 'form_diff',
        'home_form_goals_scored_avg', 'away_form_goals_scored_avg',
        'home_form_goals_conceded_avg', 'away_form_goals_conceded_avg',
        'home_form_goal_diff_avg', 'away_form_goal_diff_avg',
        'home_form_win_rate', 'away_form_win_rate',
        'goals_scored_diff', 'goals_conceded_diff',
        'h2h_matches', 'h2h_win_rate_team1',
        'h2h_goals_avg_team1', 'h2h_goals_avg_team2',
        'tournament_importance', 'is_world_cup', 'is_knockout',
        'same_confederation', 'home_conf_strength', 'away_conf_strength',
    ]
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and XGBOOST_AVAILABLE
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
    def _get_model(self):
        """Create the XGBoost model with appropriate settings."""
        if XGBOOST_AVAILABLE:
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softprob',
                'num_class': 3,
                'random_state': 42,
                'n_jobs': -1,
            }
            
            # Enable GPU if available and requested
            if self.use_gpu:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
                console.print("  🎮 Using GPU acceleration (CUDA)")
            
            return xgb.XGBClassifier(**params)
        else:
            console.print("  ⚠️ XGBoost not available, using GradientBoosting", style="yellow")
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features from dataframe."""
        # Use only available feature columns
        available_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        self.feature_columns = available_cols
        
        X = df[available_cols].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        return X.values
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the model on historical data.
        
        Args:
            df: Training dataframe with features and outcome
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        console.print("\n[bold blue]🎯 Training Match Outcome Model[/bold blue]")
        
        # Prepare data
        X = self.prepare_features(df)
        y = df['outcome_code'].values
        
        console.print(f"  Features: {len(self.feature_columns)}")
        console.print(f"  Samples: {len(X):,}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-based split (more realistic for time series)
        # Use the last test_size proportion as test set
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        console.print(f"  Training samples: {len(X_train):,}")
        console.print(f"  Test samples: {len(X_test):,}")
        
        # Train model
        console.print("  Training XGBoost classifier...")
        self.model = self._get_model()
        
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            if 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                console.print("  ⚠️ GPU failed, falling back to CPU", style="yellow")
                self.use_gpu = False
                self.model = self._get_model()
                self.model.fit(X_train, y_train)
            else:
                raise e
        
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'test_samples': len(y_test),
            'train_samples': len(y_train),
        }
        
        console.print(f"\n  [bold green]✅ Training complete![/bold green]")
        console.print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        console.print(f"  Log Loss: {logloss:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            console.print("\n  Top 10 features:")
            for i, row in importance.head(10).iterrows():
                console.print(f"    {row['feature']}: {row['importance']:.4f}")
            
            metrics['feature_importance'] = importance.to_dict('records')
        
        return metrics
    
    def predict(self, home_team: str, away_team: str, 
                features: Dict[str, float]) -> PredictionResult:
        """
        Predict match outcome.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team  
            features: Dictionary of match features
            
        Returns:
            PredictionResult with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create feature vector
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        probas = self.model.predict_proba(X_scaled)[0]
        
        # Outcome codes: 0=home_win, 1=draw, 2=away_win
        home_win_prob = probas[0]
        draw_prob = probas[1]
        away_win_prob = probas[2]
        
        # Determine predicted outcome
        outcome_idx = np.argmax(probas)
        outcomes = ['Home Win', 'Draw', 'Away Win']
        predicted_outcome = outcomes[outcome_idx]
        confidence = probas[outcome_idx]
        
        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            predicted_outcome=predicted_outcome,
            confidence=confidence
        )
    
    def save(self, filepath: Path):
        """Save the trained model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        console.print(f"  💾 Model saved to: {filepath}")
    
    def load(self, filepath: Path):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        
        console.print(f"  📂 Model loaded from: {filepath}")


class ScorelinePredictor:
    """
    Poisson-based model for predicting exact scorelines.
    """
    
    FEATURE_COLUMNS = [
        'home_elo', 'away_elo', 'elo_diff',
        'home_form_goals_scored_avg', 'away_form_goals_scored_avg',
        'home_form_goals_conceded_avg', 'away_form_goals_conceded_avg',
        'h2h_goals_avg_team1', 'h2h_goals_avg_team2',
        'tournament_importance', 'is_knockout',
    ]
    
    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.home_goals_model = None
        self.away_goals_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Average goals per match (will be calculated from training data)
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train separate models for home and away goals."""
        from sklearn.linear_model import PoissonRegressor
        
        console.print("\n[bold blue]⚽ Training Scoreline Model[/bold blue]")
        
        # Prepare features
        available_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        self.feature_columns = available_cols
        
        X = df[available_cols].fillna(0).values
        y_home = df['home_score'].values.astype(int)
        y_away = df['away_score'].values.astype(int)
        
        # Calculate averages
        self.avg_home_goals = y_home.mean()
        self.avg_away_goals = y_away.mean()
        
        console.print(f"  Average home goals: {self.avg_home_goals:.2f}")
        console.print(f"  Average away goals: {self.avg_away_goals:.2f}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Poisson regression for home goals
        console.print("  Training home goals model...")
        self.home_goals_model = PoissonRegressor(max_iter=1000)
        self.home_goals_model.fit(X_scaled, y_home)
        
        # Train Poisson regression for away goals
        console.print("  Training away goals model...")
        self.away_goals_model = PoissonRegressor(max_iter=1000)
        self.away_goals_model.fit(X_scaled, y_away)
        
        self.is_trained = True
        
        console.print(f"  [bold green]✅ Scoreline model trained![/bold green]")
        
        return {
            'avg_home_goals': self.avg_home_goals,
            'avg_away_goals': self.avg_away_goals,
        }
    
    def predict_scoreline_probs(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict probability distribution over scorelines.
        
        Returns:
            Dictionary mapping "home_score-away_score" to probability
        """
        from scipy.stats import poisson
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get expected goals
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        
        lambda_home = self.home_goals_model.predict(X_scaled)[0]
        lambda_away = self.away_goals_model.predict(X_scaled)[0]
        
        # Ensure reasonable bounds
        lambda_home = np.clip(lambda_home, 0.3, 5.0)
        lambda_away = np.clip(lambda_away, 0.2, 4.0)
        
        # Calculate probability for each scoreline
        scoreline_probs = {}
        
        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                prob = (poisson.pmf(home_goals, lambda_home) * 
                       poisson.pmf(away_goals, lambda_away))
                scoreline = f"{home_goals}-{away_goals}"
                scoreline_probs[scoreline] = prob
        
        # Normalize
        total = sum(scoreline_probs.values())
        scoreline_probs = {k: v/total for k, v in scoreline_probs.items()}
        
        return scoreline_probs
    
    def get_most_likely_scores(self, features: Dict[str, float], 
                               top_n: int = 5) -> List[Tuple[str, float]]:
        """Get the most likely scorelines."""
        probs = self.predict_scoreline_probs(features)
        sorted_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]
    
    def predict_expected_goals(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Get expected goals for each team."""
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        
        lambda_home = self.home_goals_model.predict(X_scaled)[0]
        lambda_away = self.away_goals_model.predict(X_scaled)[0]
        
        return lambda_home, lambda_away
    
    def save(self, filepath: Path):
        """Save the model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'home_goals_model': self.home_goals_model,
            'away_goals_model': self.away_goals_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'avg_home_goals': self.avg_home_goals,
            'avg_away_goals': self.avg_away_goals,
            'max_goals': self.max_goals,
            'is_trained': self.is_trained,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        console.print(f"  💾 Scoreline model saved to: {filepath}")
    
    def load(self, filepath: Path):
        """Load the model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.home_goals_model = model_data['home_goals_model']
        self.away_goals_model = model_data['away_goals_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.avg_home_goals = model_data['avg_home_goals']
        self.avg_away_goals = model_data['avg_away_goals']
        self.max_goals = model_data['max_goals']
        self.is_trained = model_data['is_trained']


def main():
    """Train and save models."""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold green]🏆 World Cup 2026 Predictor - Model Training[/bold green]",
        border_style="green"
    ))
    
    # Load training data
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "training_data.csv"
    models_dir = project_root / "models"
    
    if not data_path.exists():
        console.print("[red]❌ Training data not found. Run preprocessing first.[/red]")
        return
    
    df = pd.read_csv(data_path)
    console.print(f"📂 Loaded {len(df):,} training samples")
    
    # Train Match Outcome Model
    outcome_model = MatchOutcomeModel(use_gpu=True)
    outcome_metrics = outcome_model.train(df)
    outcome_model.save(models_dir / "match_outcome_model.pkl")
    
    # Train Scoreline Model
    scoreline_model = ScorelinePredictor()
    scoreline_metrics = scoreline_model.train(df)
    scoreline_model.save(models_dir / "scoreline_model.pkl")
    
    console.print("\n[bold green]✅ All models trained and saved![/bold green]")


if __name__ == "__main__":
    main()
