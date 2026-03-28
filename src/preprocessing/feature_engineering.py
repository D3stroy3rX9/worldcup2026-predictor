"""
Feature Engineering for World Cup 2026 Predictor
Transforms raw match data into features for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from rich.console import Console
from rich.progress import track

console = Console()


class FeatureEngineer:
    """
    Generates features for match prediction from historical data.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.matches_df = None
        self.elo_ratings = {}
        self.team_stats = defaultdict(dict)
        
    def load_data(self):
        """Load all required datasets."""
        console.print("[bold blue]📂 Loading data...[/bold blue]")
        
        # Load match results
        results_path = self.data_dir / "international_results.csv"
        self.matches_df = pd.read_csv(results_path)
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
        self.matches_df = self.matches_df.sort_values('date').reset_index(drop=True)
        console.print(f"  ✅ Loaded {len(self.matches_df):,} matches")
        
        # Load Elo ratings
        elo_path = self.data_dir / "elo_ratings.csv"
        if elo_path.exists():
            elo_df = pd.read_csv(elo_path)
            self.elo_ratings = dict(zip(elo_df['team'], elo_df['elo_rating']))
            console.print(f"  ✅ Loaded {len(self.elo_ratings)} Elo ratings")
        
        # Load FIFA rankings
        rankings_path = self.data_dir / "fifa_rankings.csv"
        if rankings_path.exists():
            self.fifa_rankings_df = pd.read_csv(rankings_path)
            console.print(f"  ✅ Loaded FIFA rankings")
    
    def calculate_elo_rating(self, team: str, default: float = 1500) -> float:
        """Get Elo rating for a team."""
        return self.elo_ratings.get(team, default)
    
    def calculate_form(self, team: str, before_date: datetime, 
                       num_matches: int = 10) -> Dict[str, float]:
        """
        Calculate team form based on recent matches.
        
        Returns:
            Dictionary with form metrics
        """
        # Filter matches before the given date involving the team
        team_matches = self.matches_df[
            (self.matches_df['date'] < before_date) &
            ((self.matches_df['home_team'] == team) | 
             (self.matches_df['away_team'] == team))
        ].tail(num_matches)
        
        if len(team_matches) == 0:
            return {
                'form_points': 0,
                'form_points_avg': 0,
                'form_goals_scored': 0,
                'form_goals_scored_avg': 0,
                'form_goals_conceded': 0,
                'form_goals_conceded_avg': 0,
                'form_goal_diff_avg': 0,
                'form_wins': 0,
                'form_draws': 0,
                'form_losses': 0,
                'form_win_rate': 0,
                'matches_played': 0
            }
        
        points = 0
        goals_scored = 0
        goals_conceded = 0
        wins = draws = losses = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                gf = match['home_score']
                ga = match['away_score']
            else:
                gf = match['away_score']
                ga = match['home_score']
            
            goals_scored += gf
            goals_conceded += ga
            
            if gf > ga:
                points += 3
                wins += 1
            elif gf == ga:
                points += 1
                draws += 1
            else:
                losses += 1
        
        n = len(team_matches)
        return {
            'form_points': points,
            'form_points_avg': points / n,
            'form_goals_scored': goals_scored,
            'form_goals_scored_avg': goals_scored / n,
            'form_goals_conceded': goals_conceded,
            'form_goals_conceded_avg': goals_conceded / n,
            'form_goal_diff_avg': (goals_scored - goals_conceded) / n,
            'form_wins': wins,
            'form_draws': draws,
            'form_losses': losses,
            'form_win_rate': wins / n,
            'matches_played': n
        }
    
    def calculate_h2h(self, team1: str, team2: str, 
                      before_date: datetime, years: int = 20) -> Dict[str, float]:
        """
        Calculate head-to-head statistics between two teams.
        """
        cutoff_date = before_date - timedelta(days=years * 365)
        
        h2h_matches = self.matches_df[
            (self.matches_df['date'] < before_date) &
            (self.matches_df['date'] >= cutoff_date) &
            (
                ((self.matches_df['home_team'] == team1) & 
                 (self.matches_df['away_team'] == team2)) |
                ((self.matches_df['home_team'] == team2) & 
                 (self.matches_df['away_team'] == team1))
            )
        ]
        
        if len(h2h_matches) == 0:
            return {
                'h2h_matches': 0,
                'h2h_wins_team1': 0,
                'h2h_draws': 0,
                'h2h_wins_team2': 0,
                'h2h_goals_team1': 0,
                'h2h_goals_team2': 0,
                'h2h_win_rate_team1': 0.5,  # Neutral if no history
            }
        
        wins_t1 = draws = wins_t2 = 0
        goals_t1 = goals_t2 = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == team1:
                g1, g2 = match['home_score'], match['away_score']
            else:
                g1, g2 = match['away_score'], match['home_score']
            
            goals_t1 += g1
            goals_t2 += g2
            
            if g1 > g2:
                wins_t1 += 1
            elif g1 == g2:
                draws += 1
            else:
                wins_t2 += 1
        
        n = len(h2h_matches)
        return {
            'h2h_matches': n,
            'h2h_wins_team1': wins_t1,
            'h2h_draws': draws,
            'h2h_wins_team2': wins_t2,
            'h2h_goals_team1': goals_t1,
            'h2h_goals_team2': goals_t2,
            'h2h_goals_avg_team1': goals_t1 / n,
            'h2h_goals_avg_team2': goals_t2 / n,
            'h2h_win_rate_team1': (wins_t1 + 0.5 * draws) / n,
        }
    
    def calculate_tournament_features(self, tournament: str) -> Dict[str, float]:
        """
        Calculate tournament-specific features.
        """
        tournament_importance = {
            'FIFA World Cup': 1.0,
            'FIFA World Cup qualification': 0.7,
            'UEFA Euro': 0.9,
            'UEFA Euro qualification': 0.6,
            'Copa América': 0.85,
            'African Cup of Nations': 0.8,
            'AFC Asian Cup': 0.75,
            'CONCACAF Gold Cup': 0.7,
            'UEFA Nations League': 0.65,
            'Friendly': 0.3,
        }
        
        importance = 0.5  # Default
        for key, value in tournament_importance.items():
            if key.lower() in tournament.lower():
                importance = value
                break
        
        return {
            'tournament_importance': importance,
            'is_world_cup': 1 if 'world cup' in tournament.lower() else 0,
            'is_knockout': 0,  # Will be set based on match context
        }
    
    def get_confederation(self, team: str) -> str:
        """Get confederation for a team."""
        # Simplified confederation mapping
        confederations = {
            'UEFA': ['Germany', 'France', 'Spain', 'Italy', 'England', 'Portugal', 
                     'Netherlands', 'Belgium', 'Croatia', 'Switzerland', 'Austria',
                     'Poland', 'Ukraine', 'Sweden', 'Norway', 'Denmark', 'Scotland',
                     'Czech Republic', 'Serbia', 'Romania', 'Greece', 'Hungary',
                     'Turkey', 'Wales', 'Ireland', 'Northern Ireland', 'Slovakia',
                     'Slovenia', 'Finland', 'Iceland', 'Albania', 'Bosnia and Herzegovina',
                     'North Macedonia', 'Kosovo', 'Montenegro', 'Georgia', 'Luxembourg',
                     'Cyprus', 'Malta', 'Andorra', 'San Marino', 'Liechtenstein',
                     'Faroe Islands', 'Gibraltar', 'Moldova', 'Belarus', 'Latvia',
                     'Lithuania', 'Estonia', 'Armenia', 'Azerbaijan', 'Kazakhstan'],
            'CONMEBOL': ['Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Chile',
                         'Peru', 'Ecuador', 'Paraguay', 'Venezuela', 'Bolivia'],
            'CONCACAF': ['Mexico', 'USA', 'Canada', 'Costa Rica', 'Panama',
                         'Jamaica', 'Honduras', 'El Salvador', 'Guatemala',
                         'Haiti', 'Trinidad and Tobago', 'Curaçao', 'Suriname',
                         'Cuba', 'Nicaragua', 'Bermuda', 'Dominican Republic'],
            'CAF': ['Morocco', 'Senegal', 'Nigeria', 'Egypt', 'Cameroon',
                    'Algeria', 'Tunisia', 'Ghana', "Côte d'Ivoire", 'Mali',
                    'South Africa', 'DR Congo', 'Burkina Faso', 'Guinea',
                    'Zambia', 'Zimbabwe', 'Kenya', 'Uganda', 'Tanzania',
                    'Cabo Verde', 'Mauritania', 'Gabon', 'Benin', 'Togo',
                    'Madagascar', 'Central African Republic', 'Equatorial Guinea',
                    'Libya', 'Sudan', 'Ethiopia', 'Namibia', 'Mozambique',
                    'Botswana', 'Angola', 'Rwanda', 'Malawi', 'Sierra Leone',
                    'Liberia', 'Niger', 'Chad', 'Comoros', 'Gambia'],
            'AFC': ['Japan', 'South Korea', 'Australia', 'Iran', 'Saudi Arabia',
                    'Qatar', 'UAE', 'Iraq', 'Oman', 'Bahrain', 'Jordan',
                    'Syria', 'Palestine', 'Lebanon', 'Kuwait', 'China PR',
                    'Thailand', 'Vietnam', 'Indonesia', 'Malaysia', 'Philippines',
                    'India', 'Uzbekistan', 'Tajikistan', 'Turkmenistan', 'Kyrgyzstan',
                    'North Korea', 'Hong Kong', 'Singapore', 'Myanmar', 'Cambodia',
                    'Laos', 'Nepal', 'Bangladesh', 'Maldives', 'Sri Lanka',
                    'Afghanistan', 'Yemen'],
            'OFC': ['New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands',
                    'Vanuatu', 'New Caledonia', 'Tahiti', 'Samoa', 'Tonga',
                    'American Samoa', 'Cook Islands'],
        }
        
        for conf, teams in confederations.items():
            if team in teams:
                return conf
        return 'Unknown'
    
    def create_match_features(self, home_team: str, away_team: str, 
                              match_date: datetime, tournament: str = 'FIFA World Cup',
                              is_knockout: bool = False) -> Dict[str, float]:
        """
        Create all features for a match prediction.
        """
        features = {}
        
        # Elo ratings
        home_elo = self.calculate_elo_rating(home_team)
        away_elo = self.calculate_elo_rating(away_team)
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        features['elo_sum'] = home_elo + away_elo
        
        # Form features for home team
        home_form = self.calculate_form(home_team, match_date)
        for key, value in home_form.items():
            features[f'home_{key}'] = value
        
        # Form features for away team
        away_form = self.calculate_form(away_team, match_date)
        for key, value in away_form.items():
            features[f'away_{key}'] = value
        
        # Form differentials
        features['form_diff'] = home_form['form_points_avg'] - away_form['form_points_avg']
        features['goals_scored_diff'] = home_form['form_goals_scored_avg'] - away_form['form_goals_scored_avg']
        features['goals_conceded_diff'] = home_form['form_goals_conceded_avg'] - away_form['form_goals_conceded_avg']
        
        # Head-to-head
        h2h = self.calculate_h2h(home_team, away_team, match_date)
        for key, value in h2h.items():
            features[key] = value
        
        # Tournament features
        tourn_features = self.calculate_tournament_features(tournament)
        features.update(tourn_features)
        features['is_knockout'] = 1 if is_knockout else 0
        
        # Confederation features
        home_conf = self.get_confederation(home_team)
        away_conf = self.get_confederation(away_team)
        features['same_confederation'] = 1 if home_conf == away_conf else 0
        
        # Confederation strength (historical World Cup performance proxy)
        conf_strength = {
            'UEFA': 0.9, 'CONMEBOL': 0.85, 'CONCACAF': 0.5,
            'CAF': 0.45, 'AFC': 0.4, 'OFC': 0.2, 'Unknown': 0.3
        }
        features['home_conf_strength'] = conf_strength.get(home_conf, 0.3)
        features['away_conf_strength'] = conf_strength.get(away_conf, 0.3)
        
        return features
    
    def build_training_dataset(self, start_year: int = 2010, 
                               end_year: int = 2024) -> pd.DataFrame:
        """
        Build the full training dataset from historical matches.
        """
        console.print(f"\n[bold blue]🔧 Building training dataset ({start_year}-{end_year})...[/bold blue]")
        
        # Filter matches
        mask = (
            (self.matches_df['date'].dt.year >= start_year) &
            (self.matches_df['date'].dt.year <= end_year) &
            (self.matches_df['home_score'].notna()) &
            (self.matches_df['away_score'].notna())
        )
        filtered_matches = self.matches_df[mask].copy()
        
        console.print(f"  Processing {len(filtered_matches):,} matches...")
        
        training_data = []
        
        for idx, match in track(filtered_matches.iterrows(), 
                                total=len(filtered_matches),
                                description="  Building features"):
            try:
                features = self.create_match_features(
                    home_team=match['home_team'],
                    away_team=match['away_team'],
                    match_date=match['date'],
                    tournament=match.get('tournament', 'Unknown'),
                    is_knockout=False
                )
                
                # Add target variables
                home_score = int(match['home_score'])
                away_score = int(match['away_score'])
                
                features['home_score'] = home_score
                features['away_score'] = away_score
                features['total_goals'] = home_score + away_score
                features['goal_diff'] = home_score - away_score
                
                # Match outcome (from home team perspective)
                if home_score > away_score:
                    features['outcome'] = 'home_win'
                    features['outcome_code'] = 0
                elif home_score == away_score:
                    features['outcome'] = 'draw'
                    features['outcome_code'] = 1
                else:
                    features['outcome'] = 'away_win'
                    features['outcome_code'] = 2
                
                # Metadata
                features['match_date'] = match['date']
                features['home_team'] = match['home_team']
                features['away_team'] = match['away_team']
                features['tournament'] = match.get('tournament', 'Unknown')
                
                training_data.append(features)
                
            except Exception as e:
                continue  # Skip problematic matches
        
        df = pd.DataFrame(training_data)
        console.print(f"  ✅ Created {len(df):,} training samples")
        
        return df
    
    def save_training_data(self, df: pd.DataFrame, output_dir: Path):
        """Save the training dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / "training_data.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"\n  💾 Saved to: {csv_path}")
        
        # Try to save as Parquet if available
        try:
            parquet_path = output_dir / "training_data.parquet"
            df.to_parquet(parquet_path, index=False)
            console.print(f"  💾 Saved to: {parquet_path}")
        except ImportError:
            console.print("  ⚠️ Parquet not available (pyarrow not installed)", style="yellow")


def main():
    """Main preprocessing pipeline."""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold green]🏆 World Cup 2026 Predictor - Feature Engineering[/bold green]\n"
        "Building training dataset with match features",
        border_style="green"
    ))
    
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Initialize feature engineer
    fe = FeatureEngineer(raw_data_dir)
    fe.load_data()
    
    # Build training dataset
    training_df = fe.build_training_dataset(start_year=2010, end_year=2024)
    
    # Save
    fe.save_training_data(training_df, processed_dir)
    
    # Print summary statistics
    console.print("\n[bold blue]📊 Dataset Summary[/bold blue]")
    console.print(f"  Total matches: {len(training_df):,}")
    console.print(f"  Features: {len([c for c in training_df.columns if c not in ['match_date', 'home_team', 'away_team', 'tournament', 'outcome']])}")
    console.print(f"  Date range: {training_df['match_date'].min().date()} to {training_df['match_date'].max().date()}")
    
    outcome_dist = training_df['outcome'].value_counts()
    console.print(f"\n  Outcome distribution:")
    console.print(f"    Home wins: {outcome_dist.get('home_win', 0):,} ({outcome_dist.get('home_win', 0)/len(training_df)*100:.1f}%)")
    console.print(f"    Draws:     {outcome_dist.get('draw', 0):,} ({outcome_dist.get('draw', 0)/len(training_df)*100:.1f}%)")
    console.print(f"    Away wins: {outcome_dist.get('away_win', 0):,} ({outcome_dist.get('away_win', 0)/len(training_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
