"""
Dynamic Tournament Mode for World Cup 2026
Predicts the entire tournament match-by-match with realistic score variance
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.stats import poisson
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


@dataclass
class TeamStanding:
    """Team's current standing in group."""
    name: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    
    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws
    
    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class MatchPrediction:
    """A single match prediction."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    stage: str
    match_number: int
    is_actual: bool = False


class DynamicTournament:
    """
    Dynamic tournament simulator that predicts match-by-match.
    """
    
    def __init__(self, fe, outcome_model, scoreline_model):
        self.fe = fe
        self.outcome_model = outcome_model
        self.scoreline_model = scoreline_model
        
        self.groups: Dict[str, List[str]] = {}
        self.standings: Dict[str, Dict[str, TeamStanding]] = {}
        self.match_results: List[MatchPrediction] = []
        self.actual_results: Dict[str, MatchPrediction] = {}
        
        self.knockout_teams: List[str] = []
        self.champion: str = None
        
    def load_groups(self, groups_file: Path):
        """Load groups from YAML file, skipping TBD teams."""
        with open(groups_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for group_name, teams in data['groups'].items():
            valid_teams = [t['name'] for t in teams if not t['name'].startswith('TBD')]
            
            if len(valid_teams) >= 2:
                self.groups[group_name] = valid_teams
                self.standings[group_name] = {
                    team: TeamStanding(name=team) for team in valid_teams
                }
        
        console.print(f"  Loaded {len(self.groups)} groups with {sum(len(t) for t in self.groups.values())} teams")
        console.print(f"  [dim](TBD teams excluded)[/dim]")
    
    def calculate_expected_goals(self, features: Dict, outcome) -> Tuple[float, float]:
        """Calculate expected goals based on team strength."""
        home_elo = features.get('home_elo', 1500)
        away_elo = features.get('away_elo', 1500)
        
        BASE_GOALS = 1.4
        
        home_strength = (home_elo - 1500) / 400
        away_strength = (away_elo - 1500) / 400
        
        exp_home = BASE_GOALS + (home_strength * 0.5) - (away_strength * 0.3)
        exp_away = BASE_GOALS + (away_strength * 0.5) - (home_strength * 0.3)
        
        exp_home = max(0.5, min(3.5, exp_home))
        exp_away = max(0.3, min(3.0, exp_away))
        
        # Adjust based on win probability
        if outcome.home_win_prob > 0.7:
            exp_home = max(exp_home, exp_away + 0.8)
        elif outcome.home_win_prob > 0.6:
            exp_home = max(exp_home, exp_away + 0.5)
        
        if outcome.away_win_prob > 0.7:
            exp_away = max(exp_away, exp_home + 0.8)
        elif outcome.away_win_prob > 0.6:
            exp_away = max(exp_away, exp_home + 0.5)
        
        return exp_home, exp_away
    
    def sample_scoreline(self, exp_home: float, exp_away: float, 
                         outcome, is_knockout: bool = False) -> Tuple[int, int]:
        """
        Sample a realistic scoreline with variance.
        Instead of always picking the most likely score, we sample from the distribution.
        """
        # Add some variance to expected goals (real football has variance!)
        variance = 0.4 if is_knockout else 0.3  # More variance in knockout games
        
        exp_home_varied = max(0.3, exp_home + np.random.normal(0, variance))
        exp_away_varied = max(0.2, exp_away + np.random.normal(0, variance))
        
        # Generate all possible scorelines with probabilities
        scorelines = []
        for h in range(8):
            for a in range(8):
                prob = poisson.pmf(h, exp_home_varied) * poisson.pmf(a, exp_away_varied)
                scorelines.append(((h, a), prob))
        
        # Determine expected outcome
        home_favored = outcome.home_win_prob > outcome.away_win_prob
        away_favored = outcome.away_win_prob > outcome.home_win_prob
        draw_favored = outcome.draw_prob > outcome.home_win_prob and outcome.draw_prob > outcome.away_win_prob
        
        # Filter by outcome type (but allow some upsets!)
        upset_chance = 0.15 if not is_knockout else 0.20  # 15-20% chance of upset
        
        if is_knockout:
            # No draws allowed - must have a winner
            valid_scorelines = [(s, p) for s, p in scorelines if s[0] != s[1]]
            
            if np.random.random() > upset_chance:
                # Normal case - favorite wins
                if home_favored:
                    valid_scorelines = [(s, p) for s, p in valid_scorelines if s[0] > s[1]]
                else:
                    valid_scorelines = [(s, p) for s, p in valid_scorelines if s[1] > s[0]]
            # else: upset - let any non-draw scoreline be possible
        else:
            # Group stage - draws allowed
            if np.random.random() > upset_chance:
                if home_favored:
                    valid_scorelines = [(s, p) for s, p in scorelines if s[0] > s[1]]
                elif away_favored:
                    valid_scorelines = [(s, p) for s, p in scorelines if s[1] > s[0]]
                else:  # Draw favored
                    valid_scorelines = [(s, p) for s, p in scorelines if s[0] == s[1]]
            else:
                # Upset possible - any scoreline
                valid_scorelines = scorelines
        
        if not valid_scorelines:
            valid_scorelines = scorelines
        
        # Normalize probabilities
        total_prob = sum(p for _, p in valid_scorelines)
        if total_prob > 0:
            valid_scorelines = [(s, p/total_prob) for s, p in valid_scorelines]
        
        # Sample from distribution (weighted random choice)
        scores, probs = zip(*valid_scorelines)
        idx = np.random.choice(len(scores), p=probs)
        
        return scores[idx]
    
    def predict_match(self, home_team: str, away_team: str, 
                      stage: str, match_num: int,
                      is_knockout: bool = False) -> MatchPrediction:
        """Predict a single match with realistic variance."""
        
        # Check for actual result
        match_key = f"{home_team} vs {away_team}"
        reverse_key = f"{away_team} vs {home_team}"
        
        if match_key in self.actual_results:
            return self.actual_results[match_key]
        if reverse_key in self.actual_results:
            actual = self.actual_results[reverse_key]
            return MatchPrediction(
                home_team=home_team,
                away_team=away_team,
                home_score=actual.away_score,
                away_score=actual.home_score,
                home_win_prob=actual.away_win_prob,
                draw_prob=actual.draw_prob,
                away_win_prob=actual.home_win_prob,
                stage=stage,
                match_number=match_num,
                is_actual=True
            )
        
        # Generate prediction
        match_date = datetime(2026, 6, 15)
        features = self.fe.create_match_features(
            home_team, away_team, match_date,
            tournament='FIFA World Cup',
            is_knockout=is_knockout
        )
        
        outcome = self.outcome_model.predict(home_team, away_team, features)
        
        # Calculate expected goals
        exp_home, exp_away = self.calculate_expected_goals(features, outcome)
        
        # Sample a scoreline with variance
        home_score, away_score = self.sample_scoreline(
            exp_home, exp_away, outcome, is_knockout
        )
        
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            home_win_prob=outcome.home_win_prob,
            draw_prob=outcome.draw_prob,
            away_win_prob=outcome.away_win_prob,
            stage=stage,
            match_number=match_num,
            is_actual=False
        )
    
    def update_standing(self, group: str, match: MatchPrediction):
        """Update group standing after a match."""
        home = self.standings[group][match.home_team]
        away = self.standings[group][match.away_team]
        
        home.played += 1
        away.played += 1
        home.goals_for += match.home_score
        home.goals_against += match.away_score
        away.goals_for += match.away_score
        away.goals_against += match.home_score
        
        if match.home_score > match.away_score:
            home.wins += 1
            away.losses += 1
        elif match.home_score < match.away_score:
            away.wins += 1
            home.losses += 1
        else:
            home.draws += 1
            away.draws += 1
    
    def get_group_ranking(self, group: str) -> List[TeamStanding]:
        """Get sorted standings for a group."""
        teams = list(self.standings[group].values())
        return sorted(teams, key=lambda t: (t.points, t.goal_diff, t.goals_for), reverse=True)
    
    def simulate_group_stage(self, verbose: bool = True) -> Dict[str, List[TeamStanding]]:
        """Simulate all group stage matches."""
        if verbose:
            console.print("\n[bold cyan]═══ GROUP STAGE ═══[/bold cyan]\n")
        
        match_num = 1
        
        for group_name in sorted(self.groups.keys()):
            teams = self.groups[group_name]
            
            if verbose:
                console.print(f"[bold]Group {group_name}[/bold]: {', '.join(teams)}")
            
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    match = self.predict_match(
                        teams[i], teams[j],
                        stage=f"Group {group_name}",
                        match_num=match_num,
                        is_knockout=False
                    )
                    self.match_results.append(match)
                    self.update_standing(group_name, match)
                    match_num += 1
                    
                    if verbose:
                        result_type = "[green](actual)[/green]" if match.is_actual else "[dim](pred)[/dim]"
                        console.print(f"  {match.home_team} [cyan]{match.home_score}[/cyan] - [cyan]{match.away_score}[/cyan] {match.away_team} {result_type}")
            
            if verbose:
                self._print_group_table(group_name)
                console.print()
        
        return {g: self.get_group_ranking(g) for g in self.groups}
    
    def _print_group_table(self, group: str):
        """Print a group standings table."""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("Team", style="white", width=18)
        table.add_column("P", justify="center", width=3)
        table.add_column("W", justify="center", width=3)
        table.add_column("D", justify="center", width=3)
        table.add_column("L", justify="center", width=3)
        table.add_column("GF", justify="center", width=3)
        table.add_column("GA", justify="center", width=3)
        table.add_column("GD", justify="center", width=4)
        table.add_column("Pts", justify="center", style="bold green", width=4)
        
        for standing in self.get_group_ranking(group):
            gd = f"+{standing.goal_diff}" if standing.goal_diff > 0 else str(standing.goal_diff)
            table.add_row(
                standing.name,
                str(standing.played),
                str(standing.wins),
                str(standing.draws),
                str(standing.losses),
                str(standing.goals_for),
                str(standing.goals_against),
                gd,
                str(standing.points)
            )
        
        console.print(table)
    
    def determine_knockout_teams(self, standings: Dict[str, List[TeamStanding]]) -> List[str]:
        """Determine which teams advance to knockout rounds."""
        advancing = []
        third_place = []
        
        for group_name, group_standings in standings.items():
            if len(group_standings) >= 2:
                advancing.append(group_standings[0].name)
                advancing.append(group_standings[1].name)
            if len(group_standings) >= 3:
                third_place.append(group_standings[2])
        
        # Best 8 third-place teams
        third_place_sorted = sorted(third_place, key=lambda t: (t.points, t.goal_diff, t.goals_for), reverse=True)
        advancing.extend([t.name for t in third_place_sorted[:8]])
        
        self.knockout_teams = advancing
        return advancing
    
    def simulate_knockout_round(self, teams: List[str], round_name: str, 
                                verbose: bool = True) -> List[str]:
        """Simulate a knockout round."""
        if verbose:
            console.print(f"\n[bold cyan]═══ {round_name.upper()} ═══[/bold cyan]\n")
        
        winners = []
        match_num = len(self.match_results) + 1
        
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):
                winners.append(teams[i])
                continue
            
            team1, team2 = teams[i], teams[i + 1]
            
            match = self.predict_match(
                team1, team2,
                stage=round_name,
                match_num=match_num,
                is_knockout=True
            )
            self.match_results.append(match)
            match_num += 1
            
            if match.home_score > match.away_score:
                winner = team1
            else:
                winner = team2
            
            winners.append(winner)
            
            if verbose:
                result_type = "[green](actual)[/green]" if match.is_actual else "[dim](pred)[/dim]"
                console.print(f"  {team1} [cyan]{match.home_score}[/cyan] - [cyan]{match.away_score}[/cyan] {team2} {result_type}")
                console.print(f"    → [bold green]{winner} advances[/bold green]")
        
        return winners
    
    def simulate_tournament(self, verbose: bool = True) -> str:
        """Simulate the entire tournament and return the champion."""
        
        if verbose:
            console.print(Panel.fit(
                "[bold cyan]FIFA WORLD CUP 2026[/bold cyan]\n"
                "[white]Dynamic Tournament Simulation[/white]",
                border_style="cyan"
            ))
        
        # Group Stage
        standings = self.simulate_group_stage(verbose=verbose)
        
        # Determine knockout teams
        knockout_teams = self.determine_knockout_teams(standings)
        
        if verbose:
            console.print(f"\n[bold]Teams advancing:[/bold] {len(knockout_teams)}")
        
        # Shuffle for bracket
        np.random.shuffle(knockout_teams)
        
        # Round of 32
        if len(knockout_teams) >= 32:
            round_of_16_teams = self.simulate_knockout_round(
                knockout_teams[:32], "Round of 32", verbose
            )
        else:
            round_of_16_teams = knockout_teams[:16]
        
        # Round of 16
        quarter_finalists = self.simulate_knockout_round(
            round_of_16_teams[:16], "Round of 16", verbose
        )
        
        # Quarter Finals
        semi_finalists = self.simulate_knockout_round(
            quarter_finalists, "Quarter Finals", verbose
        )
        
        # Semi Finals
        finalists = self.simulate_knockout_round(
            semi_finalists, "Semi Finals", verbose
        )
        
        # Final
        if verbose:
            console.print(f"\n[bold yellow]═══ THE FINAL ═══[/bold yellow]\n")
        
        final_match = self.predict_match(
            finalists[0], finalists[1],
            stage="Final",
            match_num=len(self.match_results) + 1,
            is_knockout=True
        )
        self.match_results.append(final_match)
        
        if final_match.home_score > final_match.away_score:
            self.champion = finalists[0]
        else:
            self.champion = finalists[1]
        
        if verbose:
            console.print(f"  [bold]{finalists[0]} [cyan]{final_match.home_score}[/cyan] - [cyan]{final_match.away_score}[/cyan] {finalists[1]}[/bold]")
            console.print(f"\n[bold yellow]CHAMPION: {self.champion}[/bold yellow]")
        
        return self.champion
    
    def add_actual_result(self, home_team: str, away_team: str, 
                          home_score: int, away_score: int, stage: str = "Group"):
        """Add an actual match result."""
        match_key = f"{home_team} vs {away_team}"
        
        self.actual_results[match_key] = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            home_win_prob=1.0 if home_score > away_score else 0.0,
            draw_prob=1.0 if home_score == away_score else 0.0,
            away_win_prob=1.0 if away_score > home_score else 0.0,
            stage=stage,
            match_number=0,
            is_actual=True
        )
        
        console.print(f"[green]✓[/green] Added: {home_team} {home_score} - {away_score} {away_team}")
    
    def load_actual_results(self, results_file: Path):
        """Load actual results from a CSV file."""
        if not results_file.exists():
            return
        
        df = pd.read_csv(results_file)
        
        for _, row in df.iterrows():
            self.add_actual_result(
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=int(row['home_score']),
                away_score=int(row['away_score']),
                stage=row.get('stage', 'Group')
            )
    
    def save_predictions(self, output_file: Path):
        """Save all predictions to a CSV file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for match in self.match_results:
            data.append({
                'match_number': match.match_number,
                'stage': match.stage,
                'home_team': match.home_team,
                'away_team': match.away_team,
                'home_score': match.home_score,
                'away_score': match.away_score,
                'home_win_prob': round(match.home_win_prob, 3),
                'draw_prob': round(match.draw_prob, 3),
                'away_win_prob': round(match.away_win_prob, 3),
                'is_actual': match.is_actual
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        console.print(f"[green]✓[/green] Saved {len(df)} predictions to {output_file}")


def main():
    """Run dynamic tournament simulation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from preprocessing.feature_engineering import FeatureEngineer
    from models.match_predictor import MatchOutcomeModel, ScorelinePredictor
    
    project_root = Path(__file__).parent.parent.parent
    
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    console.print("[dim]Loading models...[/dim]")
    
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        fe = FeatureEngineer(project_root / "data" / "raw")
        fe.load_data()
        
        outcome_model = MatchOutcomeModel(use_gpu=False)
        outcome_model.load(project_root / "models" / "match_outcome_model.pkl")
        
        scoreline_model = ScorelinePredictor()
        scoreline_model.load(project_root / "models" / "scoreline_model.pkl")
    
    tournament = DynamicTournament(fe, outcome_model, scoreline_model)
    tournament.load_groups(project_root / "data" / "raw" / "worldcup2026_groups.yaml")
    
    results_file = project_root / "data" / "actual_results.csv"
    if results_file.exists():
        tournament.load_actual_results(results_file)
    
    champion = tournament.simulate_tournament(verbose=True)
    
    tournament.save_predictions(project_root / "data" / "predictions" / "tournament_predictions.csv")


if __name__ == "__main__":
    main()