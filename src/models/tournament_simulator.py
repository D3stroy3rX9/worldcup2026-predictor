"""
Tournament Simulator for World Cup 2026
Monte Carlo simulation for bracket progression and winner probabilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


@dataclass
class Team:
    """Represents a team in the tournament."""
    name: str
    code: str
    group: str
    confederation: str
    elo_rating: float = 1500
    
    # Group stage stats
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_diff: int = 0
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0


@dataclass 
class MatchResult:
    """Result of a simulated match."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    is_knockout: bool = False
    home_penalties: int = 0
    away_penalties: int = 0


class TournamentSimulator:
    """
    Monte Carlo tournament simulator for World Cup 2026.
    
    Features:
    - Group stage simulation
    - Knockout bracket progression
    - Multiple simulation runs for probability estimation
    """
    
    def __init__(self, feature_engineer, outcome_model, scoreline_model):
        """
        Initialize simulator with trained models.
        
        Args:
            feature_engineer: FeatureEngineer instance
            outcome_model: Trained MatchOutcomeModel
            scoreline_model: Trained ScorelinePredictor
        """
        self.fe = feature_engineer
        self.outcome_model = outcome_model
        self.scoreline_model = scoreline_model
        
        self.teams: Dict[str, Team] = {}
        self.groups: Dict[str, List[Team]] = defaultdict(list)
        
        # World Cup 2026 format: 48 teams, 12 groups of 4
        # Top 2 from each group + 8 best 3rd place teams advance
        self.teams_advancing_per_group = 2
        self.third_place_advancing = 8
        
    def load_teams(self, groups_file: Path):
        """Load teams and groups from YAML file."""
        with open(groups_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for group_name, teams in data['groups'].items():
            for team_data in teams:
                team = Team(
                    name=team_data['name'],
                    code=team_data['code'],
                    group=group_name,
                    confederation=team_data['confederation'],
                    elo_rating=self.fe.calculate_elo_rating(team_data['name'])
                )
                self.teams[team.name] = team
                self.groups[group_name].append(team)
        
        console.print(f"  Loaded {len(self.teams)} teams in {len(self.groups)} groups")
    
    def reset_tournament(self):
        """Reset all team stats for a new simulation."""
        for team in self.teams.values():
            team.points = 0
            team.goals_for = 0
            team.goals_against = 0
            team.goal_diff = 0
            team.matches_played = 0
            team.wins = 0
            team.draws = 0
            team.losses = 0
    
    def simulate_match(self, home_team: str, away_team: str, 
                       is_knockout: bool = False,
                       match_date=None) -> MatchResult:
        """
        Simulate a single match using the trained models.
        """
        from datetime import datetime
        
        if match_date is None:
            match_date = datetime(2026, 6, 15)  # Default tournament date
        
        # Handle TBD teams by using a default rating
        if home_team.startswith('TBD') or away_team.startswith('TBD'):
            # Use simple probability based on Elo if one team is TBD
            home_elo = self.fe.calculate_elo_rating(home_team, default=1600)
            away_elo = self.fe.calculate_elo_rating(away_team, default=1600)
            
            elo_diff = home_elo - away_elo
            home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
            draw_prob = 0.25 if not is_knockout else 0
            away_win_prob = 1 - home_win_prob - draw_prob
            
            # Ensure positive probabilities
            home_win_prob = max(0.1, home_win_prob)
            away_win_prob = max(0.1, away_win_prob)
            
            # Normalize
            total = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
            
            outcome = np.random.choice([0, 1, 2], p=[home_win_prob, draw_prob, away_win_prob])
            
            # Simple score generation
            if outcome == 0:  # Home win
                home_score = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                away_score = np.random.choice([0, 1], p=[0.6, 0.4])
                if home_score <= away_score:
                    home_score = away_score + 1
            elif outcome == 1:  # Draw
                home_score = away_score = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            else:  # Away win
                away_score = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                home_score = np.random.choice([0, 1], p=[0.6, 0.4])
                if away_score <= home_score:
                    away_score = home_score + 1
            
            result = MatchResult(
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                is_knockout=is_knockout
            )
            
            if is_knockout and home_score == away_score:
                if np.random.random() < 0.5:
                    result.home_penalties = 5
                    result.away_penalties = 4
                else:
                    result.home_penalties = 4
                    result.away_penalties = 5
            
            return result
        
        # Get features for the match
        features = self.fe.create_match_features(
            home_team, away_team, match_date,
            tournament='FIFA World Cup',
            is_knockout=is_knockout
        )
        
        # Get outcome probabilities
        prediction = self.outcome_model.predict(home_team, away_team, features)
        
        # Sample outcome based on probabilities
        if is_knockout:
            # No draws in knockouts - normalize between win probabilities
            total = prediction.home_win_prob + prediction.away_win_prob
            home_win_prob = prediction.home_win_prob / total
            away_win_prob = prediction.away_win_prob / total
            outcome_probs = np.array([home_win_prob, 0.0, away_win_prob])
        else:
            outcome_probs = np.array([
                prediction.home_win_prob,
                prediction.draw_prob,
                prediction.away_win_prob
            ])
        
        # Ensure probabilities are valid and sum to exactly 1.0
        outcome_probs = np.maximum(outcome_probs, 0.001)
        outcome_probs = outcome_probs / outcome_probs.sum()  # Normalize to sum to exactly 1
        
        # Sample outcome
        outcome = np.random.choice([0, 1, 2], p=outcome_probs)
        
        # Get expected goals
        exp_home, exp_away = self.scoreline_model.predict_expected_goals(features)
        
        # Sample goals from Poisson distribution
        home_score = np.random.poisson(exp_home)
        away_score = np.random.poisson(exp_away)
        
        # Ensure score matches outcome
        if outcome == 0:  # Home win
            if home_score <= away_score:
                home_score = away_score + 1
        elif outcome == 1:  # Draw
            away_score = home_score
        else:  # Away win
            if away_score <= home_score:
                away_score = home_score + 1
        
        # Cap at reasonable values
        home_score = min(home_score, 7)
        away_score = min(away_score, 7)
        
        result = MatchResult(
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            is_knockout=is_knockout
        )
        
        # Simulate penalties if needed (knockout draw after regulation)
        if is_knockout and home_score == away_score:
            # Penalties - slight home advantage
            home_pen_prob = 0.52 + (prediction.home_win_prob - 0.5) * 0.1
            if np.random.random() < home_pen_prob:
                result.home_penalties = 5
                result.away_penalties = 4
            else:
                result.home_penalties = 4
                result.away_penalties = 5
        
        return result
    
    def simulate_group_stage(self) -> Dict[str, List[Team]]:
        """
        Simulate all group stage matches.
        
        Returns:
            Dictionary mapping group to list of teams sorted by standings
        """
        for group_name, teams in self.groups.items():
            # Each team plays every other team once
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team1 = teams[i]
                    team2 = teams[j]
                    
                    result = self.simulate_match(team1.name, team2.name, is_knockout=False)
                    
                    # Update stats
                    team1.goals_for += result.home_score
                    team1.goals_against += result.away_score
                    team2.goals_for += result.away_score
                    team2.goals_against += result.home_score
                    team1.matches_played += 1
                    team2.matches_played += 1
                    
                    if result.home_score > result.away_score:
                        team1.points += 3
                        team1.wins += 1
                        team2.losses += 1
                    elif result.home_score == result.away_score:
                        team1.points += 1
                        team2.points += 1
                        team1.draws += 1
                        team2.draws += 1
                    else:
                        team2.points += 3
                        team2.wins += 1
                        team1.losses += 1
                    
                    team1.goal_diff = team1.goals_for - team1.goals_against
                    team2.goal_diff = team2.goals_for - team2.goals_against
        
        # Sort teams in each group
        standings = {}
        for group_name, teams in self.groups.items():
            sorted_teams = sorted(
                teams,
                key=lambda t: (t.points, t.goal_diff, t.goals_for),
                reverse=True
            )
            standings[group_name] = sorted_teams
        
        return standings
    
    def get_knockout_teams(self, standings: Dict[str, List[Team]]) -> List[Team]:
        """
        Determine which teams advance from group stage.
        
        World Cup 2026 format:
        - Top 2 from each group (24 teams)
        - Best 8 third-place teams (8 teams)
        - Total: 32 teams in knockout
        """
        advancing = []
        third_place_teams = []
        
        for group_name, teams in standings.items():
            # Top 2 advance automatically
            advancing.extend(teams[:2])
            
            # Collect third place teams
            if len(teams) >= 3:
                third_place_teams.append(teams[2])
        
        # Sort third place teams
        third_place_sorted = sorted(
            third_place_teams,
            key=lambda t: (t.points, t.goal_diff, t.goals_for),
            reverse=True
        )
        
        # Best 8 third place teams advance
        advancing.extend(third_place_sorted[:self.third_place_advancing])
        
        return advancing
    
    def simulate_knockout_round(self, teams: List[Team]) -> List[Team]:
        """
        Simulate a knockout round.
        
        Args:
            teams: List of teams in the round (will be paired)
            
        Returns:
            List of winning teams
        """
        if len(teams) % 2 != 0:
            raise ValueError(f"Invalid number of teams: {len(teams)}")
        
        winners = []
        np.random.shuffle(teams)  # Randomize bracket
        
        for i in range(0, len(teams), 2):
            team1 = teams[i]
            team2 = teams[i + 1]
            
            result = self.simulate_match(team1.name, team2.name, is_knockout=True)
            
            # Determine winner
            if result.home_score > result.away_score:
                winners.append(team1)
            elif result.away_score > result.home_score:
                winners.append(team2)
            else:
                # Penalties
                if result.home_penalties > result.away_penalties:
                    winners.append(team1)
                else:
                    winners.append(team2)
        
        return winners
    
    def simulate_tournament(self) -> Dict[str, any]:
        """
        Simulate a complete tournament.
        
        Returns:
            Dictionary with tournament results
        """
        self.reset_tournament()
        
        # Group stage
        standings = self.simulate_group_stage()
        
        # Get knockout teams
        knockout_teams = self.get_knockout_teams(standings)
        
        # Round of 32
        round_of_16 = self.simulate_knockout_round(knockout_teams)
        
        # Round of 16 -> Quarter Finals
        quarter_finals = self.simulate_knockout_round(round_of_16)
        
        # Quarter Finals -> Semi Finals
        semi_finals = self.simulate_knockout_round(quarter_finals)
        
        # Semi Finals -> Final
        finalists = self.simulate_knockout_round(semi_finals)
        
        # Final
        final_result = self.simulate_match(
            finalists[0].name, finalists[1].name, is_knockout=True
        )
        
        if final_result.home_score > final_result.away_score:
            champion = finalists[0]
            runner_up = finalists[1]
        elif final_result.away_score > final_result.home_score:
            champion = finalists[1]
            runner_up = finalists[0]
        else:
            if final_result.home_penalties > final_result.away_penalties:
                champion = finalists[0]
                runner_up = finalists[1]
            else:
                champion = finalists[1]
                runner_up = finalists[0]
        
        return {
            'champion': champion.name,
            'runner_up': runner_up.name,
            'semi_finalists': [t.name for t in semi_finals],
            'quarter_finalists': [t.name for t in quarter_finals],
            'round_of_16': [t.name for t in round_of_16],
            'group_standings': {g: [t.name for t in teams] for g, teams in standings.items()},
        }
    
    def run_simulations(self, n_simulations: int = 10000) -> Dict[str, Dict]:
        """
        Run multiple tournament simulations.
        
        Returns:
            Dictionary with probability estimates for each team
        """
        console.print(f"\n[bold blue]🎲 Running {n_simulations:,} tournament simulations...[/bold blue]")
        
        results = {
            'champion': defaultdict(int),
            'runner_up': defaultdict(int),
            'semi_finals': defaultdict(int),
            'quarter_finals': defaultdict(int),
            'round_of_16': defaultdict(int),
            'group_winner': defaultdict(int),
        }
        
        for _ in track(range(n_simulations), description="  Simulating"):
            sim = self.simulate_tournament()
            
            results['champion'][sim['champion']] += 1
            results['runner_up'][sim['runner_up']] += 1
            
            for team in sim['semi_finalists']:
                results['semi_finals'][team] += 1
            
            for team in sim['quarter_finalists']:
                results['quarter_finals'][team] += 1
            
            for team in sim['round_of_16']:
                results['round_of_16'][team] += 1
            
            for group, teams in sim['group_standings'].items():
                if teams:
                    results['group_winner'][teams[0]] += 1
        
        # Convert to probabilities
        probabilities = {}
        for stage, counts in results.items():
            probabilities[stage] = {
                team: count / n_simulations 
                for team, count in counts.items()
            }
        
        return probabilities
    
    def display_probabilities(self, probabilities: Dict[str, Dict], top_n: int = 10):
        """Display tournament probabilities in a nice table."""
        console.print("\n[bold green]🏆 Tournament Winner Probabilities[/bold green]\n")
        
        table = Table(title="Championship Odds")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Team", style="white")
        table.add_column("Win %", style="green", justify="right")
        table.add_column("Final %", style="yellow", justify="right")
        table.add_column("Semi %", style="blue", justify="right")
        
        # Sort by championship probability
        champ_probs = sorted(
            probabilities['champion'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (team, prob) in enumerate(champ_probs[:top_n], 1):
            final_prob = probabilities['runner_up'].get(team, 0) + prob
            semi_prob = probabilities['semi_finals'].get(team, 0)
            
            table.add_row(
                str(i),
                team,
                f"{prob*100:.1f}%",
                f"{final_prob*100:.1f}%",
                f"{semi_prob*100:.1f}%"
            )
        
        console.print(table)


def main():
    """Run tournament simulation."""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold green]🏆 World Cup 2026 Tournament Simulator[/bold green]",
        border_style="green"
    ))
    
    # This would be called from the main CLI with loaded models
    console.print("Run via CLI: python predict.py simulate --runs 10000")


if __name__ == "__main__":
    main()
