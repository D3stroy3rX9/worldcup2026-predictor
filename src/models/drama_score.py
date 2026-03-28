"""
Drama Score Predictor
Predicts the "excitement potential" of a match
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class DramaScoreResult:
    """Drama prediction output."""
    overall_score: float  # 0-10 scale
    upset_potential: str  # Low/Medium/High
    high_scoring_prob: float
    close_game_prob: float
    late_drama_prob: float
    components: Dict[str, float]
    narrative: str


class DramaScorePredictor:
    """
    Predicts the "drama potential" of a match.
    
    Drama factors:
    1. Upset potential (underdog beating favorite)
    2. Historical volatility between teams
    3. Tournament stakes (knockout, must-win, etc.)
    4. Rivalry factor
    5. Late goal tendency
    """
    
    # Known rivalries (manual curation)
    RIVALRIES = {
        ('Argentina', 'Brazil'): 0.95,
        ('England', 'Germany'): 0.9,
        ('England', 'Argentina'): 0.85,
        ('Netherlands', 'Germany'): 0.8,
        ('Spain', 'Portugal'): 0.75,
        ('USA', 'Mexico'): 0.85,
        ('France', 'Italy'): 0.7,
        ('Brazil', 'Uruguay'): 0.75,
        ('Japan', 'South Korea'): 0.8,
        ('Egypt', 'Algeria'): 0.75,
        ('Morocco', 'Algeria'): 0.75,
        ('Iran', 'Saudi Arabia'): 0.7,
        ('Croatia', 'Serbia'): 0.85,
        ('England', 'Scotland'): 0.8,
    }
    
    def __init__(self):
        self.is_ready = True
    
    def get_rivalry_score(self, team1: str, team2: str) -> float:
        """Check if teams have a known rivalry."""
        key1 = (team1, team2)
        key2 = (team2, team1)
        
        if key1 in self.RIVALRIES:
            return self.RIVALRIES[key1]
        if key2 in self.RIVALRIES:
            return self.RIVALRIES[key2]
        
        return 0.0
    
    def calculate_upset_potential(self, elo_diff: float, 
                                   home_win_prob: float,
                                   away_win_prob: float) -> Tuple[float, str]:
        """
        Calculate upset potential based on Elo difference.
        
        Returns:
            Tuple of (score, category)
        """
        abs_diff = abs(elo_diff)
        
        # Smaller team has chance?
        if abs_diff < 50:
            # Very close match
            return 0.6, "Medium"
        elif abs_diff < 100:
            underdog_prob = min(home_win_prob, away_win_prob)
            if underdog_prob > 0.25:
                return 0.7, "Medium"
            return 0.5, "Low"
        elif abs_diff < 200:
            underdog_prob = min(home_win_prob, away_win_prob)
            if underdog_prob > 0.2:
                return 0.8, "High"
            return 0.6, "Medium"
        else:
            # Big mismatch - huge upset potential
            return 0.9, "High"
    
    def calculate_scoring_potential(self, exp_home_goals: float, 
                                    exp_away_goals: float) -> float:
        """Probability of a high-scoring game (4+ goals)."""
        from scipy.stats import poisson
        
        total_expected = exp_home_goals + exp_away_goals
        
        # P(total >= 4)
        prob_0 = poisson.pmf(0, total_expected)
        prob_1 = poisson.pmf(1, total_expected)
        prob_2 = poisson.pmf(2, total_expected)
        prob_3 = poisson.pmf(3, total_expected)
        
        return 1 - (prob_0 + prob_1 + prob_2 + prob_3)
    
    def calculate_close_game_prob(self, home_win_prob: float, 
                                   draw_prob: float,
                                   away_win_prob: float) -> float:
        """Probability of a close game (1-goal margin or draw)."""
        # More balanced probabilities = closer game expected
        probs = [home_win_prob, draw_prob, away_win_prob]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = -3 * (1/3) * np.log(1/3)  # Maximum entropy
        
        return entropy / max_entropy
    
    def calculate_stakes_multiplier(self, is_knockout: bool,
                                     is_final: bool = False,
                                     must_win: bool = False) -> float:
        """Calculate stakes based on match importance."""
        multiplier = 1.0
        
        if is_knockout:
            multiplier = 1.3
        if is_final:
            multiplier = 1.5
        if must_win:
            multiplier *= 1.2
        
        return min(multiplier, 1.5)
    
    def generate_narrative(self, drama_score: float, 
                          upset_potential: str,
                          rivalry_score: float,
                          is_knockout: bool) -> str:
        """Generate a narrative description of the match's drama potential."""
        narratives = []
        
        if drama_score >= 8:
            narratives.append("MUST-WATCH! This game has all the ingredients for an instant classic.")
        elif drama_score >= 6:
            narratives.append("High entertainment value expected.")
        elif drama_score >= 4:
            narratives.append("Solid matchup with potential for excitement.")
        else:
            narratives.append("Likely a more tactical, controlled affair.")
        
        if upset_potential == "High":
            narratives.append("Watch out for a potential giant-killing!")
        
        if rivalry_score > 0.7:
            narratives.append("Historic rivalry adds extra spice.")
        
        if is_knockout:
            narratives.append("Knockout stakes mean every moment matters.")
        
        return " ".join(narratives)
    
    def predict(self, 
                home_team: str,
                away_team: str,
                elo_diff: float,
                home_win_prob: float,
                draw_prob: float,
                away_win_prob: float,
                exp_home_goals: float = 1.5,
                exp_away_goals: float = 1.2,
                is_knockout: bool = False,
                is_final: bool = False,
                must_win: bool = False) -> DramaScoreResult:
        """
        Calculate drama score for a match.
        
        Returns:
            DramaScoreResult with overall score and components
        """
        components = {}
        
        # 1. Upset potential (0-1 scale)
        upset_score, upset_category = self.calculate_upset_potential(
            elo_diff, home_win_prob, away_win_prob
        )
        components['upset_potential'] = upset_score
        
        # 2. Rivalry factor (0-1 scale)
        rivalry_score = self.get_rivalry_score(home_team, away_team)
        components['rivalry'] = rivalry_score
        
        # 3. High-scoring probability
        high_scoring_prob = self.calculate_scoring_potential(exp_home_goals, exp_away_goals)
        components['high_scoring'] = high_scoring_prob
        
        # 4. Close game probability
        close_game_prob = self.calculate_close_game_prob(
            home_win_prob, draw_prob, away_win_prob
        )
        components['close_game'] = close_game_prob
        
        # 5. Late drama potential (based on how close the match is)
        late_drama_prob = close_game_prob * 0.7 + (1 if is_knockout else 0) * 0.3
        components['late_drama'] = late_drama_prob
        
        # Stakes multiplier
        stakes_mult = self.calculate_stakes_multiplier(is_knockout, is_final, must_win)
        components['stakes'] = stakes_mult
        
        # Calculate overall drama score (0-10 scale)
        base_score = (
            upset_score * 0.25 +
            rivalry_score * 0.15 +
            high_scoring_prob * 0.2 +
            close_game_prob * 0.25 +
            late_drama_prob * 0.15
        )
        
        # Apply stakes multiplier and scale to 0-10
        overall_score = min(base_score * stakes_mult * 10, 10.0)
        
        # Generate narrative
        narrative = self.generate_narrative(
            overall_score, upset_category, rivalry_score, is_knockout
        )
        
        return DramaScoreResult(
            overall_score=round(overall_score, 1),
            upset_potential=upset_category,
            high_scoring_prob=round(high_scoring_prob, 3),
            close_game_prob=round(close_game_prob, 3),
            late_drama_prob=round(late_drama_prob, 3),
            components=components,
            narrative=narrative
        )


def main():
    """Test drama score calculation."""
    predictor = DramaScorePredictor()
    
    # Test with Argentina vs Brazil
    result = predictor.predict(
        home_team="Argentina",
        away_team="Brazil",
        elo_diff=38,  # Argentina slightly favored
        home_win_prob=0.42,
        draw_prob=0.28,
        away_win_prob=0.30,
        exp_home_goals=1.8,
        exp_away_goals=1.5,
        is_knockout=True
    )
    
    console.print(f"\n[bold]Argentina vs Brazil[/bold]")
    console.print(f"Drama Score: {result.overall_score}/10")
    console.print(f"Upset Potential: {result.upset_potential}")
    console.print(f"Narrative: {result.narrative}")


if __name__ == "__main__":
    main()
