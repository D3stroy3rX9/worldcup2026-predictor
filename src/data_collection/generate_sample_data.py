"""
Generate sample training data for demonstration purposes.
This creates synthetic historical match data that mimics real patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console

console = Console()

# Team pools by strength tier (Elo-like ratings)
TEAM_TIERS = {
    'tier1': [  # Elite teams (Elo 2000+)
        ('Argentina', 2066), ('France', 2045), ('Brazil', 2028), 
        ('England', 2018), ('Spain', 2010), ('Portugal', 2005),
    ],
    'tier2': [  # Strong teams (Elo 1900-2000)
        ('Belgium', 1998), ('Netherlands', 1995), ('Germany', 1988),
        ('Italy', 1980), ('Croatia', 1975), ('Colombia', 1965),
        ('Uruguay', 1960), ('Mexico', 1955), ('USA', 1950),
    ],
    'tier3': [  # Good teams (Elo 1800-1900)
        ('Switzerland', 1945), ('Japan', 1940), ('Morocco', 1935),
        ('Senegal', 1930), ('South Korea', 1920), ('Iran', 1915),
        ('Australia', 1910), ('Ecuador', 1905), ('Canada', 1900),
        ('Norway', 1895), ('Austria', 1890), ('Poland', 1885),
    ],
    'tier4': [  # Average teams (Elo 1700-1800)
        ('Ukraine', 1880), ('Algeria', 1875), ('Egypt', 1870),
        ('Tunisia', 1865), ('Qatar', 1855), ('Saudi Arabia', 1850),
        ('Scotland', 1845), ('Paraguay', 1840), ('Ghana', 1835),
        ("Côte d'Ivoire", 1830), ('South Africa', 1780),
    ],
    'tier5': [  # Weaker teams (Elo below 1700)
        ('Panama', 1800), ('Jordan', 1790), ('Uzbekistan', 1785),
        ('Bolivia', 1760), ('New Zealand', 1750), ('Haiti', 1700),
        ('Cabo Verde', 1680), ('Curaçao', 1650),
    ],
}


def generate_match_result(home_elo: float, away_elo: float) -> tuple:
    """Generate a realistic match result based on Elo difference."""
    elo_diff = home_elo - away_elo
    
    # Win probability based on Elo (logistic function)
    home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Add home advantage (~100 Elo points)
    home_win_prob = 1 / (1 + 10 ** (-(elo_diff + 100) / 400))
    
    # Draw probability (higher for closer matches)
    draw_prob = 0.25 * (1 - abs(home_win_prob - 0.5) * 2)
    
    # Normalize
    away_win_prob = 1 - home_win_prob - draw_prob
    away_win_prob = max(0.05, away_win_prob)
    home_win_prob = max(0.05, home_win_prob)
    
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total
    draw_prob /= total
    away_win_prob /= total
    
    # Sample outcome
    outcome = np.random.choice(['home', 'draw', 'away'], 
                               p=[home_win_prob, draw_prob, away_win_prob])
    
    # Generate goals (Poisson-based)
    base_home_goals = 1.5 + (elo_diff / 500)
    base_away_goals = 1.2 - (elo_diff / 500)
    
    base_home_goals = max(0.5, min(3.0, base_home_goals))
    base_away_goals = max(0.3, min(2.5, base_away_goals))
    
    home_goals = np.random.poisson(base_home_goals)
    away_goals = np.random.poisson(base_away_goals)
    
    # Ensure outcome matches
    if outcome == 'home' and home_goals <= away_goals:
        home_goals = away_goals + 1
    elif outcome == 'away' and away_goals <= home_goals:
        away_goals = home_goals + 1
    elif outcome == 'draw':
        away_goals = home_goals
    
    # Cap goals at reasonable values
    home_goals = min(home_goals, 7)
    away_goals = min(away_goals, 7)
    
    return home_goals, away_goals


def generate_sample_data(n_matches: int = 5000) -> pd.DataFrame:
    """Generate sample international match data."""
    
    console.print(f"[bold blue]🔧 Generating {n_matches:,} sample matches...[/bold blue]")
    
    # Flatten teams
    all_teams = []
    for tier, teams in TEAM_TIERS.items():
        all_teams.extend(teams)
    
    team_dict = {name: elo for name, elo in all_teams}
    team_names = list(team_dict.keys())
    
    matches = []
    tournaments = [
        'FIFA World Cup', 'FIFA World Cup qualification',
        'UEFA Euro', 'UEFA Euro qualification',
        'Copa América', 'African Cup of Nations',
        'AFC Asian Cup', 'CONCACAF Gold Cup',
        'UEFA Nations League', 'Friendly'
    ]
    
    # Generate matches from 2010 to 2024
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for i in range(n_matches):
        # Random date
        days_diff = (end_date - start_date).days
        match_date = start_date + timedelta(days=np.random.randint(0, days_diff))
        
        # Random teams (weighted by tier - higher tier teams play more)
        home_team = np.random.choice(team_names)
        away_team = np.random.choice([t for t in team_names if t != home_team])
        
        home_elo = team_dict[home_team]
        away_elo = team_dict[away_team]
        
        # Generate result
        home_score, away_score = generate_match_result(home_elo, away_elo)
        
        # Random tournament (weighted)
        tournament = np.random.choice(
            tournaments,
            p=[0.05, 0.2, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05, 0.1, 0.25]
        )
        
        matches.append({
            'date': match_date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'tournament': tournament,
            'city': 'Various',
            'country': 'Various',
            'neutral': np.random.random() < 0.3
        })
    
    df = pd.DataFrame(matches)
    df = df.sort_values('date').reset_index(drop=True)
    
    console.print(f"  ✅ Generated {len(df):,} matches")
    console.print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    console.print(f"  Unique teams: {df['home_team'].nunique()}")
    
    return df


def main():
    """Generate and save sample data."""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold green]🏆 Sample Data Generator[/bold green]\n"
        "Creating synthetic training data for demonstration",
        border_style="green"
    ))
    
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_sample_data(n_matches=5000)
    
    # Save
    output_path = raw_data_dir / "international_results.csv"
    df.to_csv(output_path, index=False)
    console.print(f"\n  💾 Saved to: {output_path}")
    
    # Also create Elo ratings file
    elo_data = []
    for tier, teams in TEAM_TIERS.items():
        for name, elo in teams:
            elo_data.append({
                'team': name,
                'elo_rating': elo,
                'scraped_date': datetime.now().strftime('%Y-%m-%d'),
                'is_estimated': False
            })
    
    elo_df = pd.DataFrame(elo_data)
    elo_path = raw_data_dir / "elo_ratings.csv"
    elo_df.to_csv(elo_path, index=False)
    console.print(f"  💾 Saved Elo ratings to: {elo_path}")
    
    # Create simple FIFA rankings
    fifa_data = []
    for i, (name, elo) in enumerate(sorted(
        [(n, e) for teams in TEAM_TIERS.values() for n, e in teams],
        key=lambda x: x[1], reverse=True
    ), 1):
        fifa_data.append({
            'rank': i,
            'country_full': name,
            'country_abrv': name[:3].upper(),
            'total_points': elo * 0.8,  # Approximate FIFA points
            'rank_date': '2024-12-01'
        })
    
    fifa_df = pd.DataFrame(fifa_data)
    fifa_path = raw_data_dir / "fifa_rankings.csv"
    fifa_df.to_csv(fifa_path, index=False)
    console.print(f"  💾 Saved FIFA rankings to: {fifa_path}")
    
    console.print("\n[bold green]✅ Sample data generation complete![/bold green]")


if __name__ == "__main__":
    main()
