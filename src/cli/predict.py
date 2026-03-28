#!/usr/bin/env python3
"""
World Cup 2026 Predictor - Command Line Interface
Clean, beautiful predictions
"""

import click
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# Colors
GREEN = "green"
RED = "red"  
BLUE = "blue"
CYAN = "cyan"
YELLOW = "yellow"
DIM = "dim"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def print_banner():
    """Print the application banner."""
    banner = """
    ██╗    ██╗ ██████╗██████╗  ██████╗  ██████╗  ██████╗ 
    ██║    ██║██╔════╝╚════██╗██╔═████╗╚════██╗██╔════╝ 
    ██║ █╗ ██║██║      █████╔╝██║██╔██║ █████╔╝███████╗ 
    ██║███╗██║██║     ██╔═══╝ ████╔╝██║██╔═══╝ ██╔═══██╗
    ╚███╔███╔╝╚██████╗███████╗╚██████╔╝███████╗╚██████╔╝
     ╚══╝╚══╝  ╚═════╝╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝ 
                    PREDICTOR
    """
    console.print(banner, style="bold cyan")


def load_models():
    """Load all trained models silently."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # Suppress all output during loading
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from preprocessing.feature_engineering import FeatureEngineer
        from models.match_predictor import MatchOutcomeModel, ScorelinePredictor
        from models.drama_score import DramaScorePredictor
        
        project_root = get_project_root()
        
        fe = FeatureEngineer(project_root / "data" / "raw")
        fe.load_data()
        
        outcome_model = MatchOutcomeModel(use_gpu=False)  # Use CPU to avoid warnings
        outcome_model.load(project_root / "models" / "match_outcome_model.pkl")
        
        scoreline_model = ScorelinePredictor()
        scoreline_model.load(project_root / "models" / "scoreline_model.pkl")
        
        drama_predictor = DramaScorePredictor()
    
    return fe, outcome_model, scoreline_model, drama_predictor


def prob_bar(prob: float, width: int = 25, color: str = "white") -> str:
    """Create a colored probability bar."""
    filled = int(prob * width)
    empty = width - filled
    bar = "█" * filled + " " * empty
    return f"[{color}]{bar}[/{color}]"


def get_outcome_color(prob: float, is_winner: bool, is_draw: bool = False) -> str:
    """Get color based on outcome."""
    if is_draw:
        return BLUE
    elif is_winner:
        return GREEN
    else:
        return RED


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """World Cup 2026 Match Prediction System"""
    pass


@cli.command()
@click.argument('home_team')
@click.argument('away_team')
@click.option('--knockout', '-k', is_flag=True, help='Knockout match (no draws)')
@click.option('--final', '-f', is_flag=True, help='Final match')
def match(home_team: str, away_team: str, knockout: bool, final: bool):
    """Predict a single match outcome."""
    from scipy.stats import poisson
    
    print_banner()
    
    console.print("  Loading...", style=DIM)
    
    try:
        fe, outcome_model, scoreline_model, drama_predictor = load_models()
    except FileNotFoundError:
        console.print("\n  [red]Models not found. Run setup first:[/red]")
        console.print("     python src/cli/predict.py setup")
        return
    
    # Create features
    match_date = datetime(2026, 6, 15)
    features = fe.create_match_features(
        home_team, away_team, match_date,
        tournament='FIFA World Cup',
        is_knockout=knockout or final
    )
    
    # Get outcome prediction
    outcome = outcome_model.predict(home_team, away_team, features)
    
    # ============================================
    # NEW: Strength-based expected goals calculation
    # ============================================
    home_elo = features.get('home_elo', 1500)
    away_elo = features.get('away_elo', 1500)
    
    # World Cup baseline: higher scoring than average international match
    BASE_GOALS = 1.4
    
    # Calculate strength-adjusted expected goals
    home_strength = (home_elo - 1500) / 400
    away_strength = (away_elo - 1500) / 400
    
    # Strong teams score more, weak teams score less
    exp_home = BASE_GOALS + (home_strength * 0.5) - (away_strength * 0.3)
    exp_away = BASE_GOALS + (away_strength * 0.5) - (home_strength * 0.3)
    
    # Ensure reasonable bounds
    exp_home = max(0.5, min(3.5, exp_home))
    exp_away = max(0.3, min(3.0, exp_away))
    
    # Adjust to match outcome probabilities
    if outcome.home_win_prob > 0.7:
        exp_home = max(exp_home, exp_away + 0.8)
    elif outcome.home_win_prob > 0.6:
        exp_home = max(exp_home, exp_away + 0.5)
    
    if outcome.away_win_prob > 0.7:
        exp_away = max(exp_away, exp_home + 0.8)
    elif outcome.away_win_prob > 0.6:
        exp_away = max(exp_away, exp_home + 0.5)
    
    # ============================================
    # NEW: Generate realistic scorelines from xG
    # ============================================
    def generate_scorelines(xg_home: float, xg_away: float, top_n: int = 50):
        scorelines = []
        for h in range(7):
            for a in range(7):
                prob = poisson.pmf(h, xg_home) * poisson.pmf(a, xg_away)
                scorelines.append((f"{h}-{a}", prob))
        return sorted(scorelines, key=lambda x: x[1], reverse=True)[:top_n]
    
    all_scores = generate_scorelines(exp_home, exp_away)
    
    # Filter scores to match predicted outcome
    if outcome.home_win_prob > outcome.draw_prob and outcome.home_win_prob > outcome.away_win_prob:
        top_scores = [(s, p) for s, p in all_scores if int(s.split('-')[0]) > int(s.split('-')[1])][:6]
    elif outcome.away_win_prob > outcome.draw_prob and outcome.away_win_prob > outcome.home_win_prob:
        top_scores = [(s, p) for s, p in all_scores if int(s.split('-')[1]) > int(s.split('-')[0])][:6]
    else:
        top_scores = [(s, p) for s, p in all_scores if s.split('-')[0] == s.split('-')[1]][:6]
    
    if not top_scores:
        top_scores = all_scores[:6]
    
    # Drama score
    drama = drama_predictor.predict(
        home_team, away_team,
        features['elo_diff'],
        outcome.home_win_prob,
        outcome.draw_prob,
        outcome.away_win_prob,
        exp_home, exp_away,
        is_knockout=knockout,
        is_final=final
    )
    
    # Determine winner for coloring
    home_is_winner = outcome.home_win_prob > outcome.away_win_prob and outcome.home_win_prob > outcome.draw_prob
    away_is_winner = outcome.away_win_prob > outcome.home_win_prob and outcome.away_win_prob > outcome.draw_prob
    draw_is_likely = outcome.draw_prob > outcome.home_win_prob and outcome.draw_prob > outcome.away_win_prob
    
    # Display results
    match_type = "FINAL" if final else ("KNOCKOUT" if knockout else "GROUP STAGE")
    
    console.print(f"\n  [dim]{match_type}[/dim]")
    console.print(f"  [bold white]{home_team}[/bold white] vs [bold white]{away_team}[/bold white]")
    console.print()
    
    # Outcome probabilities with colors
    console.print("  [bold]MATCH OUTCOME[/bold]\n")
    
    if draw_is_likely:
        home_color = DIM
        draw_color = BLUE
        away_color = DIM
    elif home_is_winner:
        home_color = GREEN
        draw_color = DIM
        away_color = RED
    else:
        home_color = RED
        draw_color = DIM
        away_color = GREEN
    
    console.print(f"  {home_team:15} {outcome.home_win_prob*100:5.1f}%  {prob_bar(outcome.home_win_prob, 25, home_color)}")
    if not knockout and not final:
        console.print(f"  {'Draw':15} {outcome.draw_prob*100:5.1f}%  {prob_bar(outcome.draw_prob, 25, draw_color)}")
    console.print(f"  {away_team:15} {outcome.away_win_prob*100:5.1f}%  {prob_bar(outcome.away_win_prob, 25, away_color)}")
    
    # Predicted score
    best_score = top_scores[0][0]
    home_goals, away_goals = best_score.split('-')
    
    console.print(f"\n  [bold]PREDICTED SCORE[/bold]\n")
    console.print(f"  [bold cyan]{home_team} {home_goals}[/bold cyan] - [bold cyan]{away_goals} {away_team}[/bold cyan]")
    console.print(f"  [dim]Expected goals: {exp_home:.1f} - {exp_away:.1f}[/dim]")
    
    # Likely scores table
    console.print(f"\n  [bold]LIKELY SCORELINES[/bold]\n")
    
    score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    score_table.add_column(home_team, justify="center", style="cyan", width=10)
    score_table.add_column(away_team, justify="center", style="cyan", width=10)
    score_table.add_column("Chance", justify="right", style="yellow", width=10)
    
    for score, prob in top_scores:
        h, a = score.split('-')
        score_table.add_row(h, a, f"{prob*100:.1f}%")
    
    console.print(score_table)
    
    # Drama score
    drama_filled = int(drama.overall_score)
    drama_empty = 10 - drama_filled
    drama_bar = f"[yellow]{'*' * drama_filled}[/yellow][dim]{'-' * drama_empty}[/dim]"
    
    console.print(f"\n  [bold]DRAMA SCORE[/bold]  {drama.overall_score}/10  {drama_bar}")
    console.print(f"  [dim]{drama.narrative}[/dim]")
    
    # Confidence
    console.print(f"\n  [bold]CONFIDENCE[/bold]  {outcome.confidence*100:.0f}%")
    console.print()
    """Predict a single match outcome."""
    print_banner()
    
    console.print("  Loading...", style=DIM)
    
    try:
        fe, outcome_model, scoreline_model, drama_predictor = load_models()
    except FileNotFoundError:
        console.print("\n  [red]❌ Models not found. Run setup first:[/red]")
        console.print("     python src/cli/predict.py setup")
        return
    
    
    # Create features
    match_date = datetime(2026, 6, 15)
    features = fe.create_match_features(
        home_team, away_team, match_date,
        tournament='FIFA World Cup',
        is_knockout=knockout or final
    )
    
# Get predictions
    outcome = outcome_model.predict(home_team, away_team, features)
    all_scores = scoreline_model.get_most_likely_scores(features, top_n=50)
    
    # Filter scores to match predicted outcome
    if outcome.home_win_prob > outcome.draw_prob and outcome.home_win_prob > outcome.away_win_prob:
        # Home win - filter for home > away
        top_scores = [(s, p) for s, p in all_scores if int(s.split('-')[0]) > int(s.split('-')[1])][:6]
    elif outcome.away_win_prob > outcome.draw_prob and outcome.away_win_prob > outcome.home_win_prob:
        # Away win - filter for away > home
        top_scores = [(s, p) for s, p in all_scores if int(s.split('-')[1]) > int(s.split('-')[0])][:6]
    else:
        # Draw - filter for equal scores
        top_scores = [(s, p) for s, p in all_scores if s.split('-')[0] == s.split('-')[1]][:6]
    
    # Fallback if no matching scores found
    if not top_scores:
        top_scores = all_scores[:6]
    
    # Expected goals
    exp_home, exp_away = scoreline_model.predict_expected_goals(features)
    
    # Adjust expected goals to match outcome prediction
    if outcome.home_win_prob > outcome.away_win_prob:
        # Home favorite - ensure home xG > away xG
        if exp_home <= exp_away:
            exp_home = exp_away + 0.3 + (outcome.home_win_prob - 0.5) * 1.0
    elif outcome.away_win_prob > outcome.home_win_prob:
        # Away favorite - ensure away xG > home xG
        if exp_away <= exp_home:
            exp_away = exp_home + 0.3 + (outcome.away_win_prob - 0.5) * 1.0
    # Drama score
    drama = drama_predictor.predict(
        home_team, away_team,
        features['elo_diff'],
        outcome.home_win_prob,
        outcome.draw_prob,
        outcome.away_win_prob,
        exp_home, exp_away,
        is_knockout=knockout,
        is_final=final
    )
    
    # Determine winner for coloring
    home_is_winner = outcome.home_win_prob > outcome.away_win_prob and outcome.home_win_prob > outcome.draw_prob
    away_is_winner = outcome.away_win_prob > outcome.home_win_prob and outcome.away_win_prob > outcome.draw_prob
    draw_is_likely = outcome.draw_prob > outcome.home_win_prob and outcome.draw_prob > outcome.away_win_prob
    
    # Display results
    match_type = "FINAL" if final else ("KNOCKOUT" if knockout else "GROUP STAGE")
    
    console.print(f"\n  [dim]{match_type}[/dim]")
    console.print(f"  [bold white]{home_team}[/bold white] vs [bold white]{away_team}[/bold white]")
    console.print()
    
    # Outcome probabilities with colors
    console.print("  [bold]MATCH OUTCOME[/bold]\n")
    
    if draw_is_likely:
        # Draw is most likely - draw is blue, others are gray
        home_color = DIM
        draw_color = BLUE
        away_color = DIM
    elif home_is_winner:
        # Home wins - home green, away red, draw gray
        home_color = GREEN
        draw_color = DIM
        away_color = RED
    else:
        # Away wins - away green, home red, draw gray
        home_color = RED
        draw_color = DIM
        away_color = GREEN
    
    console.print(f"  {home_team:15} {outcome.home_win_prob*100:5.1f}%  {prob_bar(outcome.home_win_prob, 25, home_color)}")
    if not knockout and not final:
        console.print(f"  {'Draw':15} {outcome.draw_prob*100:5.1f}%  {prob_bar(outcome.draw_prob, 25, draw_color)}")
    console.print(f"  {away_team:15} {outcome.away_win_prob*100:5.1f}%  {prob_bar(outcome.away_win_prob, 25, away_color)}")
    
    # Predicted score
    best_score = top_scores[0][0]
    home_goals, away_goals = best_score.split('-')
    
    console.print(f"\n  [bold]PREDICTED SCORE[/bold]\n")
    console.print(f"  [bold cyan]{home_team} {home_goals}[/bold cyan] - [bold cyan]{away_goals} {away_team}[/bold cyan]")
    console.print(f"  [dim]Expected goals: {exp_home:.1f} - {exp_away:.1f}[/dim]")
    
    # Likely scores table
    console.print(f"\n  [bold]LIKELY SCORELINES[/bold]\n")
    
    score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    score_table.add_column(home_team, justify="center", style="cyan", width=10)
    score_table.add_column(away_team, justify="center", style="cyan", width=10)
    score_table.add_column("Chance", justify="right", style="yellow", width=10)
    
    for score, prob in top_scores:
        h, a = score.split('-')
        score_table.add_row(h, a, f"{prob*100:.1f}%")
    
    console.print(score_table)
    
    # Drama score
    drama_filled = int(drama.overall_score)
    drama_empty = 10 - drama_filled
    drama_bar = f"[yellow]{'🔥' * drama_filled}[/yellow][dim]{'-' * drama_empty}[/dim]"
    
    console.print(f"\n  [bold]DRAMA SCORE[/bold]  {drama.overall_score}/10  {drama_bar}")
    console.print(f"  [dim]{drama.narrative}[/dim]")
    
    # Confidence
    console.print(f"\n  [bold]CONFIDENCE[/bold]  {outcome.confidence*100:.0f}%")
    console.print()


@cli.command()
@click.argument('group_name')
def group(group_name: str):
    """Predict all matches in a group."""
    import yaml
    
    print_banner()
    
    project_root = get_project_root()
    groups_file = project_root / "data" / "raw" / "worldcup2026_groups.yaml"
    
    with open(groups_file, 'r') as f:
        data = yaml.safe_load(f)
    
    group_name = group_name.upper()
    if group_name not in data['groups']:
        console.print(f"  [red]❌ Group {group_name} not found[/red]")
        console.print(f"  Available: {', '.join(sorted(data['groups'].keys()))}")
        return
    
    teams = [t['name'] for t in data['groups'][group_name]]
    
    console.print(f"\n  [bold]GROUP {group_name}[/bold]: {', '.join(teams)}\n")
    # console.print("  Loading...", style=DIM)
    
    try:
        fe, outcome_model, scoreline_model, drama_predictor = load_models()
    except FileNotFoundError:
        console.print("  [red]❌ Models not found.[/red]")
        return

    
    # Table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Match", style="white", width=30)
    table.add_column("Score", justify="center", style="bold", width=8)
    table.add_column("Prediction", justify="center", width=20)
    table.add_column("Drama", justify="center", width=8)
    
    match_date = datetime(2026, 6, 15)
    
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1 = teams[i]
            team2 = teams[j]
            
            if team1.startswith('TBD') or team2.startswith('TBD'):
                table.add_row(f"{team1} vs {team2}", "TBD", "-", "-")
                continue
            
            features = fe.create_match_features(
                team1, team2, match_date,
                tournament='FIFA World Cup',
                is_knockout=False
            )
            
            outcome = outcome_model.predict(team1, team2, features)
            top_score = scoreline_model.get_most_likely_scores(features, top_n=1)[0][0]
            
            exp_home, exp_away = scoreline_model.predict_expected_goals(features)
            drama = drama_predictor.predict(
                team1, team2,
                features['elo_diff'],
                outcome.home_win_prob,
                outcome.draw_prob,
                outcome.away_win_prob,
                exp_home, exp_away
            )
            
            # Color the prediction based on favorite
            if outcome.home_win_prob > outcome.away_win_prob:
                pred = f"[green]{team1[:3].upper()}[/green] {outcome.home_win_prob*100:.0f}%"
            elif outcome.away_win_prob > outcome.home_win_prob:
                pred = f"[green]{team2[:3].upper()}[/green] {outcome.away_win_prob*100:.0f}%"
            else:
                pred = f"[blue]Draw[/blue] {outcome.draw_prob*100:.0f}%"
            
            drama_str = f"{'🔥' * min(int(drama.overall_score/2), 5)}"
            
            table.add_row(
                f"{team1} vs {team2}",
                f"[cyan]{top_score}[/cyan]",
                pred,
                drama_str
            )
    
    console.print(table)
    console.print()


@cli.command()
@click.option('--runs', '-r', default=1000, help='Number of simulations')
def simulate(runs: int):
    """Run Monte Carlo tournament simulation."""
    from models.tournament_simulator import TournamentSimulator
    
    print_banner()
    
    console.print(f"\n  [bold]TOURNAMENT SIMULATION[/bold] ({runs:,} runs)\n")
    console.print("  Loading...", style=DIM)
    
    try:
        fe, outcome_model, scoreline_model, _ = load_models()
    except FileNotFoundError:
        console.print("  [red]❌ Models not found.[/red]")
        return
    
    
    project_root = get_project_root()
    groups_file = project_root / "data" / "raw" / "worldcup2026_groups.yaml"
    
    simulator = TournamentSimulator(fe, outcome_model, scoreline_model)
    simulator.load_teams(groups_file)
    
    console.print(f"  Simulating {runs:,} tournaments...\n")
    
    # Run simulation with progress
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    
    with Progress(
        TextColumn("  "),
        BarColumn(bar_width=40),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("", total=runs)
        
        results = {
            'champion': {},
            'finalist': {},
            'semi': {},
        }
        
        for i in range(runs):
            sim = simulator.simulate_tournament()
            
            champ = sim['champion']
            runner = sim['runner_up']
            
            results['champion'][champ] = results['champion'].get(champ, 0) + 1
            results['finalist'][champ] = results['finalist'].get(champ, 0) + 1
            results['finalist'][runner] = results['finalist'].get(runner, 0) + 1
            
            for team in sim['semi_finalists']:
                results['semi'][team] = results['semi'].get(team, 0) + 1
            
            progress.update(task, advance=1)
    
    # Display results
    console.print(f"\n  [bold]CHAMPIONSHIP ODDS[/bold]\n")
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Team", style="white", width=15)
    table.add_column("Win", justify="right", style="green bold", width=8)
    table.add_column("Final", justify="right", style="yellow", width=8)
    table.add_column("Semi", justify="right", style="cyan", width=8)
    
    sorted_champs = sorted(results['champion'].items(), key=lambda x: x[1], reverse=True)
    
    for i, (team, count) in enumerate(sorted_champs[:15], 1):
        win_pct = count / runs * 100
        final_pct = results['finalist'].get(team, 0) / runs * 100
        semi_pct = results['semi'].get(team, 0) / runs * 100
        
        table.add_row(
            str(i),
            team,
            f"{win_pct:.1f}%",
            f"{final_pct:.1f}%",
            f"{semi_pct:.1f}%"
        )
    
    console.print(table)
    console.print()


@cli.command()
def teams():
    """List all teams in the tournament."""
    import yaml
    
    print_banner()
    
    project_root = get_project_root()
    groups_file = project_root / "data" / "raw" / "worldcup2026_groups.yaml"
    
    with open(groups_file, 'r') as f:
        data = yaml.safe_load(f)
    
    console.print(f"\n  [bold]WORLD CUP 2026 TEAMS[/bold]\n")
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Group", style="cyan bold", width=8)
    table.add_column("Teams", style="white")
    
    for group_name in sorted(data['groups'].keys()):
        teams = data['groups'][group_name]
        team_names = [t['name'] for t in teams]
        table.add_row(group_name, ", ".join(team_names))
    
    console.print(table)
    console.print()


@cli.command()
def setup():
    """Run full setup: download, preprocess, and train."""
    print_banner()
    
    console.print("\n  [bold]SETUP[/bold]\n")
    
    console.print("  [cyan]Step 1/3:[/cyan] Downloading data...")
    from data_collection.download_data import main as download_main
    download_main()
    
    console.print("\n  [cyan]Step 2/3:[/cyan] Building features...")
    from preprocessing.feature_engineering import main as preprocess_main
    preprocess_main()
    
    console.print("\n  [cyan]Step 3/3:[/cyan] Training models...")
    from models.match_predictor import main as train_main
    train_main()
    
    console.print("\n  [green]✅ Setup complete![/green]")
    console.print("\n  Try: python src/cli/predict.py match \"Brazil\" \"Germany\"\n")

@cli.command()
@click.option('--verbose/--quiet', '-v/-q', default=True, help='Show detailed output')
@click.option('--save', '-s', is_flag=True, help='Save predictions to CSV')
def tournament(verbose: bool, save: bool):
    """Simulate the entire World Cup 2026 tournament match-by-match."""
    from models.dynamic_tournament import DynamicTournament
    
    print_banner()
    
    console.print("\n  [dim]Loading models...[/dim]")
    
    try:
        fe, outcome_model, scoreline_model, _ = load_models()
    except FileNotFoundError:
        console.print("  [red]Models not found. Run setup first.[/red]")
        return
    
    project_root = get_project_root()
    
    # Create tournament
    tournament_sim = DynamicTournament(fe, outcome_model, scoreline_model)
    tournament_sim.load_groups(project_root / "data" / "raw" / "worldcup2026_groups.yaml")
    
    # Check for actual results
    results_file = project_root / "data" / "actual_results.csv"
    if results_file.exists():
        tournament_sim.load_actual_results(results_file)
        console.print(f"  [green]✓[/green] Loaded actual results from {results_file.name}")
    
    # Run simulation
    champion = tournament_sim.simulate_tournament(verbose=verbose)
    
    # Save if requested
    if save:
        output_dir = project_root / "data" / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        tournament_sim.save_predictions(output_dir / "tournament_predictions.csv")
    
    console.print()


@cli.command()
@click.argument('home_team')
@click.argument('away_team')
@click.argument('home_score', type=int)
@click.argument('away_score', type=int)
@click.option('--stage', '-s', default='Group', help='Match stage (Group/Round of 16/etc)')
def add_result(home_team: str, away_team: str, home_score: int, away_score: int, stage: str):
    """Add an actual match result for tournament updates.
    
    Example: python src/cli/predict.py add-result "Brazil" "Morocco" 2 0 --stage "Group C"
    """
    import csv
    
    project_root = get_project_root()
    results_file = project_root / "data" / "actual_results.csv"
    
    # Create file with header if it doesn't exist
    if not results_file.exists():
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['home_team', 'away_team', 'home_score', 'away_score', 'stage'])
    
    # Append result
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([home_team, away_team, home_score, away_score, stage])
    
    console.print(f"[green]✓[/green] Added: {home_team} {home_score} - {away_score} {away_team} ({stage})")
    console.print(f"[dim]Saved to: {results_file}[/dim]")


@cli.command()
def results():
    """Show all recorded actual results."""
    import pandas as pd
    
    project_root = get_project_root()
    results_file = project_root / "data" / "actual_results.csv"
    
    if not results_file.exists():
        console.print("[yellow]No actual results recorded yet.[/yellow]")
        console.print("[dim]Use 'add-result' command to add match results.[/dim]")
        return
    
    df = pd.read_csv(results_file)
    
    if len(df) == 0:
        console.print("[yellow]No actual results recorded yet.[/yellow]")
        return
    
    print_banner()
    console.print("\n[bold]RECORDED RESULTS[/bold]\n")
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="dim", width=12)
    table.add_column("Home", width=15)
    table.add_column("Score", justify="center", style="bold green", width=7)
    table.add_column("Away", width=15)
    
    for _, row in df.iterrows():
        table.add_row(
            row['stage'],
            row['home_team'],
            f"{row['home_score']} - {row['away_score']}",
            row['away_team']
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(df)} matches[/dim]\n")

# ============================================
# ADD THIS TO YOUR src/cli/predict.py FILE
# Add this import at the top with other imports:
#     from models.match_preview import MatchPreviewGenerator
#
# Then add this command before 'if __name__ == "__main__":'
# ============================================

@cli.command()
@click.argument('home_team')
@click.argument('away_team')
@click.option('--knockout', '-k', is_flag=True, help='Knockout match')
@click.option('--final', '-f', is_flag=True, help='Final match')
def preview(home_team: str, away_team: str, knockout: bool, final: bool):
    """Show a cinematic FIFA-style match preview with history and commentary."""
    from models.match_preview import MatchPreviewGenerator
    
    print_banner()
    
    console.print("  [dim]Loading data...[/dim]")
    
    try:
        fe, outcome_model, _, _ = load_models()
    except FileNotFoundError:
        console.print("  [red]Models not found. Run setup first.[/red]")
        return
    
    project_root = get_project_root()
    
    # Create features to get Elo ratings
    match_date = datetime(2026, 6, 15)
    features = fe.create_match_features(
        home_team, away_team, match_date,
        tournament='FIFA World Cup',
        is_knockout=knockout or final
    )
    
    home_elo = int(features.get('home_elo', 1500))
    away_elo = int(features.get('away_elo', 1500))
    
    # Generate preview
    preview_gen = MatchPreviewGenerator()
    preview_gen.load_data(project_root / "data" / "raw")
    
    preview_gen.print_preview(
        home_team, away_team,
        home_elo, away_elo,
        is_knockout=knockout,
        is_final=final
    )
    
    # Now show the prediction
    console.print("[yellow]--- PREDICTION ---[/yellow]\n")
    
    # Get outcome prediction (reuse the match command logic)
    from scipy.stats import poisson
    
    outcome = outcome_model.predict(home_team, away_team, features)
    
    # Calculate xG
    BASE_GOALS = 1.4
    home_strength = (home_elo - 1500) / 400
    away_strength = (away_elo - 1500) / 400
    
    exp_home = BASE_GOALS + (home_strength * 0.5) - (away_strength * 0.3)
    exp_away = BASE_GOALS + (away_strength * 0.5) - (home_strength * 0.3)
    exp_home = max(0.5, min(3.5, exp_home))
    exp_away = max(0.3, min(3.0, exp_away))
    
    if outcome.home_win_prob > 0.7:
        exp_home = max(exp_home, exp_away + 0.8)
    elif outcome.home_win_prob > 0.6:
        exp_home = max(exp_home, exp_away + 0.5)
    if outcome.away_win_prob > 0.7:
        exp_away = max(exp_away, exp_home + 0.8)
    elif outcome.away_win_prob > 0.6:
        exp_away = max(exp_away, exp_home + 0.5)
    
    # Generate scoreline
    def generate_scorelines(xg_home, xg_away, top_n=10):
        scores = []
        for h in range(7):
            for a in range(7):
                prob = poisson.pmf(h, xg_home) * poisson.pmf(a, xg_away)
                scores.append((f"{h}-{a}", prob))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    all_scores = generate_scorelines(exp_home, exp_away)
    
    if outcome.home_win_prob > outcome.draw_prob and outcome.home_win_prob > outcome.away_win_prob:
        top_scores = [(s, p) for s, p in all_scores if int(s.split('-')[0]) > int(s.split('-')[1])][:5]
    elif outcome.away_win_prob > outcome.draw_prob:
        top_scores = [(s, p) for s, p in all_scores if int(s.split('-')[1]) > int(s.split('-')[0])][:5]
    else:
        top_scores = [(s, p) for s, p in all_scores if s.split('-')[0] == s.split('-')[1]][:5]
    
    if not top_scores:
        top_scores = all_scores[:5]
    
    best_score = top_scores[0][0]
    home_goals, away_goals = best_score.split('-')
    
    # Display
    console.print(f"  [bold]Outcome Probabilities:[/bold]")
    console.print(f"    {home_team}: [green]{outcome.home_win_prob*100:.1f}%[/green]")
    if not knockout and not final:
        console.print(f"    Draw: [yellow]{outcome.draw_prob*100:.1f}%[/yellow]")
    console.print(f"    {away_team}: [red]{outcome.away_win_prob*100:.1f}%[/red]")
    
    console.print(f"\n  [bold]Predicted Score:[/bold] [cyan]{home_team} {home_goals} - {away_goals} {away_team}[/cyan]")
    console.print(f"  [dim]Expected Goals: {exp_home:.1f} - {exp_away:.1f}[/dim]")
    
    console.print(f"\n  [bold]Most Likely Scores:[/bold]")
    for score, prob in top_scores[:5]:
        h, a = score.split('-')
        console.print(f"    {h} - {a}  [yellow]{prob*100:.1f}%[/yellow]")
    
    console.print()


if __name__ == '__main__':
    cli()