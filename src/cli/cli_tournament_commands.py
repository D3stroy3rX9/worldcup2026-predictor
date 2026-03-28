# ============================================
# ADD THIS TO YOUR src/cli/predict.py FILE
# Add this new command after the existing commands
# ============================================

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
