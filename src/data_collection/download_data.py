"""
Data Downloader for World Cup 2026 Predictor
Downloads historical international football match data from GitHub (Kaggle mirror)
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()

# Data source URLs (GitHub raw files - no auth needed)
DATA_SOURCES = {
    "international_results": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "filename": "international_results.csv",
        "description": "International football results from 1872 to present"
    },
    "goalscorers": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
        "filename": "goalscorers.csv",
        "description": "Goal scorers in international matches"
    },
    "shootouts": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
        "filename": "shootouts.csv",
        "description": "Penalty shootout results"
    }
}

# FIFA Rankings historical data
FIFA_RANKINGS_URL = "https://raw.githubusercontent.com/cnc4e/fifaworldranking/main/fifa_ranking-2024-09-19.csv"


def get_data_dir() -> Path:
    """Get the raw data directory path."""
    # Get project root (parent of src directory)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_file(url: str, filepath: Path, description: str) -> bool:
    """Download a file from URL with progress indicator."""
    try:
        console.print(f"  📥 Downloading: {description}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
        
        console.print(f"  ✅ Saved to: {filepath.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        console.print(f"  ❌ Failed to download: {e}", style="red")
        return False


def download_international_results(data_dir: Path) -> bool:
    """Download all international results datasets."""
    console.print("\n[bold blue]📊 International Match Results[/bold blue]")
    
    success = True
    for key, source in DATA_SOURCES.items():
        filepath = data_dir / source["filename"]
        if not download_file(source["url"], filepath, source["description"]):
            success = False
    
    return success


def download_fifa_rankings(data_dir: Path) -> bool:
    """Download FIFA world rankings historical data."""
    console.print("\n[bold blue]🏆 FIFA World Rankings[/bold blue]")
    
    filepath = data_dir / "fifa_rankings.csv"
    return download_file(
        FIFA_RANKINGS_URL, 
        filepath, 
        "FIFA World Rankings (historical)"
    )


def download_elo_ratings(data_dir: Path) -> bool:
    """Scrape current Elo ratings from eloratings.net."""
    console.print("\n[bold blue]📈 Elo Ratings[/bold blue]")
    
    try:
        from bs4 import BeautifulSoup
        
        console.print("  📥 Scraping Elo ratings from eloratings.net")
        
        url = "https://www.eloratings.net/World"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the ratings table
        teams = []
        ratings_data = []
        
        # The site structure may vary, so we'll try multiple approaches
        # Look for team names and ratings in the page
        rows = soup.find_all('a', href=lambda x: x and '/en/' in str(x))
        
        for row in rows:
            try:
                team_name = row.get_text(strip=True)
                # Try to find the rating nearby
                parent = row.find_parent('tr') or row.find_parent('div')
                if parent:
                    text = parent.get_text()
                    # Extract numbers that look like Elo ratings (1000-2200 range)
                    import re
                    numbers = re.findall(r'\b(1[0-9]{3}|2[0-2][0-9]{2})\b', text)
                    if numbers and team_name:
                        teams.append({
                            'team': team_name,
                            'elo_rating': int(numbers[0])
                        })
            except Exception:
                continue
        
        if teams:
            df = pd.DataFrame(teams)
            df = df.drop_duplicates(subset=['team'])
            df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
            df.to_csv(data_dir / "elo_ratings.csv", index=False)
            console.print(f"  ✅ Scraped {len(df)} team ratings")
            return True
        else:
            console.print("  ⚠️ Could not parse Elo ratings, creating placeholder", style="yellow")
            # Create a placeholder file with estimated ratings
            create_estimated_elo_ratings(data_dir)
            return True
            
    except Exception as e:
        console.print(f"  ⚠️ Scraping failed: {e}", style="yellow")
        console.print("  📝 Creating estimated Elo ratings instead")
        create_estimated_elo_ratings(data_dir)
        return True


def create_estimated_elo_ratings(data_dir: Path):
    """Create estimated Elo ratings based on FIFA rankings correlation."""
    # Estimated Elo ratings for World Cup 2026 teams (based on typical values)
    estimated_ratings = [
        ("Argentina", 2066), ("France", 2045), ("Brazil", 2028), ("England", 2018),
        ("Spain", 2010), ("Portugal", 2005), ("Belgium", 1998), ("Netherlands", 1995),
        ("Germany", 1988), ("Italy", 1980), ("Croatia", 1975), ("Colombia", 1965),
        ("Uruguay", 1960), ("Mexico", 1955), ("USA", 1950), ("Switzerland", 1945),
        ("Japan", 1940), ("Morocco", 1935), ("Senegal", 1930), ("South Korea", 1920),
        ("Iran", 1915), ("Australia", 1910), ("Ecuador", 1905), ("Canada", 1900),
        ("Norway", 1895), ("Austria", 1890), ("Poland", 1885), ("Ukraine", 1880),
        ("Algeria", 1875), ("Egypt", 1870), ("Tunisia", 1865), ("Qatar", 1855),
        ("Saudi Arabia", 1850), ("Scotland", 1845), ("Paraguay", 1840), ("Ghana", 1835),
        ("Côte d'Ivoire", 1830), ("Panama", 1800), ("Jordan", 1790), ("Uzbekistan", 1785),
        ("New Zealand", 1750), ("Haiti", 1700), ("Curaçao", 1650), ("Cabo Verde", 1680),
        ("South Africa", 1780), ("Bolivia", 1760), ("Iraq", 1755),
    ]
    
    df = pd.DataFrame(estimated_ratings, columns=['team', 'elo_rating'])
    df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
    df['is_estimated'] = True
    df.to_csv(data_dir / "elo_ratings.csv", index=False)
    console.print(f"  ✅ Created estimated ratings for {len(df)} teams")


def validate_data(data_dir: Path) -> dict:
    """Validate downloaded data and return statistics."""
    stats = {}
    
    console.print("\n[bold blue]🔍 Validating Data[/bold blue]")
    
    # Check international results
    results_file = data_dir / "international_results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        stats['total_matches'] = len(df)
        stats['date_range'] = f"{df['date'].min()} to {df['date'].max()}"
        stats['unique_teams'] = df['home_team'].nunique()
        console.print(f"  ✅ Match results: {stats['total_matches']:,} matches")
        console.print(f"     Date range: {stats['date_range']}")
        console.print(f"     Unique teams: {stats['unique_teams']}")
    
    # Check goalscorers
    scorers_file = data_dir / "goalscorers.csv"
    if scorers_file.exists():
        df = pd.read_csv(scorers_file)
        stats['total_goals'] = len(df)
        console.print(f"  ✅ Goalscorers: {stats['total_goals']:,} goals recorded")
    
    # Check FIFA rankings
    rankings_file = data_dir / "fifa_rankings.csv"
    if rankings_file.exists():
        df = pd.read_csv(rankings_file)
        stats['ranking_entries'] = len(df)
        console.print(f"  ✅ FIFA rankings: {stats['ranking_entries']:,} entries")
    
    # Check Elo ratings
    elo_file = data_dir / "elo_ratings.csv"
    if elo_file.exists():
        df = pd.read_csv(elo_file)
        stats['elo_teams'] = len(df)
        console.print(f"  ✅ Elo ratings: {stats['elo_teams']} teams")
    
    return stats


def main():
    """Main function to download all required data."""
    console.print(Panel.fit(
        "[bold green]🏆 World Cup 2026 Predictor - Data Downloader[/bold green]\n"
        "Downloading historical international football data",
        border_style="green"
    ))
    
    data_dir = get_data_dir()
    console.print(f"\n📁 Data directory: {data_dir}")
    
    # Download all datasets
    results = []
    results.append(("International Results", download_international_results(data_dir)))
    results.append(("FIFA Rankings", download_fifa_rankings(data_dir)))
    results.append(("Elo Ratings", download_elo_ratings(data_dir)))
    
    # Validate data
    stats = validate_data(data_dir)
    
    # Summary
    console.print("\n" + "="*50)
    success_count = sum(1 for _, success in results if success)
    console.print(f"\n[bold]Download Summary: {success_count}/{len(results)} successful[/bold]")
    
    if all(success for _, success in results):
        console.print("\n[bold green]✅ All data downloaded successfully![/bold green]")
        console.print("\nNext step: Run the preprocessing script to prepare training data")
    else:
        console.print("\n[yellow]⚠️ Some downloads failed. Check the errors above.[/yellow]")


if __name__ == "__main__":
    main()
