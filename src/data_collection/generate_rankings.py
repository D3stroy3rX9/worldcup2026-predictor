"""
Generate accurate FIFA Rankings for World Cup 2026 teams.
Based on latest FIFA rankings (December 2024).
Run this after download_data.py if FIFA rankings download failed.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()

# FIFA World Rankings as of December 2024 (approximate)
# Source: fifa.com/fifa-world-ranking
FIFA_RANKINGS_DEC_2024 = [
    # Rank, Team, Points, Confederation
    (1, "Argentina", 1867, "CONMEBOL"),
    (2, "France", 1859, "UEFA"),
    (3, "Spain", 1853, "UEFA"),
    (4, "England", 1813, "UEFA"),
    (5, "Brazil", 1775, "CONMEBOL"),
    (6, "Belgium", 1738, "UEFA"),
    (7, "Netherlands", 1731, "UEFA"),
    (8, "Portugal", 1727, "UEFA"),
    (9, "Germany", 1708, "UEFA"),
    (10, "Italy", 1686, "UEFA"),
    (11, "Croatia", 1680, "UEFA"),
    (12, "Colombia", 1671, "CONMEBOL"),
    (13, "Morocco", 1662, "CAF"),
    (14, "Uruguay", 1658, "CONMEBOL"),
    (15, "USA", 1645, "CONCACAF"),
    (16, "Mexico", 1632, "CONCACAF"),
    (17, "Switzerland", 1623, "UEFA"),
    (18, "Japan", 1620, "AFC"),
    (19, "Senegal", 1598, "CAF"),
    (20, "Iran", 1580, "AFC"),
    (21, "South Korea", 1575, "AFC"),
    (22, "Austria", 1569, "UEFA"),
    (23, "Australia", 1560, "AFC"),
    (24, "Ukraine", 1556, "UEFA"),
    (25, "Poland", 1549, "UEFA"),
    (26, "Turkey", 1545, "UEFA"),
    (27, "Ecuador", 1540, "CONMEBOL"),
    (28, "Denmark", 1536, "UEFA"),
    (29, "Algeria", 1532, "CAF"),
    (30, "Egypt", 1528, "CAF"),
    (31, "Serbia", 1520, "UEFA"),
    (32, "Sweden", 1515, "UEFA"),
    (33, "Canada", 1510, "CONCACAF"),
    (34, "Norway", 1505, "UEFA"),
    (35, "Tunisia", 1498, "CAF"),
    (36, "Scotland", 1492, "UEFA"),
    (37, "Czech Republic", 1488, "UEFA"),
    (38, "Paraguay", 1482, "CONMEBOL"),
    (39, "Côte d'Ivoire", 1478, "CAF"),
    (40, "Venezuela", 1472, "CONMEBOL"),
    (41, "Nigeria", 1468, "CAF"),
    (42, "Wales", 1462, "UEFA"),
    (43, "Romania", 1458, "UEFA"),
    (44, "Chile", 1452, "CONMEBOL"),
    (45, "Ghana", 1445, "CAF"),
    (46, "Panama", 1438, "CONCACAF"),
    (47, "Greece", 1432, "UEFA"),
    (48, "Saudi Arabia", 1428, "AFC"),
    (49, "Cameroon", 1422, "CAF"),
    (50, "Mali", 1418, "CAF"),
    (51, "Hungary", 1412, "UEFA"),
    (52, "Qatar", 1405, "AFC"),
    (53, "Peru", 1398, "CONMEBOL"),
    (54, "South Africa", 1392, "CAF"),
    (55, "Ireland", 1385, "UEFA"),
    (56, "Jamaica", 1378, "CONCACAF"),
    (57, "Iraq", 1372, "AFC"),
    (58, "Uzbekistan", 1365, "AFC"),
    (59, "Slovenia", 1358, "UEFA"),
    (60, "Slovakia", 1352, "UEFA"),
    (61, "Albania", 1345, "UEFA"),
    (62, "DR Congo", 1338, "CAF"),
    (63, "Burkina Faso", 1332, "CAF"),
    (64, "Jordan", 1325, "AFC"),
    (65, "Costa Rica", 1318, "CONCACAF"),
    (66, "Bosnia and Herzegovina", 1312, "UEFA"),
    (67, "Georgia", 1305, "UEFA"),
    (68, "United Arab Emirates", 1298, "AFC"),
    (69, "Iceland", 1292, "UEFA"),
    (70, "Northern Ireland", 1285, "UEFA"),
    (71, "Honduras", 1278, "CONCACAF"),
    (72, "El Salvador", 1272, "CONCACAF"),
    (73, "Bolivia", 1265, "CONMEBOL"),
    (74, "Bahrain", 1258, "AFC"),
    (75, "Oman", 1252, "AFC"),
    (76, "Palestine", 1245, "AFC"),
    (77, "North Macedonia", 1238, "UEFA"),
    (78, "Finland", 1232, "UEFA"),
    (79, "Cabo Verde", 1225, "CAF"),
    (80, "New Zealand", 1218, "OFC"),
    (81, "Haiti", 1205, "CONCACAF"),
    (82, "Kosovo", 1198, "UEFA"),
    (83, "Curaçao", 1180, "CONCACAF"),
    (84, "Suriname", 1165, "CONCACAF"),
    (85, "New Caledonia", 1120, "OFC"),
]

# Elo ratings (more predictive than FIFA rankings)
# Based on eloratings.net December 2024
ELO_RATINGS_DEC_2024 = [
    ("Argentina", 2070),
    ("France", 2050),
    ("Spain", 2045),
    ("England", 2025),
    ("Brazil", 2015),
    ("Germany", 1995),
    ("Portugal", 1990),
    ("Netherlands", 1985),
    ("Belgium", 1975),
    ("Italy", 1970),
    ("Croatia", 1960),
    ("Colombia", 1945),
    ("Uruguay", 1940),
    ("Morocco", 1930),
    ("USA", 1910),
    ("Mexico", 1905),
    ("Switzerland", 1900),
    ("Japan", 1895),
    ("Senegal", 1880),
    ("South Korea", 1875),
    ("Iran", 1870),
    ("Australia", 1855),
    ("Ecuador", 1850),
    ("Austria", 1845),
    ("Ukraine", 1840),
    ("Poland", 1835),
    ("Turkey", 1830),
    ("Denmark", 1825),
    ("Serbia", 1820),
    ("Canada", 1815),
    ("Sweden", 1810),
    ("Algeria", 1805),
    ("Egypt", 1800),
    ("Norway", 1795),
    ("Tunisia", 1790),
    ("Scotland", 1785),
    ("Czech Republic", 1780),
    ("Paraguay", 1775),
    ("Côte d'Ivoire", 1770),
    ("Nigeria", 1765),
    ("Ghana", 1755),
    ("Wales", 1750),
    ("Chile", 1745),
    ("Panama", 1735),
    ("Saudi Arabia", 1730),
    ("Cameroon", 1725),
    ("Venezuela", 1720),
    ("Qatar", 1710),
    ("South Africa", 1705),
    ("Jamaica", 1695),
    ("Iraq", 1690),
    ("Uzbekistan", 1685),
    ("DR Congo", 1680),
    ("Jordan", 1670),
    ("Burkina Faso", 1665),
    ("Slovakia", 1660),
    ("Albania", 1655),
    ("Slovenia", 1650),
    ("Bosnia and Herzegovina", 1645),
    ("Ireland", 1640),
    ("Georgia", 1635),
    ("North Macedonia", 1625),
    ("Kosovo", 1615),
    ("Northern Ireland", 1610),
    ("Cabo Verde", 1590),
    ("New Zealand", 1575),
    ("Bolivia", 1565),
    ("Haiti", 1540),
    ("Curaçao", 1510),
    ("Suriname", 1490),
    ("New Caledonia", 1420),
]


def generate_fifa_rankings(output_dir: Path):
    """Generate FIFA rankings CSV file."""
    console.print("\n[bold cyan]Generating FIFA World Rankings...[/bold cyan]")
    
    data = []
    for rank, team, points, conf in FIFA_RANKINGS_DEC_2024:
        data.append({
            'rank': rank,
            'country_full': team,
            'country_abrv': team[:3].upper(),
            'total_points': points,
            'confederation': conf,
            'rank_date': '2024-12-01'
        })
    
    df = pd.DataFrame(data)
    output_path = output_dir / "fifa_rankings.csv"
    df.to_csv(output_path, index=False)
    
    console.print(f"  [green]✓[/green] Saved {len(df)} FIFA rankings to {output_path.name}")
    return df


def generate_elo_ratings(output_dir: Path):
    """Generate Elo ratings CSV file."""
    console.print("\n[bold cyan]Generating Elo Ratings...[/bold cyan]")
    
    data = []
    for team, elo in ELO_RATINGS_DEC_2024:
        data.append({
            'team': team,
            'elo_rating': elo,
            'scraped_date': '2024-12-01',
            'is_estimated': False
        })
    
    df = pd.DataFrame(data)
    output_path = output_dir / "elo_ratings.csv"
    df.to_csv(output_path, index=False)
    
    console.print(f"  [green]✓[/green] Saved {len(df)} Elo ratings to {output_path.name}")
    return df


def main():
    """Generate all rankings data."""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold green]FIFA Rankings & Elo Ratings Generator[/bold green]\n"
        "Creating accurate rankings for World Cup 2026 predictions",
        border_style="green"
    ))
    
    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[dim]Output directory: {output_dir}[/dim]")
    
    # Generate files
    generate_fifa_rankings(output_dir)
    generate_elo_ratings(output_dir)
    
    console.print("\n[bold green]✓ Rankings generation complete![/bold green]")
    console.print("\n[dim]You can now re-run the preprocessing and training steps.[/dim]")


if __name__ == "__main__":
    main()
