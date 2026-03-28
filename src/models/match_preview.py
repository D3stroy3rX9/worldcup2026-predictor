"""
Match Preview Generator
Creates FIFA-style match previews with historical facts and dramatic commentary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import random

console = Console()


@dataclass
class HeadToHead:
    """Head-to-head statistics between two teams."""
    total_matches: int
    team1_wins: int
    team2_wins: int
    draws: int
    team1_goals: int
    team2_goals: int
    last_5_results: List[str]
    biggest_win_team1: Optional[str]
    biggest_win_team2: Optional[str]
    last_meeting: Optional[str]
    world_cup_meetings: int


@dataclass  
class TeamProfile:
    """Team profile with key facts."""
    name: str
    elo_rating: int
    fifa_rank: int
    confederation: str
    world_cup_wins: int
    world_cup_best: str
    recent_form: List[str]
    goals_per_game: float
    clean_sheet_pct: float
    key_strength: str
    playing_style: str


class MatchPreviewGenerator:
    """Generates cinematic match previews with historical data and commentary."""
    
    WORLD_CUP_WINNERS = {
        "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3,
        "France": 2, "Uruguay": 2, "England": 1, "Spain": 1,
    }
    
    WORLD_CUP_BEST = {
        "Netherlands": "Runner-up (3x)", "Croatia": "Runner-up (2018)",
        "Belgium": "3rd place (2018)", "Portugal": "3rd place (2006)",
        "USA": "3rd place (1930)", "South Korea": "4th place (2002)",
        "Morocco": "4th place (2022)", "Mexico": "Quarter-finals",
        "Japan": "Round of 16", "Senegal": "Quarter-finals (2002)",
        "Ghana": "Quarter-finals (2010)", "Colombia": "Quarter-finals",
        "Switzerland": "Quarter-finals", "Scotland": "Group stage",
        "Canada": "Group stage", "Saudi Arabia": "R16 (1994)",
        "Iran": "Group stage", "Tunisia": "Group stage",
        "Egypt": "Group stage", "Algeria": "R16 (2014)",
        "Ecuador": "R16 (2006)", "Paraguay": "Quarter-finals",
        "Norway": "Round of 16", "Austria": "3rd place (1954)",
        "Jordan": "Debut 2026", "Uzbekistan": "Debut 2026",
        "Cabo Verde": "Debut 2026", "Curaçao": "Debut 2026",
        "Australia": "Round of 16", "New Zealand": "Group stage",
        "Panama": "Group stage", "Haiti": "Group stage (1974)",
        "Cameroon": "Quarter-finals (1990)", "Nigeria": "Round of 16",
        "Côte d'Ivoire": "Group stage",
    }
    
    PLAYING_STYLES = {
        "Brazil": "Samba football - Flair and attacking brilliance",
        "Argentina": "Passion and guile - Technical mastery with fierce determination",
        "Germany": "Efficiency personified - Tactical discipline and clinical finishing",
        "France": "Les Bleus - Explosive pace and individual brilliance",
        "Spain": "Tiki-taka masters - Possession-based beautiful football",
        "England": "Modern pressing - High intensity with technical quality",
        "Portugal": "Technical excellence - Creative flair with defensive solidity",
        "Netherlands": "Total Football - Fluid, attacking, and fearless",
        "Belgium": "Golden generation - Technical quality across the pitch",
        "Italy": "Catenaccio evolved - Defensive mastery with counter-attack",
        "Croatia": "Technical warriors - Midfield excellence and fighting spirit",
        "Uruguay": "La Garra Charrúa - Fighting spirit and defensive steel",
        "Mexico": "El Tri - Quick combinations and passionate football",
        "USA": "High energy - Athletic pressing and direct play",
        "Japan": "Precision football - Organized, technical, disciplined",
        "South Korea": "Relentless energy - High pressing, never-ending running",
        "Morocco": "Atlas Lions - Defensive organization with quick transitions",
        "Senegal": "Teranga Lions - Physical power with technical ability",
        "Colombia": "Los Cafeteros - Creative flair and passionate play",
        "Switzerland": "Compact and efficient - Well-organized defensive unit",
    }
    
    KEY_STRENGTHS = {
        "Brazil": "Attack", "Argentina": "Creativity", "Germany": "Tactics",
        "France": "Talent", "Spain": "Possession", "England": "Depth",
        "Portugal": "Technique", "Netherlands": "Attack", "Belgium": "Midfield",
        "Italy": "Defense", "Croatia": "Midfield", "Uruguay": "Defense",
        "Morocco": "Defense", "USA": "Athleticism", "Japan": "Organization",
        "Mexico": "Attack", "Colombia": "Flair", "Senegal": "Power",
    }
    
    # Peter Drury commentary templates
    COMMENTARY_INTROS = [
        "And here we are... where dreams are made and hearts are broken.",
        "The stage is set. The world watches. {team1} versus {team2}.",
        "Football. The beautiful game. Tonight, it doesn't get more beautiful than this.",
        "They said this day would come. {team1} and {team2}. Finally, face to face.",
        "Can you feel it? The electricity in the air? This is what we live for.",
        "History beckons. Glory awaits. But only one can answer the call tonight.",
        "In stadiums like this, legends are born and immortality is earned.",
        "The world holds its breath. {team1}. {team2}. Let battle commence.",
    ]
    
    RIVALRY_COMMENTARY = [
        "These two giants have danced this dance before. Oh, the memories!",
        "History runs deep between these two. Every meeting writes a new chapter.",
        "When {team1} meets {team2}, form goes out the window. This is about pride.",
        "The rivalry that needs no introduction. The fixture the world circles.",
    ]
    
    UNDERDOG_COMMENTARY = [
        "David meets Goliath. But in football, the slingshot comes in many forms.",
        "{underdog} - unheralded, underestimated, but utterly unafraid.",
        "They don't give {underdog} a chance. Nobody told {underdog} that.",
        "The world expects {favorite}. But {underdog} didn't come to make up numbers.",
        "Against all odds, against all expectations - {underdog} dares to dream.",
    ]
    
    FINAL_COMMENTARY = [
        "THE FINAL. Two nations. One trophy. Ninety minutes to immortality.",
        "This is it. The culmination of everything. The final step to glory.",
        "Every kick from here could echo through eternity.",
        "One lifts the trophy. One wonders what might have been. THE FINAL.",
    ]
    
    KNOCKOUT_COMMENTARY = [
        "Win or go home. There is no tomorrow. This is knockout football.",
        "Every tackle matters. Every chance could be the last. Do or die.",
        "Ninety minutes. Perhaps more. When the whistle blows, one dream ends.",
    ]
    
    def __init__(self, matches_df: pd.DataFrame = None):
        self.matches_df = matches_df
    
    def load_data(self, data_dir: Path):
        """Load historical match data."""
        matches_file = data_dir / "international_results.csv"
        if matches_file.exists():
            self.matches_df = pd.read_csv(matches_file)
            self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
    
    def get_head_to_head(self, team1: str, team2: str) -> HeadToHead:
        """Calculate head-to-head statistics."""
        if self.matches_df is None:
            return self._empty_h2h()
        
        h2h = self.matches_df[
            ((self.matches_df['home_team'] == team1) & (self.matches_df['away_team'] == team2)) |
            ((self.matches_df['home_team'] == team2) & (self.matches_df['away_team'] == team1))
        ].sort_values('date', ascending=False)
        
        if len(h2h) == 0:
            return self._empty_h2h()
        
        team1_wins = team2_wins = draws = team1_goals = team2_goals = 0
        last_5_results = []
        biggest_win_team1 = biggest_win_team2 = None
        biggest_margin_team1 = biggest_margin_team2 = 0
        world_cup_meetings = 0
        
        for _, match in h2h.iterrows():
            is_team1_home = match['home_team'] == team1
            t1_goals = match['home_score'] if is_team1_home else match['away_score']
            t2_goals = match['away_score'] if is_team1_home else match['home_score']
            
            team1_goals += t1_goals
            team2_goals += t2_goals
            
            if t1_goals > t2_goals:
                team1_wins += 1
                result = "W"
                if t1_goals - t2_goals > biggest_margin_team1:
                    biggest_margin_team1 = t1_goals - t2_goals
                    biggest_win_team1 = f"{int(t1_goals)}-{int(t2_goals)} ({match['date'].year})"
            elif t2_goals > t1_goals:
                team2_wins += 1
                result = "L"
                if t2_goals - t1_goals > biggest_margin_team2:
                    biggest_margin_team2 = t2_goals - t1_goals
                    biggest_win_team2 = f"{int(t2_goals)}-{int(t1_goals)} ({match['date'].year})"
            else:
                draws += 1
                result = "D"
            
            if len(last_5_results) < 5:
                last_5_results.append(result)
            
            if 'tournament' in match and 'World Cup' in str(match.get('tournament', '')):
                world_cup_meetings += 1
        
        # Last meeting
        last = h2h.iloc[0]
        t1_g = last['home_score'] if last['home_team'] == team1 else last['away_score']
        t2_g = last['away_score'] if last['home_team'] == team1 else last['home_score']
        winner = team1 if t1_g > t2_g else (team2 if t2_g > t1_g else "Draw")
        last_meeting = f"{int(t1_g)}-{int(t2_g)} {winner} ({last['date'].year})"
        
        return HeadToHead(
            total_matches=len(h2h), team1_wins=team1_wins, team2_wins=team2_wins,
            draws=draws, team1_goals=team1_goals, team2_goals=team2_goals,
            last_5_results=last_5_results, biggest_win_team1=biggest_win_team1,
            biggest_win_team2=biggest_win_team2, last_meeting=last_meeting,
            world_cup_meetings=world_cup_meetings
        )
    
    def _empty_h2h(self) -> HeadToHead:
        return HeadToHead(0, 0, 0, 0, 0, 0, [], None, None, None, 0)
    
    def get_team_profile(self, team: str, elo: int = 1500) -> TeamProfile:
        """Get team profile with key facts."""
        recent_form = self._get_recent_form(team)
        goals_per_game, clean_sheet_pct = self._get_team_stats(team)
        
        wc_wins = self.WORLD_CUP_WINNERS.get(team, 0)
        wc_best = f"{wc_wins}x Champion" if wc_wins > 0 else self.WORLD_CUP_BEST.get(team, "Qualifier")
        
        style = self.PLAYING_STYLES.get(team, "Determined to make their mark on the world stage")
        strength = self.KEY_STRENGTHS.get(team, "Team spirit")
        conf = self._get_confederation(team)
        
        return TeamProfile(
            name=team, elo_rating=elo, fifa_rank=self._elo_to_rank(elo),
            confederation=conf, world_cup_wins=wc_wins, world_cup_best=wc_best,
            recent_form=recent_form, goals_per_game=goals_per_game,
            clean_sheet_pct=clean_sheet_pct, key_strength=strength, playing_style=style
        )
    
    def _get_recent_form(self, team: str, n: int = 5) -> List[str]:
        if self.matches_df is None:
            return ["?"] * n
        
        matches = self.matches_df[
            (self.matches_df['home_team'] == team) | (self.matches_df['away_team'] == team)
        ].sort_values('date', ascending=False).head(n)
        
        form = []
        for _, m in matches.iterrows():
            is_home = m['home_team'] == team
            gf = m['home_score'] if is_home else m['away_score']
            ga = m['away_score'] if is_home else m['home_score']
            form.append("W" if gf > ga else ("L" if gf < ga else "D"))
        return form if form else ["?"] * n
    
    def _get_team_stats(self, team: str) -> Tuple[float, float]:
        if self.matches_df is None:
            return 1.5, 0.3
        
        matches = self.matches_df[
            (self.matches_df['home_team'] == team) | (self.matches_df['away_team'] == team)
        ].tail(20)
        
        if len(matches) == 0:
            return 1.5, 0.3
        
        goals = clean_sheets = 0
        for _, m in matches.iterrows():
            is_home = m['home_team'] == team
            goals += m['home_score'] if is_home else m['away_score']
            if (is_home and m['away_score'] == 0) or (not is_home and m['home_score'] == 0):
                clean_sheets += 1
        
        return round(goals / len(matches), 2), round(clean_sheets / len(matches), 2)
    
    def _get_confederation(self, team: str) -> str:
        confs = {
            "UEFA": ["Germany", "France", "Spain", "England", "Italy", "Portugal", "Netherlands", "Belgium", "Croatia", "Switzerland", "Austria", "Scotland", "Norway"],
            "CONMEBOL": ["Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Ecuador", "Paraguay"],
            "CONCACAF": ["USA", "Mexico", "Canada", "Panama", "Haiti", "Curaçao"],
            "CAF": ["Morocco", "Senegal", "Egypt", "Nigeria", "Algeria", "Tunisia", "Ghana", "Côte d'Ivoire", "Cameroon", "South Africa", "Cabo Verde"],
            "AFC": ["Japan", "South Korea", "Australia", "Iran", "Saudi Arabia", "Qatar", "Uzbekistan", "Jordan"],
        }
        for conf, teams in confs.items():
            if team in teams:
                return conf
        return "FIFA"
    
    def _elo_to_rank(self, elo: int) -> int:
        if elo >= 2050: return 1
        if elo >= 2000: return 3
        if elo >= 1950: return 6
        if elo >= 1900: return 10
        if elo >= 1850: return 15
        if elo >= 1800: return 20
        return 30 + (1800 - elo) // 25
    
    def generate_commentary(self, team1: str, team2: str, h2h: HeadToHead,
                           is_knockout: bool, is_final: bool,
                           team1_elo: int, team2_elo: int) -> List[str]:
        """Generate Peter Drury style commentary lines."""
        lines = []
        
        if is_final:
            lines.append(random.choice(self.FINAL_COMMENTARY))
        elif is_knockout:
            lines.append(random.choice(self.KNOCKOUT_COMMENTARY))
        else:
            lines.append(random.choice(self.COMMENTARY_INTROS).format(team1=team1, team2=team2))
        
        if h2h.total_matches == 0:
            lines.append(f"A FIRST! {team1} and {team2} have NEVER met. Fresh history awaits!")
        elif h2h.total_matches >= 5:
            lines.append(random.choice(self.RIVALRY_COMMENTARY).format(team1=team1, team2=team2))
        
        elo_diff = abs(team1_elo - team2_elo)
        if elo_diff > 150:
            favorite = team1 if team1_elo > team2_elo else team2
            underdog = team2 if team1_elo > team2_elo else team1
            lines.append(random.choice(self.UNDERDOG_COMMENTARY).format(favorite=favorite, underdog=underdog))
        
        return lines
    
    def generate_storylines(self, team1: str, team2: str, h2h: HeadToHead,
                           profile1: TeamProfile, profile2: TeamProfile) -> List[str]:
        """Generate match storylines."""
        storylines = []
        
        if h2h.total_matches == 0:
            storylines.append("🆕 FIRST EVER MEETING between these two nations!")
        
        if profile1.world_cup_best == "Debut 2026":
            storylines.append(f"⭐ WORLD CUP DEBUT for {team1}!")
        if profile2.world_cup_best == "Debut 2026":
            storylines.append(f"⭐ WORLD CUP DEBUT for {team2}!")
        
        rivalries = [("Argentina", "Brazil"), ("England", "Germany"), ("USA", "Mexico"),
                     ("Spain", "Portugal"), ("England", "Argentina"), ("Netherlands", "Germany")]
        if any((team1 in r and team2 in r) for r in rivalries):
            storylines.append(f"⚔️ CLASSIC RIVALRY: {team1} vs {team2} - Football folklore!")
        
        if h2h.total_matches >= 5 and h2h.team1_wins >= h2h.total_matches * 0.6:
            storylines.append(f"👑 {team1} has won {h2h.team1_wins} of last {h2h.total_matches} meetings!")
        elif h2h.total_matches >= 5 and h2h.team2_wins >= h2h.total_matches * 0.6:
            storylines.append(f"👑 {team2} has won {h2h.team2_wins} of last {h2h.total_matches} meetings!")
        
        if profile1.goals_per_game >= 2.0:
            storylines.append(f"⚽ {team1} averages {profile1.goals_per_game} goals/game!")
        if profile2.goals_per_game >= 2.0:
            storylines.append(f"⚽ {team2} averages {profile2.goals_per_game} goals/game!")
        
        return storylines[:4]
    
    def print_preview(self, team1: str, team2: str, 
                      team1_elo: int, team2_elo: int,
                      is_knockout: bool = False, is_final: bool = False):
        """Print a beautiful match preview."""
        
        h2h = self.get_head_to_head(team1, team2)
        profile1 = self.get_team_profile(team1, team1_elo)
        profile2 = self.get_team_profile(team2, team2_elo)
        commentary = self.generate_commentary(team1, team2, h2h, is_knockout, is_final, team1_elo, team2_elo)
        storylines = self.generate_storylines(team1, team2, h2h, profile1, profile2)
        
        # Header
        match_type = "THE FINAL" if is_final else ("KNOCKOUT STAGE" if is_knockout else "GROUP STAGE")
        
        console.print()
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]  {match_type:^56}[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")
        
        # Teams
        console.print()
        console.print(f"      [bold white]{team1:^20}[/bold white]  [bold yellow]VS[/bold yellow]  [bold white]{team2:^20}[/bold white]")
        console.print(f"      [dim]{profile1.confederation:^20}[/dim]      [dim]{profile2.confederation:^20}[/dim]")
        console.print()
        
        # Commentary
        console.print("[yellow]--- THE STAGE IS SET ---[/yellow]\n")
        for line in commentary:
            console.print(f"  [italic white]\"{line}\"[/italic white]")
        console.print()
        
        # Team Profiles
        console.print("[yellow]--- TEAM PROFILES ---[/yellow]\n")
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Stat", style="cyan", width=15)
        table.add_column(team1, justify="center", width=18)
        table.add_column(team2, justify="center", width=18)
        
        table.add_row("ELO Rating", f"[bold]{team1_elo}[/bold]", f"[bold]{team2_elo}[/bold]")
        table.add_row("FIFA Rank", f"#{profile1.fifa_rank}", f"#{profile2.fifa_rank}")
        table.add_row("World Cup", profile1.world_cup_best, profile2.world_cup_best)
        
        form1 = " ".join([f"[green]W[/green]" if r=="W" else f"[red]L[/red]" if r=="L" else f"[yellow]D[/yellow]" for r in profile1.recent_form])
        form2 = " ".join([f"[green]W[/green]" if r=="W" else f"[red]L[/red]" if r=="L" else f"[yellow]D[/yellow]" for r in profile2.recent_form])
        table.add_row("Form (Last 5)", form1, form2)
        table.add_row("Goals/Game", str(profile1.goals_per_game), str(profile2.goals_per_game))
        table.add_row("Clean Sheets", f"{int(profile1.clean_sheet_pct*100)}%", f"{int(profile2.clean_sheet_pct*100)}%")
        table.add_row("Key Strength", profile1.key_strength, profile2.key_strength)
        
        console.print(table)
        
        # Head to Head
        if h2h.total_matches > 0:
            console.print("\n[yellow]--- HEAD TO HEAD ---[/yellow]\n")
            
            console.print(f"  [bold]All-time:[/bold] {h2h.total_matches} matches")
            console.print(f"  [green]{team1} wins: {h2h.team1_wins}[/green] | [yellow]Draws: {h2h.draws}[/yellow] | [red]{team2} wins: {h2h.team2_wins}[/red]")
            console.print(f"  [dim]Goals: {team1} {h2h.team1_goals} - {h2h.team2_goals} {team2}[/dim]")
            
            if h2h.last_meeting:
                console.print(f"\n  [bold]Last meeting:[/bold] {h2h.last_meeting}")
            if h2h.biggest_win_team1:
                console.print(f"  [dim]{team1}'s biggest win: {h2h.biggest_win_team1}[/dim]")
            if h2h.biggest_win_team2:
                console.print(f"  [dim]{team2}'s biggest win: {h2h.biggest_win_team2}[/dim]")
        
        # Storylines
        if storylines:
            console.print("\n[yellow]--- KEY STORYLINES ---[/yellow]\n")
            for story in storylines:
                console.print(f"  {story}")
        
        # Playing Styles
        console.print("\n[yellow]--- PLAYING STYLES ---[/yellow]\n")
        console.print(f"  [cyan]{team1}:[/cyan] {profile1.playing_style}")
        console.print(f"  [cyan]{team2}:[/cyan] {profile2.playing_style}")
        
        console.print()
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")
        console.print()


def main():
    """Test the match preview generator."""
    project_root = Path(__file__).parent.parent.parent
    
    generator = MatchPreviewGenerator()
    generator.load_data(project_root / "data" / "raw")
    
    generator.print_preview("Argentina", "England", 2070, 2025, is_knockout=True)


if __name__ == "__main__":
    main()
