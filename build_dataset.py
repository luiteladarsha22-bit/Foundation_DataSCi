"""
Build the HIT140 Task 1 dataset: FIFA World Cup 2026 team discipline.
Source: FotMob (fotmob.com) - FIFA World Cup 2026 team stats + round-by-round results.
"""
import pandas as pd

# --- Yellow / red card totals: FotMob "Yellow cards" team stats page -------
cards = [
    ("Argentina",13,1), ("Egypt",12,0), ("Canada",11,0), ("Paraguay",9,1),
    ("Ecuador",8,1), ("England",8,1), ("Brazil",8,0), ("Colombia",8,0),
    ("Bosnia and Herzegovina",7,1), ("USA",7,1), ("Curacao",7,0), ("Haiti",7,0),
    ("Morocco",7,0), ("Portugal",7,0), ("Belgium",6,1), ("Switzerland",6,1),
    ("DR Congo",6,0), ("France",6,0), ("Ghana",6,0), ("Iran",6,0),
    ("Saudi Arabia",6,0), ("Spain",6,0), ("South Africa",5,2), ("Uruguay",5,1),
    ("Australia",5,0), ("Austria",5,0), ("Cape Verde",5,0), ("Panama",5,0),
    ("Scotland",5,0), ("Sweden",5,0), ("Qatar",4,2), ("Iraq",4,1),
    ("Mexico",4,1), ("Croatia",4,0), ("Ivory Coast",4,0), ("Japan",4,0),
    ("Jordan",4,0), ("New Zealand",4,0), ("South Korea",4,0), ("Uzbekistan",4,0),
    ("Algeria",3,0), ("Germany",3,0), ("Netherlands",3,0), ("Norway",3,0),
    ("Senegal",3,0), ("Turkiye",2,0), ("Czechia",1,0), ("Tunisia",1,0),
]

# --- Matches played, derived from FotMob round-by-round results ------------
# Every team plays 3 group matches; knockout appearances add to that.
finalists_and_bronze = ["Spain","Argentina","France","England"]              # 8
qf_losers           = ["Morocco","Belgium","Norway","Switzerland"]           # 6
r16_losers          = ["Paraguay","Canada","Portugal","USA",
                       "Brazil","Mexico","Egypt","Colombia"]                 # 5
r32_losers          = ["South Africa","Japan","Germany","Netherlands",
                       "Ivory Coast","Sweden","Ecuador","DR Congo",
                       "Senegal","Bosnia and Herzegovina","Austria","Croatia",
                       "Algeria","Australia","Cape Verde","Ghana"]           # 4

def matches_played(team):
    if team in finalists_and_bronze: return 8
    if team in qf_losers:            return 6
    if team in r16_losers:           return 5
    if team in r32_losers:           return 4
    return 3                                                                 # group-stage exit

# --- Confederation of each team -------------------------------------------
CONF = {
    "UEFA": ["England","Bosnia and Herzegovina","Portugal","Belgium","Switzerland",
             "France","Spain","Austria","Scotland","Sweden","Croatia","Germany",
             "Netherlands","Norway","Turkiye","Czechia"],
    "CONMEBOL": ["Argentina","Paraguay","Ecuador","Brazil","Colombia","Uruguay"],
    "CAF": ["Egypt","Morocco","DR Congo","Ghana","South Africa","Cape Verde",
            "Ivory Coast","Algeria","Senegal","Tunisia"],
    "CONCACAF": ["Canada","USA","Curacao","Haiti","Panama","Mexico"],
    "AFC": ["Iran","Saudi Arabia","Australia","Qatar","Iraq","Japan","Jordan",
            "South Korea","Uzbekistan"],
    "OFC": ["New Zealand"],
}
TEAM_CONF = {t: c for c, teams in CONF.items() for t in teams}

df = pd.DataFrame(cards, columns=["team","yellow_cards","red_cards"])
df["matches_played"]  = df["team"].map(matches_played)
df["confederation"]   = df["team"].map(TEAM_CONF)
df["yc_per_match"]    = df["yellow_cards"] / df["matches_played"]
df = df.sort_values(["confederation","team"]).reset_index(drop=True)

# --- Integrity checks ------------------------------------------------------
assert len(df) == 48,                     f"expected 48 teams, got {len(df)}"
assert df["confederation"].notna().all(), "some team has no confederation"
assert df["matches_played"].sum() == 208, f"team-matches = {df['matches_played'].sum()}, expected 208"
print("checks passed: 48 teams, 208 team-matches (= 104 matches)\n")

print(df.groupby("confederation").agg(
    teams=("team","size"), yc=("yellow_cards","sum"),
    mp=("matches_played","sum"), yc_per_match=("yc_per_match","mean")).round(3))

df.to_excel("FIFA_2026_team_discipline.xlsx", index=False)
print("\nsaved FIFA_2026_team_discipline.xlsx")
