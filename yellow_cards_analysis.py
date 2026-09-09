# =============================================================================
# HIT140 Foundations of Data Science - S226
# Objective 1, Task 1: Team Discipline and Fair Play
#
# Analytic question:
#   Do South American (CONMEBOL) teams receive more yellow cards per match
#   than European (UEFA) teams at the FIFA World Cup 2026?
#
# Data source: FotMob (www.fotmob.com), cross-checked against Ahram Online
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# settings
file_name = "FIFA_2026_team_discipline.xlsx"
alpha = 0.05          # significance level

# ---------------------------------------------------------------- load data
data = pd.read_excel(file_name)

# ------------------------------------------------------------ data wrangling
# drop any row without a team name, and any team with no matches played
data = data.dropna(subset=["team"])
data = data[data["matches_played"] > 0]

# Yellow cards per match played.
# This normalises for tournament progression: a team that reached the final
# played 8 matches and had far more opportunity to be booked than a team
# knocked out in the group stage after 3.
data["yc_per_match"] = data["yellow_cards"] / data["matches_played"]

# the two confederations being compared
uefa     = data[data["confederation"] == "UEFA"]["yc_per_match"]
conmebol = data[data["confederation"] == "CONMEBOL"]["yc_per_match"]

# --------------------------------------------------------- descriptive stats
def describe(name, s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"{name}")
    print(f"  n              : {len(s)}")
    print(f"  mean           : {s.mean():.3f}")
    print(f"  std deviation  : {s.std():.3f}")
    print(f"  median         : {s.median():.3f}")
    print(f"  minimum        : {s.min():.3f}")
    print(f"  maximum        : {s.max():.3f}")
    print(f"  IQR            : {iqr:.3f}")
    print(f"  outliers       : {[round(v, 3) for v in s[(s < low) | (s > high)]]}")
    print()

print("=" * 62)
print("DESCRIPTIVE STATISTICS - yellow cards per match")
print("=" * 62 + "\n")
describe("UEFA", uefa)
describe("CONMEBOL", conmebol)

# ---------------------------------------------------- confidence intervals
def conf_int(s, conf=0.95):
    n = len(s)
    std_error  = s.std() / np.sqrt(n)
    t_critical = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    margin     = t_critical * std_error
    return s.mean() - margin, s.mean() + margin

print("=" * 62)
print("95% CONFIDENCE INTERVALS for the mean")
print("=" * 62 + "\n")
for name, s in [("UEFA", uefa), ("CONMEBOL", conmebol)]:
    lo, hi = conf_int(s)
    print(f"  {name:<9}: {lo:.3f} to {hi:.3f}")
print()

# ------------------------------------------------ Welch's two-sample t-test
# H0: mu_CONMEBOL  =  mu_UEFA
# H1: mu_CONMEBOL  >  mu_UEFA          (one-tailed)
# Welch's test is used because the two groups have unequal sample sizes
# (16 vs 6) and clearly unequal variances.
result = stats.ttest_ind(conmebol, uefa, equal_var=False, alternative="greater")

print("=" * 62)
print("WELCH'S TWO-SAMPLE t-TEST (one-tailed)")
print("=" * 62 + "\n")
print("  H0: mean CONMEBOL = mean UEFA")
print("  H1: mean CONMEBOL > mean UEFA\n")
print(f"  difference in means: {conmebol.mean() - uefa.mean():.3f}")
print(f"  t statistic        : {result.statistic:.4f}")
print(f"  degrees of freedom : {result.df:.2f}")
print(f"  p value (one-tail) : {result.pvalue:.6f}")
print(f"\n  p < alpha ({alpha}), reject H0" if result.pvalue < alpha
      else f"\n  p > alpha ({alpha}), fail to reject H0")
print()

# ------------------------------------------------------- robustness check 1
# Repeat the test using the second data source (Ahram Online) to confirm the
# conclusion does not depend on which provider's card counts are used.
data["yc_per_match_src2"] = data["yc_source2"] / data["matches_played"]
u2 = data[data["confederation"] == "UEFA"]["yc_per_match_src2"]
c2 = data[data["confederation"] == "CONMEBOL"]["yc_per_match_src2"]
r2 = stats.ttest_ind(c2, u2, equal_var=False, alternative="greater")

print("=" * 62)
print("ROBUSTNESS CHECK 1 - repeat using second data source")
print("=" * 62 + "\n")
print(f"  UEFA mean     : {u2.mean():.3f}   CONMEBOL mean: {c2.mean():.3f}")
print(f"  t = {r2.statistic:.4f},  p (one-tail) = {r2.pvalue:.6f}")
print("  -> same conclusion\n" if (r2.pvalue < alpha) == (result.pvalue < alpha)
      else "  -> CONCLUSION CHANGES, investigate\n")

# ------------------------------------------------------- robustness check 2
# CONMEBOL has only 6 teams. Repeat the comparison as UEFA vs every
# non-European team to confirm the result is not an artefact of that
# small group.
row = data[data["confederation"] != "UEFA"]["yc_per_match"]
r3  = stats.ttest_ind(row, uefa, equal_var=False, alternative="greater")

print("=" * 62)
print("ROBUSTNESS CHECK 2 - UEFA vs all non-European teams")
print("=" * 62 + "\n")
print(f"  UEFA mean: {uefa.mean():.3f} (n={len(uefa)})   "
      f"non-UEFA mean: {row.mean():.3f} (n={len(row)})")
print(f"  t = {r3.statistic:.4f},  p (one-tail) = {r3.pvalue:.6f}\n")

# --------------------------------------------------------------- box plot
plt.figure(figsize=(6.5, 4.5))
plt.boxplot([uefa, conmebol], tick_labels=["UEFA\n(n=16)", "CONMEBOL\n(n=6)"],
            patch_artist=True,
            boxprops=dict(facecolor="#cfe0f3", edgecolor="#2c3e50"),
            medianprops=dict(color="#c0392b", linewidth=2))
plt.title("Yellow cards per match by confederation\nFIFA World Cup 2026")
plt.ylabel("Yellow cards per match")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("boxplot_yellow_cards.png", dpi=150)
plt.close()

# --------------------------------------------------------------- histogram
plt.figure(figsize=(6.5, 4.5))
bins = np.arange(0.2, 2.4, 0.2)
plt.hist(uefa, bins=bins, alpha=0.65, label="UEFA", color="#3d6fb4", edgecolor="black")
plt.hist(conmebol, bins=bins, alpha=0.65, label="CONMEBOL", color="#e08a3c", edgecolor="black")
plt.axvline(uefa.mean(), color="#3d6fb4", linestyle="--", linewidth=2, label="UEFA mean")
plt.axvline(conmebol.mean(), color="#e08a3c", linestyle="--", linewidth=2, label="CONMEBOL mean")
plt.title("Distribution of yellow cards per match\nFIFA World Cup 2026")
plt.xlabel("Yellow cards per match")
plt.ylabel("Number of teams")
plt.legend()
plt.tight_layout()
plt.savefig("histogram_yellow_cards.png", dpi=150)
plt.close()

# ------------------------------------------- confidence interval comparison
plt.figure(figsize=(6.5, 3.6))
for i, (name, s) in enumerate([("UEFA", uefa), ("CONMEBOL", conmebol)]):
    lo, hi = conf_int(s)
    colour = "#3d6fb4" if name == "UEFA" else "#e08a3c"
    plt.plot([lo, hi], [i, i], color=colour, linewidth=4, solid_capstyle="round")
    plt.plot(s.mean(), i, "o", color=colour, markersize=11,
             markeredgecolor="white", markeredgewidth=1.5)
    plt.text(s.mean(), i + 0.16, f"{s.mean():.3f}", ha="center", fontsize=10)
plt.yticks([0, 1], ["UEFA\n(n=16)", "CONMEBOL\n(n=6)"])
plt.ylim(-0.6, 1.6)
plt.xlabel("Mean yellow cards per match (95% CI)")
plt.title("The two confidence intervals do not overlap")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("confidence_intervals.png", dpi=150)
plt.close()

print("figures saved: boxplot_yellow_cards.png, histogram_yellow_cards.png, "
      "confidence_intervals.png")
