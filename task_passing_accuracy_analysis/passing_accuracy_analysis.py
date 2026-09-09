
# Objective 1: Team Passing Accuracy
#
# Analytic question:
#   Did teams at the FIFA World Cup 2026 complete a significantly higher
#   percentage of their passes than teams at the FIFA World Cup 2022?


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# constants
alpha = 0.05          # significance level
SHOW_PLOTS = True     #flag to show plots in a window
OUTPUT_FOLDER = "output"
DATA_FOLDER = "data"

FILE_NAME = os.path.join(DATA_FOLDER, "FIFA_passing_accuracy_2022_2026.xlsx")


#load data
data = pd.read_excel(FILE_NAME)

# create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# data wrangling
# drop any row without a team name or without a recorded pass success value
data = data.dropna(subset=["team", "pass_success_pct"])

# a pass success percentage must lie between 0 and 100
data = data[(data["pass_success_pct"] > 0) & (data["pass_success_pct"] <= 100)]

# the two tournaments being compared
wc2026 = data[data["tournament"] == 2026]["pass_success_pct"]
wc2022 = data[data["tournament"] == 2022]["pass_success_pct"]


# helper function to save and optionally show a plot
def save_and_show(filename):
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    plt.savefig(filepath, dpi=150)

    if SHOW_PLOTS:
        plt.show()

    plt.close()

# descriptive stats
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
    print(f"  outliers       : {[round(v, 1) for v in s[(s < low) | (s > high)]]}")
    print()


print("DESCRIPTIVE STATISTICS - team pass success (%)")
print("=" * 64 + "\n")
describe("World Cup 2026", wc2026)
describe("World Cup 2022", wc2022)

# confidence intervals
def conf_int(s, conf=0.95):
    n = len(s)
    std_error  = s.std() / np.sqrt(n)
    t_critical = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    margin     = t_critical * std_error
    return s.mean() - margin, s.mean() + margin


print("95% CONFIDENCE INTERVALS for the mean")
print("=" * 64 + "\n")
for name, s in [("2026", wc2026), ("2022", wc2022)]:
    lo, hi = conf_int(s)
    print(f"  World Cup {name}: {lo:.3f} to {hi:.3f}")
print()

# Welch's two-sample t-test
# H0: mu_2026  =  mu_2022
# H1: mu_2026  >  mu_2022              (one-tailed)
result = stats.ttest_ind(wc2026, wc2022, equal_var=False, alternative="greater")


print("WELCH'S TWO-SAMPLE t-TEST (one-tailed)")
print("=" * 64 + "\n")
print("  H0: mean pass success 2026 = mean pass success 2022")
print("  H1: mean pass success 2026 > mean pass success 2022\n")
print(f"  difference in means: {wc2026.mean() - wc2022.mean():.3f} percentage points")
print(f"  t statistic        : {result.statistic:.4f}")
print(f"  degrees of freedom : {result.df:.2f}")
print(f"  p value (one-tail) : {result.pvalue:.6f}")
print(f"\n  p < alpha ({alpha}), reject H0" if result.pvalue < alpha
      else f"\n  p > alpha ({alpha}), fail to reject H0")
print()

# robustness check 1
# The 2026 sample is the 32 teams that reached the knockout stage out of 48
# entrants, while the 2022 sample is all 32 entrants. The 2026 group is
# therefore more selected. Restricting both years to the teams that reached
# the round of 16 narrows that difference.
def subset(year, r16):
    m = (data["tournament"] == year) & (data["reached_round_of_16"] == r16)
    return data[m]["pass_success_pct"]

r16_26, r16_22 = subset(2026, True), subset(2022, True)
chk1 = stats.ttest_ind(r16_26, r16_22, equal_var=False, alternative="greater")


print("ROBUSTNESS CHECK 1 - round-of-16 teams only")
print("=" * 64 + "\n")
print(f"  2026: {r16_26.mean():.3f} (n={len(r16_26)})    "
      f"2022: {r16_22.mean():.3f} (n={len(r16_22)})")
print(f"  t = {chk1.statistic:.4f},  p (one-tail) = {chk1.pvalue:.6f}")
print("  -> significant" if chk1.pvalue < alpha
      else "  -> NOT significant at alpha = 0.05")
print()

# robustness check 2
out_26, out_22 = subset(2026, False), subset(2022, False)
chk2 = stats.ttest_ind(out_26, out_22, equal_var=False, alternative="greater")


print("ROBUSTNESS CHECK 2 - teams eliminated before the round of 16")
print("=" * 64 + "\n")
print(f"  2026: {out_26.mean():.3f} (n={len(out_26)})    "
      f"2022: {out_22.mean():.3f} (n={len(out_22)})")
print(f"  t = {chk2.statistic:.4f},  p (one-tail) = {chk2.pvalue:.6f}")
print("  -> significant" if chk2.pvalue < alpha
      else "  -> NOT significant at alpha = 0.05")
print()

# robustness check 3
# Paraguay (67.3%) is a low outlier in 2026. Repeat without it.
no_out = data[(data["tournament"] == 2026) &
              (data["team"] != "Paraguay")]["pass_success_pct"]
chk3 = stats.ttest_ind(no_out, wc2022, equal_var=False, alternative="greater")


print("ROBUSTNESS CHECK 3 - 2026 excluding the Paraguay outlier")
print("=" * 64 + "\n")
print(f"  2026 mean without Paraguay: {no_out.mean():.3f} (n={len(no_out)})")
print(f"  t = {chk3.statistic:.4f},  p (one-tail) = {chk3.pvalue:.6f}\n")

# box plot
plt.figure(figsize=(6.5, 4.5))
plt.boxplot([wc2022, wc2026], tick_labels=["2022\n(n=32)", "2026\n(n=32)"],
            patch_artist=True,
            boxprops=dict(facecolor="#d6e4f0", edgecolor="#2c3e50"),
            medianprops=dict(color="#c0392b", linewidth=2))
plt.title("Team pass success by tournament\nFIFA World Cup 2022 vs 2026")
plt.ylabel("Pass success (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_and_show("boxplot_pass_success.png")

# histogram
plt.figure(figsize=(6.5, 4.5))
bins = np.arange(66, 94, 2)
plt.hist(wc2022, bins=bins, alpha=0.65, label="2022", color="#7a8b99", edgecolor="black")
plt.hist(wc2026, bins=bins, alpha=0.65, label="2026", color="#2e7d5b", edgecolor="black")
plt.axvline(wc2022.mean(), color="#7a8b99", linestyle="--", linewidth=2, label="2022 mean")
plt.axvline(wc2026.mean(), color="#2e7d5b", linestyle="--", linewidth=2, label="2026 mean")
plt.title("Distribution of team pass success\nFIFA World Cup 2022 vs 2026")
plt.xlabel("Pass success (%)")
plt.ylabel("Number of teams")
plt.legend()
plt.tight_layout()
save_and_show("histogram_pass_success.png")

# confidence interval comparison
plt.figure(figsize=(6.5, 3.6))
for i, (name, s, col) in enumerate([("2022", wc2022, "#7a8b99"),
                                    ("2026", wc2026, "#2e7d5b")]):
    lo, hi = conf_int(s)
    plt.plot([lo, hi], [i, i], color=col, linewidth=4, solid_capstyle="round")
    plt.plot(s.mean(), i, "o", color=col, markersize=11,
             markeredgecolor="white", markeredgewidth=1.5)
    plt.text(s.mean(), i + 0.17, f"{s.mean():.2f}%", ha="center", fontsize=10)
plt.yticks([0, 1], ["2022\n(n=32)", "2026\n(n=32)"])
plt.ylim(-0.6, 1.6)
plt.xlabel("Mean pass success % (95% CI)")
plt.title("The two confidence intervals do not overlap")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
save_and_show("confidence_intervals.png")

# where the improvement actually is
plt.figure(figsize=(6.5, 4.2))
labels = ["Reached\nround of 16", "Eliminated\nbefore R16"]
v2022 = [r16_22.mean(), out_22.mean()]
v2026 = [r16_26.mean(), out_26.mean()]
x = np.arange(2); w = 0.35
plt.bar(x - w/2, v2022, w, label="2022", color="#7a8b99", edgecolor="black")
plt.bar(x + w/2, v2026, w, label="2026", color="#2e7d5b", edgecolor="black")
for i in range(2):
    plt.text(x[i] - w/2, v2022[i] + 0.4, f"{v2022[i]:.1f}", ha="center", fontsize=10)
    plt.text(x[i] + w/2, v2026[i] + 0.4, f"{v2026[i]:.1f}", ha="center", fontsize=10)
plt.xticks(x, labels)
plt.ylim(70, 92)
plt.ylabel("Mean pass success (%)")
plt.title("The improvement is concentrated in the weaker teams")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_and_show("improvement_by_stage.png")

print("figures saved: boxplot_pass_success.png, histogram_pass_success.png,")
print("               confidence_intervals.png, improvement_by_stage.png")
