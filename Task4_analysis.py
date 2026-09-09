from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "eligible_starting_midfielders.csv"
OUTPUT_DIR = ROOT / "results"

SEED = 42
SAMPLE_SIZE = 25
ALPHA = 0.05
GROUPS = ["90-minute", "Extra time"]


def confidence_interval(values):
    values = pd.Series(values).dropna().astype(float)
    mean = values.mean()
    standard_error = stats.sem(values)
    critical_value = stats.t.ppf(0.975, len(values) - 1)
    lower = mean - critical_value * standard_error
    upper = mean + critical_value * standard_error
    return lower, upper


def choose_sample(data):
    samples = []

    for condition in GROUPS:
        group = data[data["match_condition"] == condition].copy()
        group = group.sample(frac=1, random_state=SEED)
        group = group.drop_duplicates("player_id")

        if len(group) < SAMPLE_SIZE:
            raise ValueError(f"Not enough observations for {condition}")

        sample = group.sample(n=SAMPLE_SIZE, random_state=SEED)
        samples.append(sample)

    return pd.concat(samples, ignore_index=True)


def make_summary(sample):
    rows = []

    for condition in GROUPS:
        values = sample.loc[
            sample["match_condition"] == condition,
            "total_distance_km",
        ]
        lower, upper = confidence_interval(values)

        rows.append(
            {
                "condition": condition,
                "n": len(values),
                "mean_km": values.mean(),
                "sd_km": values.std(ddof=1),
                "q1_km": values.quantile(0.25),
                "median_km": values.median(),
                "q3_km": values.quantile(0.75),
                "ci_lower_km": lower,
                "ci_upper_km": upper,
            }
        )

    return pd.DataFrame(rows)


def difference_interval(extra, regular):
    difference = extra.mean() - regular.mean()
    variance_extra = extra.var(ddof=1) / len(extra)
    variance_regular = regular.var(ddof=1) / len(regular)
    standard_error = np.sqrt(variance_extra + variance_regular)

    degrees_freedom = (variance_extra + variance_regular) ** 2 / (
        variance_extra**2 / (len(extra) - 1)
        + variance_regular**2 / (len(regular) - 1)
    )

    critical_value = stats.t.ppf(0.975, degrees_freedom)
    lower = difference - critical_value * standard_error
    upper = difference + critical_value * standard_error
    return difference, lower, upper


def make_violin_plot(sample):
    sns.set_theme(style="whitegrid")
    colours = {"90-minute": "#2867B2", "Extra time": "#F29124"}

    plt.figure(figsize=(9, 6))
    sns.violinplot(
        data=sample,
        x="match_condition",
        y="total_distance_km",
        order=GROUPS,
        hue="match_condition",
        palette=colours,
        inner="quart",
        cut=0,
        legend=False,
    )
    sns.stripplot(
        data=sample,
        x="match_condition",
        y="total_distance_km",
        order=GROUPS,
        color="black",
        alpha=0.5,
        jitter=0.12,
    )

    plt.title("Distance covered by starting midfielders")
    plt.xlabel("Match condition")
    plt.ylabel("Total distance covered (km)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "violin_plot.png", dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_FILE)
    data["player_id"] = data["player_id"].astype(str)
    data["total_distance_km"] = pd.to_numeric(
        data["total_distance_km"],
        errors="coerce",
    )

    data = data[
        (data["started"] == True)
        & (data["position_code"] == 2)
        & (data["estimated_minutes_played"] >= 80)
        & (data["total_distance_km"] > 0)
    ].copy()

    sample = choose_sample(data)
    summary = make_summary(sample)

    regular = sample.loc[
        sample["match_condition"] == "90-minute",
        "total_distance_km",
    ].to_numpy(dtype=float)

    extra = sample.loc[
        sample["match_condition"] == "Extra time",
        "total_distance_km",
    ].to_numpy(dtype=float)

    test = stats.ttest_ind(extra, regular, equal_var=False)
    difference, lower, upper = difference_interval(extra, regular)

    if test.pvalue < ALPHA:
        decision = "Reject H0"
    else:
        decision = "Fail to reject H0"

    sample.to_csv(OUTPUT_DIR / "sampled_50_performances.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    make_violin_plot(sample)

    print("\nSummary statistics")
    print(summary.round(3).to_string(index=False))
    print("\nWelch independent two-sample t-test")
    print(f"t-statistic: {test.statistic:.3f}")
    print(f"p-value: {test.pvalue:.4f}")
    print(f"Mean difference: {difference:.3f} km")
    print(f"95% CI for difference: [{lower:.3f}, {upper:.3f}] km")
    print(f"Decision at alpha = {ALPHA}: {decision}")


if __name__ == "__main__":
    main()