import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# settings
file_name = r"C:\Users\LENOVO\Documents\Python\.vscode\FIFA.xlsx"
benchmark = 35   # 2022 World Cup SOT% benchmark
alpha = 0.05     # significance level

# loading the data
data = pd.read_excel(file_name, sheet_name="Sheet1", header=2)

# renaming columns to simpler names
data = data.rename(columns={
    "Match": "match_id",
    "Team": "fixture",
    "total shots": "shots_total",
    "SOT": "shots_on_target"
})

# drop rows with no match id
data = data.dropna(subset=["match_id"])

# removing matches with 0 total shots
data = data[data["shots_total"] > 0]

# calculating SOT% for each team-match
data["sot_percent"] = (data["shots_on_target"] / data["shots_total"]) * 100
sot = data["sot_percent"]

# descriptive statistics
n = len(sot)
mean = sot.mean()
std_dev = sot.std()
std_error = std_dev / (n ** 0.5)
median = sot.median()
q1 = sot.quantile(0.25)
q3 = sot.quantile(0.75)
iqr = q3 - q1

print("Sample size:", n)
print("Mean SOT%:", round(mean, 2))
print("Standard deviation:", round(std_dev, 2))
print("Median:", round(median, 2))
print("IQR:", round(iqr, 2))

# checking for outliers
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
outliers = sot[(sot < lower_limit) | (sot > upper_limit)]
print("Outliers:", list(outliers))

# histogram of SOT%
plt.hist(sot, bins=8, color="steelblue", edgecolor="black")
plt.axvline(mean, color="red", linestyle="--", label="Sample mean")
plt.axvline(benchmark, color="black", linestyle=":", label="2022 benchmark")
plt.title("Histogram of SOT%")
plt.xlabel("SOT%")
plt.ylabel("Number of matches")
plt.legend()
plt.show()

# boxplot of SOT%
plt.boxplot(sot)
plt.axhline(benchmark, color="black", linestyle=":", label="2022 benchmark")
plt.title("Boxplot of SOT%")
plt.ylabel("SOT%")
plt.legend()
plt.show()

# 95% confidence interval for the mean
df = n - 1
t_critical = stats.t.ppf(0.975, df)
margin_of_error = t_critical * std_error
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

print("95% Confidence interval:", round(ci_lower, 2), "to", round(ci_upper, 2))

# one sample t-test: H0: mean = 35, H1: mean > 35
t_stat, p_two_tail = stats.ttest_1samp(sot, benchmark)
p_one_tail = p_two_tail / 2 if t_stat > 0 else 1 - (p_two_tail / 2)

print("t statistic:", round(t_stat, 4))
print("p value (one-tailed):", round(p_one_tail, 4))

# decision
if p_one_tail < alpha:
    print("p < alpha, reject H0")
else:
    print("p > alpha, fail to reject H0")