from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
figdir = root / "paper" / "figures"
figdir.mkdir(parents=True, exist_ok=True)

def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

# Exact recursion effects; each curve is computed from a finite-state model.
exact = read_rows(root / "results" / "exact.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.1))
for regime, marker in [("positive", "o"), ("zero", "s"), ("negative", "^")]:
    selected = [r for r in exact if r["regime"] == regime and int(r["B"]) == 4]
    ax.plot([int(r["C"]) for r in selected],
            [float(r["gamma_exact"]) for r in selected],
            marker=marker, label=f"{regime.capitalize()} inheritance effect")
ax.set_xscale("log", base=2)
ax.set_xlabel("Development proposals, C")
ax.set_ylabel("Exact recursion dividend")
ax.legend(fontsize=8)
save_figure(fig, "exact_dividend")

# Sharp structural bounds versus a structure-agnostic coupling bound.
sensitivity = read_rows(root / "results" / "sensitivity.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for location, style in [("rare", "-"), ("global", "--")]:
    selected = [r for r in sensitivity if r["location"] == location]
    x = [float(r["epsilon"]) for r in selected]
    ax.plot(x, [float(r["lower"]) for r in selected],
            linestyle=style, label=f"{location.capitalize()} uncertainty: lower")
    ax.plot(x, [float(r["upper"]) for r in selected],
            linestyle=style, label=f"{location.capitalize()} uncertainty: upper")
selected = [r for r in sensitivity if r["location"] == "rare"]
ax.plot([float(r["epsilon"]) for r in selected],
        [float(r["uniform_lower"]) for r in selected],
        linestyle=":", label="Uniform coupling: lower")
ax.plot([float(r["epsilon"]) for r in selected],
        [float(r["uniform_upper"]) for r in selected],
        linestyle=":", label="Uniform coupling: upper")
ax.set_xlabel("Maximum per-step total-variation radius")
ax.set_ylabel("Identified contrast endpoint")
ax.legend(fontsize=7, ncol=2)
save_figure(fig, "sensitivity")

# Independent confirmation: simultaneous, finite-sample intervals.
certificates = read_rows(root / "results" / "confirmation" / "certificates.csv")
selected = [r for r in certificates if r["contrast"] == "recursion_dividend"]
means = np.array([float(r["mean"]) for r in selected]) * 100
lower = np.array([float(r["eb_lower"]) for r in selected]) * 100
upper = np.array([float(r["eb_upper"]) for r in selected]) * 100
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.errorbar(means, np.arange(3), xerr=np.vstack([means - lower, upper - means]),
            fmt="o", capsize=5)
ax.axvline(-1, linestyle="--", label="Prespecified equivalence limits")
ax.axvline(1, linestyle="--")
ax.axvline(0, linestyle=":")
ax.set_yticks(np.arange(3), [r["family"].capitalize() for r in selected])
ax.set_xlabel("Recursion dividend, percentage points")
ax.set_xlim(-1.3, 1.3)
ax.legend(fontsize=8)
save_figure(fig, "confirmation")

lineage = read_rows(root / "results" / "lineage.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.1))
for column, label, marker in [
    ("naive_hoeffding_coverage", "Incorrectly pooled descendants", "o"),
    ("cluster_hoeffding_coverage", "Independent development-level bound", "s"),
    ("cluster_t_coverage", "Development-level t interval (approximate)", "^"),
]:
    ax.plot([int(r["m"]) for r in lineage],
            [float(r[column]) for r in lineage], marker=marker, label=label)
ax.axhline(.95, linestyle=":", label="Nominal coverage")
ax.set_xscale("log", base=4)
ax.set_xlabel("Audited descendants per development run")
ax.set_ylabel("Coverage across 10,000 repetitions")
ax.set_ylim(0, 1.06)
ax.legend(fontsize=7)
save_figure(fig, "lineage_coverage")

fig, ax = plt.subplots(figsize=(6.4, 3.1))
ax.plot([int(r["m"]) for r in lineage], [float(r["variance_exact"]) for r in lineage],
        marker="o", label="Exact nested variance")
ax.plot([int(r["m"]) for r in lineage], [float(r["variance_empirical"]) for r in lineage],
        marker="x", linestyle="--", label="Monte Carlo variance")
ax.axhline(.16 / 16, linestyle=":", label="Between-development variance floor")
ax.set_xscale("log", base=4)
ax.set_xlabel("Audited descendants per development run")
ax.set_ylabel("Variance of the population-effect estimate")
ax.legend(fontsize=8)
save_figure(fig, "lineage_variance")

print("Saved five paper figures from executed experiment outputs.")