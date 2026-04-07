import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import generate_dataset



def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float,
                      rng: np.random.Generator = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale)
    return true_value + noise


def dp_count(data: pd.Series, epsilon: float, rng=None) -> float:
    return laplace_mechanism(len(data), sensitivity=1.0, epsilon=epsilon, rng=rng)


def dp_sum(data: pd.Series, epsilon: float, value_range: tuple, rng=None) -> float:
    sensitivity = value_range[1] - value_range[0]
    return laplace_mechanism(data.sum(), sensitivity=sensitivity, epsilon=epsilon, rng=rng)


def dp_mean(data: pd.Series, epsilon: float, value_range: tuple, rng=None) -> float:
    noisy_sum = dp_sum(data, epsilon / 2, value_range, rng=rng)
    noisy_count = dp_count(data, epsilon / 2, rng=rng)
    if noisy_count <= 0:
        noisy_count = 1
    return noisy_sum / noisy_count



def demonstrate_privacy_utility_tradeoff(df: pd.DataFrame):
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    num_trials = 50
    rng = np.random.default_rng(123)

    yield_data = df["yield_tons"]
    true_mean = yield_data.mean()
    value_range = (0, yield_data.max())

    print("=" * 70)
    print("  MODULE 1: DIFFERENTIAL PRIVACY - Laplace Mechanism")
    print("=" * 70)
    print(f"\n  Query: Average crop yield across {len(df)} farms")
    print(f"  True answer: {true_mean:.2f} tons\n")
    print(f"  {'eps':>6}  {'Noisy Mean':>12}  {'Abs Error':>10}  {'Rel Error %':>12}")
    print("  " + "-" * 46)

    results = {}
    for eps in epsilons:
        noisy_values = [dp_mean(yield_data, eps, value_range, rng=rng) for _ in range(num_trials)]
        avg_noisy = np.mean(noisy_values)
        avg_error = np.mean([abs(v - true_mean) for v in noisy_values])
        rel_error = (avg_error / true_mean) * 100
        results[eps] = {"noisy_values": noisy_values, "avg_error": avg_error}
        print(f"  {eps:>6.1f}  {avg_noisy:>12.2f}  {avg_error:>10.2f}  {rel_error:>11.1f}%")

    print("\n  Observation: Smaller eps gives stronger privacy, but higher error.")
    print("  As eps increases, noisy answers converge to the true answer.\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    errors = [results[e]["avg_error"] for e in epsilons]
    axes[0].bar([str(e) for e in epsilons], errors, color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"])
    axes[0].set_xlabel("Privacy Budget (eps)", fontsize=12)
    axes[0].set_ylabel("Average Absolute Error (tons)", fontsize=12)
    axes[0].set_title("Privacy-Utility Tradeoff", fontsize=14, fontweight="bold")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    axes[1].hist(results[0.1]["noisy_values"], bins=15, alpha=0.6, label="eps = 0.1 (strong privacy)", color="#e74c3c")
    axes[1].hist(results[5.0]["noisy_values"], bins=15, alpha=0.6, label="eps = 5.0 (weak privacy)", color="#3498db")
    axes[1].axvline(x=true_mean, color="black", linewidth=2, linestyle="--", label=f"True mean = {true_mean:.1f}")
    axes[1].set_xlabel("Noisy Mean Yield (tons)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Distribution of DP Answers", fontsize=14, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = "dp_privacy_utility_tradeoff.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")

    return results


def demonstrate_per_region_query(df: pd.DataFrame):
    """Show DP-protected per-region average yield."""
    epsilon = 1.0
    rng = np.random.default_rng(456)

    print("\n  Per-region average yield (eps = 1.0):")
    print(f"  {'Region':>10}  {'True Mean':>10}  {'DP Mean':>10}  {'Error':>8}")
    print("  " + "-" * 44)

    for region in sorted(df["region"].unique()):
        region_yield = df.loc[df["region"] == region, "yield_tons"]
        true_mean = region_yield.mean()
        value_range = (0, region_yield.max())
        noisy_mean = dp_mean(region_yield, epsilon, value_range, rng=rng)
        error = abs(noisy_mean - true_mean)
        print(f"  {region:>10}  {true_mean:>10.2f}  {noisy_mean:>10.2f}  {error:>8.2f}")

    print()


def run():
    """Run the full Differential Privacy demonstration."""
    df = generate_dataset()
    demonstrate_privacy_utility_tradeoff(df)
    demonstrate_per_region_query(df)


if __name__ == "__main__":
    run()
