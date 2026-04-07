"""Secure multi-party computation with additive secret sharing."""

import numpy as np
import pandas as pd

from dataset import generate_dataset


def create_shares(secret: float, num_parties: int,
                  rng: np.random.Generator = None) -> list:
    """
    Split a secret value into n additive shares.

    The shares sum to the original secret:
        secret = share_1 + share_2 + ... + share_n

    Each individual share reveals nothing about the secret.
    """
    if rng is None:
        rng = np.random.default_rng()

    shares = [rng.uniform(-1e6, 1e6) for _ in range(num_parties - 1)]
    shares.append(secret - sum(shares))
    return shares


def reconstruct_secret(shares: list) -> float:
    """Reconstruct the secret by summing all shares."""
    return sum(shares)


def create_shares_for_dataset(values: list, num_parties: int,
                               rng: np.random.Generator = None) -> list:
    """
    Create secret shares for a list of values.

    Returns:
        List of lists, where shares_by_party[i] contains party i's shares
        for all values.
    """
    if rng is None:
        rng = np.random.default_rng()

    shares_by_party = [[] for _ in range(num_parties)]

    for v in values:
        shares = create_shares(v, num_parties, rng)
        for party_idx, share in enumerate(shares):
            shares_by_party[party_idx].append(share)

    return shares_by_party


def secure_sum(shares_by_party: list) -> float:
    """
    Compute the sum of secret-shared values.

    Protocol:
    1. Each party computes the sum of their own shares (locally).
    2. Parties exchange their partial sums.
    3. The final sum is the sum of all partial sums.

    No party ever sees another party's individual shares.
    """
    partial_sums = [sum(party_shares) for party_shares in shares_by_party]
    return sum(partial_sums)


def secure_mean(shares_by_party: list, count: int) -> float:
    """Compute mean from secret-shared values."""
    return secure_sum(shares_by_party) / count


def run():
    """Run the full Secure MPC demonstration."""
    print("=" * 70)
    print("  MODULE 3: SECURE MULTI-PARTY COMPUTATION - Secret Sharing")
    print("=" * 70)

    df = generate_dataset()
    rng = np.random.default_rng(789)

    num_parties = 5
    regions = sorted(df["region"].unique())

    print(f"\n  Scenario: {num_parties} regional cooperatives want to compute")
    print("  the total and average crop yield across ALL regions,")
    print("  but no cooperative wants to reveal its individual data.\n")

    region_yields = {}
    print("  Step 1 - Each region's private total yield:")
    print(f"  {'Region':>10}  {'Farms':>6}  {'Total Yield (tons)':>20}")
    print("  " + "-" * 42)
    for region in regions:
        region_data = df[df["region"] == region]
        total = region_data["yield_tons"].sum()
        region_yields[region] = total
        print(f"  {region:>10}  {len(region_data):>6}  {total:>20.2f}")

    all_values = list(region_yields.values())
    true_total = sum(all_values)
    true_mean = true_total / len(all_values)

    print(f"\n  Step 2 - Split each region's value into {num_parties} secret shares:")
    print("           (Each share looks like random noise)\n")

    shares_by_party = create_shares_for_dataset(all_values, num_parties, rng)

    for party_idx in range(num_parties):
        shares_str = [f"{s:>12.2f}" for s in shares_by_party[party_idx]]
        print(f"    Party {party_idx+1} sees: [{', '.join(shares_str)}]")

    print(f"\n  Step 3 - Verify secrecy:")
    print(f"    Party 1's share for Region '{regions[0]}': {shares_by_party[0][0]:>12.2f}")
    print(f"    True value for Region '{regions[0]}':       {all_values[0]:>12.2f}")
    print("    Party 1 cannot determine the true value from its share alone.")

    print("\n  Step 4 - Secure aggregation (each party sums locally, then combine):")

    partial_sums = []
    for party_idx in range(num_parties):
        ps = sum(shares_by_party[party_idx])
        partial_sums.append(ps)
        print(f"    Party {party_idx+1} partial sum: {ps:>14.2f}")

    reconstructed_total = sum(partial_sums)
    reconstructed_mean = reconstructed_total / len(all_values)

    total_match = "YES" if abs(reconstructed_total - true_total) < 1e-6 else "NO"
    mean_match = "YES" if abs(reconstructed_mean - true_mean) < 1e-6 else "NO"
    print("\n  Step 5 - Results:")
    print(f"  Total Yield | True: {true_total:>14.2f} tons | Reconstructed: {reconstructed_total:>14.2f} tons | Match: {total_match}")
    print(f"  Mean  Yield | True: {true_mean:>14.2f} tons | Reconstructed: {reconstructed_mean:>14.2f} tons | Match: {mean_match}\n")

    print("  Bonus: Individual farm-level secret sharing")
    sample_farm = df.iloc[0]
    secret_val = sample_farm["yield_tons"]
    shares = create_shares(secret_val, 3, rng)
    print(f"  Farm: {sample_farm['farm_id']}  |  True yield: {secret_val:.2f} tons")
    print(f"  Split into 3 shares: {[f'{s:.2f}' for s in shares]}")
    print(f"  Reconstructed: {reconstruct_secret(shares):.2f} tons")
    print("  No single party can learn the farm's real yield.\n")


if __name__ == "__main__":
    run()
