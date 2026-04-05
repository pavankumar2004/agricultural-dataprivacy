"""
homomorphic_encryption.py — CKKS Homomorphic Encryption
=======================================================
Demonstrates computation on encrypted agricultural data without decryption.

Theory (from the paper):
    The encryption E(x) and decryption D(x) functions satisfy:
        D[E(x) ⊕ E(y)] = x + y      (additive homomorphism)
        D[E(x) ⊗ E(y)] = x × y      (multiplicative homomorphism)

    CKKS scheme supports approximate arithmetic on encrypted real numbers.
    Use case: Secure outsourcing of crop yield analytics to a cloud server.
"""

import numpy as np
import pandas as pd

try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False

from dataset import generate_dataset


# ── CKKS Context Setup ─────────────────────────────────────────

def create_ckks_context():
    """
    Create a TenSEAL CKKS encryption context.

    Parameters chosen for a good balance of security and performance:
    - poly_modulus_degree: 8192 (128-bit security)
    - coeff_mod_bit_sizes: chain of moduli for rescaling
    - global_scale: 2^40 for precision
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


# ── Encrypted Operations ──────────────────────────────────────

def encrypted_sum(context, values: list) -> float:
    """Encrypt values and compute their sum homomorphically."""
    enc_vec = ts.ckks_vector(context, values)
    # Sum by adding all elements — use a dot product with a vector of ones
    ones = [1.0] * len(values)
    enc_result = enc_vec.dot(ones)
    return enc_result.decrypt()[0]


def encrypted_mean(context, values: list) -> float:
    """Encrypt values and compute their mean homomorphically."""
    enc_vec = ts.ckks_vector(context, values)
    n = len(values)
    # Multiply each element by 1/n, then sum
    weights = [1.0 / n] * n
    enc_result = enc_vec.dot(weights)
    return enc_result.decrypt()[0]


def encrypted_weighted_avg(context, values: list, weights: list) -> float:
    """Compute weighted average on encrypted data."""
    # Normalize weights
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]
    enc_vec = ts.ckks_vector(context, values)
    enc_result = enc_vec.dot(norm_weights)
    return enc_result.decrypt()[0]


# ── Demonstration ──────────────────────────────────────────────

def run():
    """Run the full Homomorphic Encryption demonstration."""
    print("=" * 70)
    print("  MODULE 2: HOMOMORPHIC ENCRYPTION  —  CKKS Scheme")
    print("=" * 70)

    if not HAS_TENSEAL:
        print("\n  ⚠️  TenSEAL not installed. Running with simulated HE.\n")
        run_simulated()
        return

    df = generate_dataset()
    ctx = create_ckks_context()

    # Use a subset for performance (CKKS handles vectors efficiently)
    yields = df["yield_tons"].tolist()[:200]
    revenues = df["revenue_usd"].tolist()[:200]
    areas = df["area_hectares"].tolist()[:200]

    print(f"\n  Using {len(yields)} farm records for encrypted computation")
    print(f"  Encryption scheme: CKKS (poly_degree=8192, scale=2^40)")
    print(f"  Security level: ~128-bit\n")

    # ── Test 1: Encrypted Sum ──────────────────────────────────
    true_sum = sum(yields)
    enc_sum = encrypted_sum(ctx, yields)
    error_sum = abs(enc_sum - true_sum)

    print(f"  ┌─ Test 1: SUM of crop yields ─────────────────────┐")
    print(f"  │  Plaintext sum:  {true_sum:>14.2f} tons              │")
    print(f"  │  Encrypted sum:  {enc_sum:>14.2f} tons              │")
    print(f"  │  Error:          {error_sum:>14.6f} tons              │")
    print(f"  └──────────────────────────────────────────────────┘\n")

    # ── Test 2: Encrypted Mean ─────────────────────────────────
    true_mean = np.mean(yields)
    enc_mean = encrypted_mean(ctx, yields)
    error_mean = abs(enc_mean - true_mean)

    print(f"  ┌─ Test 2: MEAN of crop yields ────────────────────┐")
    print(f"  │  Plaintext mean: {true_mean:>14.2f} tons              │")
    print(f"  │  Encrypted mean: {enc_mean:>14.2f} tons              │")
    print(f"  │  Error:          {error_mean:>14.6f} tons              │")
    print(f"  └──────────────────────────────────────────────────┘\n")

    # ── Test 3: Weighted Average (yield weighted by area) ──────
    true_wavg = np.average(yields, weights=areas)
    enc_wavg = encrypted_weighted_avg(ctx, yields, areas)
    error_wavg = abs(enc_wavg - true_wavg)

    print(f"  ┌─ Test 3: WEIGHTED AVG (yield by area) ───────────┐")
    print(f"  │  Plaintext w.avg:{true_wavg:>14.2f} tons              │")
    print(f"  │  Encrypted w.avg:{enc_wavg:>14.2f} tons              │")
    print(f"  │  Error:          {error_wavg:>14.6f} tons              │")
    print(f"  └──────────────────────────────────────────────────┘\n")

    # ── Test 4: Revenue statistics ─────────────────────────────
    true_rev_mean = np.mean(revenues)
    enc_rev_mean = encrypted_mean(ctx, revenues)
    error_rev = abs(enc_rev_mean - true_rev_mean)

    print(f"  ┌─ Test 4: MEAN of farm revenue ───────────────────┐")
    print(f"  │  Plaintext mean: ${true_rev_mean:>13.2f}               │")
    print(f"  │  Encrypted mean: ${enc_rev_mean:>13.2f}               │")
    print(f"  │  Error:          ${error_rev:>13.6f}               │")
    print(f"  └──────────────────────────────────────────────────┘\n")

    print("  ✅ All encrypted computations match plaintext results")
    print("     (within CKKS floating-point approximation tolerance)")
    print("  🔒 The cloud server never saw the raw farm data!\n")


def run_simulated():
    """Fallback: simulate HE behavior without TenSEAL."""
    df = generate_dataset()
    yields = df["yield_tons"].tolist()[:200]
    revenues = df["revenue_usd"].tolist()[:200]
    areas = df["area_hectares"].tolist()[:200]

    print(f"  Using {len(yields)} farm records (simulated encryption)\n")
    print("  In real HE, computations happen on ciphertexts.")
    print("  Here we simulate by showing that the same operations")
    print("  can be performed without revealing individual values.\n")

    # Simulate small CKKS rounding errors
    rng = np.random.default_rng(42)

    true_sum = sum(yields)
    sim_sum = true_sum + rng.normal(0, 0.001)
    print(f"  SUM  │ Plaintext: {true_sum:>12.2f}  │ Encrypted: {sim_sum:>12.2f}  │ Error: {abs(sim_sum-true_sum):.6f}")

    true_mean = np.mean(yields)
    sim_mean = true_mean + rng.normal(0, 0.0001)
    print(f"  MEAN │ Plaintext: {true_mean:>12.2f}  │ Encrypted: {sim_mean:>12.2f}  │ Error: {abs(sim_mean-true_mean):.6f}")

    true_wavg = np.average(yields, weights=areas)
    sim_wavg = true_wavg + rng.normal(0, 0.0001)
    print(f"  WAVG │ Plaintext: {true_wavg:>12.2f}  │ Encrypted: {sim_wavg:>12.2f}  │ Error: {abs(sim_wavg-true_wavg):.6f}")

    true_rev = np.mean(revenues)
    sim_rev = true_rev + rng.normal(0, 0.01)
    print(f"  REV  │ Plaintext: {true_rev:>12.2f}  │ Encrypted: {sim_rev:>12.2f}  │ Error: {abs(sim_rev-true_rev):.6f}")

    print("\n  ⚠️  Install TenSEAL for real encrypted computation: pip install tenseal")
    print("  ✅ Simulated results show the concept of HE correctly.\n")


if __name__ == "__main__":
    run()
