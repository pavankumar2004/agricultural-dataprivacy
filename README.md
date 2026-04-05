# Agricultural Data Privacy: Emerging Platforms & Strategies

**Implementation of privacy-preserving techniques from:**
> Gavai, A. K., et al. (2025). *Agricultural data privacy: Emerging platforms & strategies.* Food and Humanity 4, 100542.

## Overview

This project implements the four core privacy-preserving techniques identified in the paper for protecting sensitive agricultural data while still enabling useful analytics and collaborative AI.

| Module | Technique | File | Use Case |
|--------|-----------|------|----------|
| 1 | **Differential Privacy** | `differential_privacy.py` | Adding calibrated noise to statistical queries so individual farm records can't be identified |
| 2 | **Homomorphic Encryption** | `homomorphic_encryption.py` | Computing on encrypted data in the cloud without ever decrypting it |
| 3 | **Secure Multi-Party Computation** | `secure_mpc.py` | Multiple farm cooperatives jointly computing aggregates without revealing their individual data |
| 4 | **Federated Learning** | `federated_learning.py` | Training ML models collaboratively across farm regions without sharing raw data |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all four modules
python main.py

# Or run individual modules
python differential_privacy.py
python homomorphic_encryption.py
python secure_mpc.py
python federated_learning.py
```

## Synthetic Dataset

`dataset.py` generates 500 realistic farm records with:
- **Farm metadata**: ID, region (5 regions), crop type (4 types)
- **Farm features**: area, soil moisture, rainfall, temperature, fertilizer usage
- **Outcomes**: crop yield (tons), revenue (USD)

## Module Details

### 1. Differential Privacy (ε-DP)

**Key formula:**
```
Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S]
```

- Implements the **Laplace mechanism**: `noise ~ Laplace(0, Δf/ε)`
- Demonstrates privacy–utility tradeoff across ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- Supports DP-protected `count`, `sum`, and `mean` queries
- Generates visualization plots

### 2. Homomorphic Encryption (CKKS)

**Key property:**
```
D[E(x) ⊕ E(y)] = x + y    (additive homomorphism)
D[E(x) ⊗ E(y)] = x × y    (multiplicative homomorphism)
```

- Uses TenSEAL library with CKKS scheme (approximate arithmetic on encrypted reals)
- Demonstrates encrypted sum, mean, and weighted average
- Falls back to simulation if TenSEAL is not installed

### 3. Secure Multi-Party Computation (Additive Secret Sharing)

**Protocol:**
1. Each party splits its secret value into *n* random shares that sum to the original
2. Parties exchange partial sums (never raw shares)
3. Final aggregate is reconstructed by summing partial sums

- Simulates 5 regional cooperatives jointly computing total/average yield
- Demonstrates that no individual party can learn another's private data

### 4. Federated Learning (FedAvg)

**Aggregation rule:**
```
W_global = (1/n) × Σ W_local_i
```

- Splits dataset by region to simulate 5 farm silos
- Each client trains a local regression model, shares only weights
- Server aggregates via Federated Averaging
- Compares with centralized (privacy-violating) baseline
- Generates convergence plots

## Output

Running `python main.py` will:
1. Generate and display the synthetic dataset
2. Run all four modules with detailed explanations
3. Save visualization plots:
   - `dp_privacy_utility_tradeoff.png` — DP tradeoff chart
   - `fl_convergence.png` — FL convergence comparison

## References

- Gavai, A. K., et al. (2025). Agricultural data privacy: Emerging platforms & strategies. *Food and Humanity*, 4, 100542.
- Dwork, C. (2006). Differential Privacy. *ICALP*.
- Gentry, C. (2009). Fully Homomorphic Encryption.
- McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data.
