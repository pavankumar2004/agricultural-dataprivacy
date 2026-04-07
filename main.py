"""Run all privacy-preserving demo modules."""

from dataset import generate_dataset


def print_banner():
    """Print the project banner."""
    print()
    print("=" * 70)
    print("  AGRICULTURAL DATA PRIVACY: EMERGING PLATFORMS & STRATEGIES")
    print("  Based on: Gavai et al. (2025) - Food and Humanity 4, 100542")
    print("  Implementation of four privacy-preserving techniques")
    print("=" * 70)
    print()


def print_section_header(num: int, title: str, description: str):
    """Print a section header between modules."""
    print()
    print("-" * 70)
    print(f"  [{num}/4]  {title}")
    print(f"  {description}")
    print("-" * 70)
    print()


def run_all():
    """Execute all four privacy modules sequentially."""
    print_banner()

    print("  Generating synthetic agricultural dataset...")
    df = generate_dataset()
    print(f"  Created {len(df)} farm records across "
          f"{df['region'].nunique()} regions, "
          f"{df['crop_type'].nunique()} crop types\n")
    print(f"  Dataset Summary:")
    print(f"  {'':>4}{'Column':>18}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'':>4}{'-'*62}")
    for col in ["area_hectares", "yield_tons", "soil_moisture", "rainfall_mm",
                "temperature_c", "fertilizer_kg", "revenue_usd"]:
        stats = df[col].describe()
        print(f"  {'':>4}{col:>18}  {stats['mean']:>10.1f}  {stats['std']:>10.1f}"
              f"  {stats['min']:>10.1f}  {stats['max']:>10.1f}")

    print_section_header(1, "DIFFERENTIAL PRIVACY",
                         "Laplace mechanism for privacy-preserving statistical queries")
    import differential_privacy
    differential_privacy.run()

    print_section_header(2, "HOMOMORPHIC ENCRYPTION",
                         "CKKS scheme for computation on encrypted farm data")
    import homomorphic_encryption
    homomorphic_encryption.run()

    print_section_header(3, "SECURE MULTI-PARTY COMPUTATION",
                         "Additive secret sharing for multi-region aggregation")
    import secure_mpc
    secure_mpc.run()

    print_section_header(4, "FEDERATED LEARNING",
                         "FedAvg for collaborative crop yield prediction")
    import federated_learning
    federated_learning.run()

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("  Technique               Use Case                      Status")
    print("  ----------------------- ----------------------------- --------")
    print("  Differential Privacy    Statistical reporting         Done")
    print("  Homomorphic Encryption  Secure cloud computation      Done")
    print("  Secure MPC              Multi-party aggregation       Done")
    print("  Federated Learning      Collaborative ML training     Done")
    print()
    print("  Key Insight: These techniques can be combined in a")
    print("  modular architecture to build comprehensive privacy-")
    print("  preserving agricultural data platforms.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run_all()
