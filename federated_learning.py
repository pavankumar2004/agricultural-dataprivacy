"""
federated_learning.py — Federated Averaging (FedAvg) for Crop Yield Prediction
===============================================================================
Demonstrates collaborative machine learning across multiple farm regions
WITHOUT sharing raw data — only model parameters (weights) are exchanged.

Theory (from the paper):
    Federated Learning trains models locally on farm-level devices and only
    shares model parameters (gradients), not raw data.

    Federated Averaging:
        W_global = (1/n) × Σ W_local_i

    Each round:
        1. Server broadcasts global model to all clients.
        2. Each client trains locally on its own data for a few epochs.
        3. Clients send updated weights back to the server.
        4. Server averages the weights to produce a new global model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from dataset import generate_dataset, get_region_data


# ── Federated Learning Components ──────────────────────────────

class FarmClient:
    """
    Represents a single farm region (client) in the federated system.
    Each client has its own private data and trains a local model.
    """

    def __init__(self, name: str, data: pd.DataFrame, feature_cols: list,
                 target_col: str):
        self.name = name
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Prepare local data
        self.X = data[feature_cols].values
        self.y = data[target_col].values

        # Local model (linear regression via SGD)
        self.model = SGDRegressor(
            max_iter=1, tol=None, warm_start=True,
            learning_rate="constant", eta0=0.001, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Initialize model with a single pass
        self.model.partial_fit(self.X_scaled, self.y)

    def get_weights(self) -> tuple:
        """Return current model weights (coef, intercept)."""
        return self.model.coef_.copy(), self.model.intercept_.copy()

    def set_weights(self, coef: np.ndarray, intercept: np.ndarray):
        """Set model weights from the global model."""
        self.model.coef_ = coef.copy()
        self.model.intercept_ = intercept.copy()

    def train_local(self, epochs: int = 5):
        """Train the model locally for a few epochs on private data."""
        for _ in range(epochs):
            self.model.partial_fit(self.X_scaled, self.y)

    def evaluate(self) -> dict:
        """Evaluate model on local data."""
        y_pred = self.model.predict(self.X_scaled)
        return {
            "mse": mean_squared_error(self.y, y_pred),
            "r2": r2_score(self.y, y_pred),
        }


class FederatedServer:
    """
    Central server that coordinates federated learning.
    The server NEVER sees raw data — only model weights.
    """

    def __init__(self, clients: list):
        self.clients = clients
        self.global_coef = None
        self.global_intercept = None
        self.history = []

    def federated_average(self):
        """
        FedAvg: Average all client model weights.
            W_global = (1/n) × Σ W_local_i
        """
        all_coefs = []
        all_intercepts = []

        for client in self.clients:
            coef, intercept = client.get_weights()
            all_coefs.append(coef)
            all_intercepts.append(intercept)

        self.global_coef = np.mean(all_coefs, axis=0)
        self.global_intercept = np.mean(all_intercepts, axis=0)

    def broadcast_global_model(self):
        """Send the global model to all clients."""
        for client in self.clients:
            client.set_weights(self.global_coef, self.global_intercept)

    def train_round(self, local_epochs: int = 5):
        """
        Execute one round of federated learning:
        1. Broadcast global model
        2. Each client trains locally
        3. Aggregate via FedAvg
        """
        # Step 1: Broadcast
        if self.global_coef is not None:
            self.broadcast_global_model()

        # Step 2: Local training
        for client in self.clients:
            client.train_local(epochs=local_epochs)

        # Step 3: Aggregate
        self.federated_average()

        # Evaluate
        round_metrics = {}
        for client in self.clients:
            client.set_weights(self.global_coef, self.global_intercept)
            metrics = client.evaluate()
            round_metrics[client.name] = metrics

        avg_mse = np.mean([m["mse"] for m in round_metrics.values()])
        avg_r2 = np.mean([m["r2"] for m in round_metrics.values()])
        self.history.append({"avg_mse": avg_mse, "avg_r2": avg_r2,
                            "per_client": round_metrics})

        return avg_mse, avg_r2


# ── Centralized Baseline ──────────────────────────────────────

def train_centralized(df: pd.DataFrame, feature_cols: list,
                      target_col: str) -> dict:
    """Train a centralized model on ALL data (privacy-violating baseline)."""
    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(max_iter=200, tol=1e-4, learning_rate="constant",
                         eta0=0.001, random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    return {
        "mse": mean_squared_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }


# ── Demonstration ──────────────────────────────────────────────

def run():
    """Run the full Federated Learning demonstration."""
    print("=" * 70)
    print("  MODULE 4: FEDERATED LEARNING  —  FedAvg")
    print("=" * 70)

    df = generate_dataset()
    feature_cols = ["area_hectares", "soil_moisture", "rainfall_mm",
                    "temperature_c", "fertilizer_kg"]
    target_col = "yield_tons"

    # Split data by region (simulating 5 independent farm silos)
    region_data = get_region_data(df)

    print(f"\n  Task: Predict crop yield from farm features")
    print(f"  Features: {feature_cols}")
    print(f"  Target: {target_col}")
    print(f"\n  Simulating {len(region_data)} farm regions as FL clients:")
    for region, data in region_data.items():
        print(f"    📍 {region}: {len(data)} farms")

    # ── Create FL system ───────────────────────────────────────
    clients = []
    for region, data in region_data.items():
        client = FarmClient(region, data, feature_cols, target_col)
        clients.append(client)

    server = FederatedServer(clients)

    # ── Run federated training ─────────────────────────────────
    num_rounds = 20
    local_epochs = 5

    print(f"\n  Training: {num_rounds} rounds × {local_epochs} local epochs per round")
    print(f"  {'Round':>7}  {'Avg MSE':>12}  {'Avg R²':>10}")
    print("  " + "-" * 35)

    for round_num in range(1, num_rounds + 1):
        avg_mse, avg_r2 = server.train_round(local_epochs=local_epochs)
        if round_num % 5 == 0 or round_num == 1:
            print(f"  {round_num:>7}  {avg_mse:>12.2f}  {avg_r2:>10.4f}")

    # ── Centralized baseline ───────────────────────────────────
    print("\n  ── Comparison with centralized training ──")
    centralized = train_centralized(df, feature_cols, target_col)

    final_fed = server.history[-1]
    print(f"\n  ┌────────────────────────────────────────────────┐")
    print(f"  │  Model              MSE          R²             │")
    print(f"  │  ─────────────  ──────────  ──────────          │")
    print(f"  │  Federated      {final_fed['avg_mse']:>10.2f}  {final_fed['avg_r2']:>10.4f}          │")
    print(f"  │  Centralized    {centralized['mse']:>10.2f}  {centralized['r2']:>10.4f}          │")
    print(f"  └────────────────────────────────────────────────┘")
    print(f"\n  🔒 Federated model achieves comparable accuracy WITHOUT")
    print(f"     any farm sharing its raw data with other farms!")

    # ── Per-region breakdown ───────────────────────────────────
    print(f"\n  Per-region performance (final federated model):")
    print(f"  {'Region':>10}  {'MSE':>10}  {'R²':>8}")
    print("  " + "-" * 32)
    for client_name, metrics in final_fed["per_client"].items():
        print(f"  {client_name:>10}  {metrics['mse']:>10.2f}  {metrics['r2']:>8.4f}")

    # ── Convergence plot ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rounds = list(range(1, num_rounds + 1))
    mses = [h["avg_mse"] for h in server.history]
    r2s = [h["avg_r2"] for h in server.history]

    # MSE convergence
    axes[0].plot(rounds, mses, "o-", color="#e74c3c", linewidth=2, markersize=5,
                 label="Federated (FedAvg)")
    axes[0].axhline(y=centralized["mse"], color="#3498db", linewidth=2,
                    linestyle="--", label="Centralized baseline")
    axes[0].set_xlabel("Communication Round", fontsize=12)
    axes[0].set_ylabel("Mean Squared Error", fontsize=12)
    axes[0].set_title("FL Convergence — MSE", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # R² convergence
    axes[1].plot(rounds, r2s, "o-", color="#2ecc71", linewidth=2, markersize=5,
                 label="Federated (FedAvg)")
    axes[1].axhline(y=centralized["r2"], color="#3498db", linewidth=2,
                    linestyle="--", label="Centralized baseline")
    axes[1].set_xlabel("Communication Round", fontsize=12)
    axes[1].set_ylabel("R² Score", fontsize=12)
    axes[1].set_title("FL Convergence — R²", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = "fl_convergence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Plot saved → {path}\n")

    return server.history


if __name__ == "__main__":
    run()
