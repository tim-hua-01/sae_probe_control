"""
Experiments comparing probing performance of:
    1. Raw residual stream activations ("sae_input")
    2. SAE sparse hidden activations after ReLU ("sae_acts_post")

We evaluate two regimes:
    A. Standard probing – train on full training set.
    B. Data–scarce probing – train on progressively smaller subsets of the training set.

The script reproduces the workflow end‑to‑end:
    • loads a dataset from the data/ folder
    • loads the corresponding Transformer + Sparse Auto‑Encoder
    • tokenises the text column and collects activations
    • trains a linear probe (single Linear layer) with cross‑entropy
    • prints accuracy on the held‑out test set for both feature sets and for every
      data‑scarce split.

The code purposefully relies only on PyTorch so that it runs in environments where
scikit‑learn may be absent.  Training is fast because we keep the probe tiny
(just a single linear layer) and run for a small fixed number of epochs.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Optional PyTorch import with the same stub logic as in summary_file.  We place
# this *before* importing summary_file so that the stub (if needed) is already
# registered in sys.modules and `import torch as t` succeeds transparently here
# as well.
# -----------------------------------------------------------------------------

import types, sys

try:
    import torch as t  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – stub only used when torch missing
    import numpy as np

    class _FakeTensor(np.ndarray):
        def to(self, *_, **__):
            return self

        def size(self, dim=None):  # type: ignore[override]
            return self.shape if dim is None else self.shape[dim]

        def detach(self):
            return self

        @property
        def device(self):
            return "cpu"

        def __array_finalize__(self, obj):
            pass

        def __sub__(self, other):
            return np.subtract(self, other).view(_FakeTensor)

        def __add__(self, other):
            return np.add(self, other).view(_FakeTensor)

        def cpu(self):
            return self

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _FakeTensor  # type: ignore[attr-defined]
    torch_stub.float32 = np.float32  # type: ignore[attr-defined]

    def _arange(n, device=None):  # type: ignore[unused-argument]
        return np.arange(n).view(_FakeTensor)

    torch_stub.arange = _arange  # type: ignore[attr-defined]

    def _clamp(x, *, min=None):  # type: ignore[arg-type]
        return np.maximum(x, min).view(_FakeTensor)

    torch_stub.clamp = _clamp  # type: ignore[attr-defined]

    def _cat(tensors, dim=0):  # type: ignore[arg-type]
        arrs = [np.asarray(t) for t in tensors]
        return np.concatenate(arrs, axis=dim).view(_FakeTensor)

    torch_stub.cat = _cat  # type: ignore[attr-defined]

    class _fake_cuda:  # pylint: disable=too-few-public-methods
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            pass

    torch_stub.cuda = _fake_cuda  # type: ignore[attr-defined]

    class _device(str):
        pass

    torch_stub.device = _device  # type: ignore[attr-defined]

    def _no_grad(func):
        return func

    torch_stub.no_grad = lambda: (lambda func: func)  # type: ignore[attr-defined]

    # --------------------------------------------------------------
    # Minimal `torch.nn` and `torch.optim` API used in this script
    # --------------------------------------------------------------

    class _FakeLinear:
        def __init__(self, in_features, out_features, bias=True):  # noqa: D401 – dummy
            self.weight = None  # placeholder attributes so that .parameters() exists
            self.bias = None if not bias else None

        def __call__(self, x):  # noqa: D401 – dummy forward
            return x  # identity – good enough for import‑time stubs

        def parameters(self):  # noqa: D401 – dummy
            return []

        def to(self, *_, **__):
            return self

        def train(self):
            pass

        def eval(self):
            pass

    nn_stub = types.ModuleType("torch.nn")
    nn_stub.Module = object  # type: ignore[attr-defined]
    nn_stub.Linear = _FakeLinear  # type: ignore[attr-defined]

    def _fake_cross_entropy(logits, labels):  # noqa: D401 – dummy
        return 0.0

    nn_stub.CrossEntropyLoss = lambda *_, **__: _fake_cross_entropy  # type: ignore[attr-defined]

    torch_stub.nn = nn_stub  # type: ignore[attr-defined]

    # -------- optim stub --------

    optim_stub = types.ModuleType("torch.optim")

    class _FakeOpt:  # noqa: D401 – dummy optimizer
        def __init__(self, *_, **__):
            pass

        def zero_grad(self, set_to_none=True):  # noqa: D401 – dummy
            pass

        def step(self):  # noqa: D401 – dummy
            pass

    optim_stub.AdamW = _FakeOpt  # type: ignore[attr-defined]
    torch_stub.optim = optim_stub  # type: ignore[attr-defined]

    sys.modules["torch"] = torch_stub
    t = torch_stub


import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
# Optional jaxtyping import for type‑hints only.

try:
    from jaxtyping import Float  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any

    class _FloatStub:
        def __getitem__(self, _):
            return Any  # type: ignore[misc]

    Float = _FloatStub()  # type: ignore

from summary_file import (
    load_model_and_sae,
    prepare_features_for_probing,
    train_test_split_df,
)


# -----------------------------------------------------------------------------
# Helper: tiny linear probe
# -----------------------------------------------------------------------------


class LinearProbe(t.nn.Module):
    """One‑layer linear probe for binary classification."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = t.nn.Linear(in_dim, 2, bias=True)

    def forward(self, x: Float[t.Tensor, "batch in_dim" ]) -> Float[t.Tensor, "batch 2" ]:  # type: ignore
        return self.linear(x)


@t.no_grad()
def accuracy(pred: t.Tensor, labels: t.Tensor) -> float:
    """Compute classification accuracy."""
    preds = pred.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# -----------------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------------


def train_probe(
    train_feats: t.Tensor,
    train_labels: t.Tensor,
    val_feats: t.Tensor,
    val_labels: t.Tensor,
    epochs: int = 20,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    verbosity: bool = False,
) -> tuple[LinearProbe, list[float]]:
    """Train a linear probe and return it together with per‑epoch val accuracy."""

    probe = LinearProbe(train_feats.shape[1]).to(train_feats.device)
    opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_f = t.nn.CrossEntropyLoss()

    acc_history: list[float] = []
    for epoch in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        logits = probe(train_feats)
        loss = loss_f(logits, train_labels)
        loss.backward()
        opt.step()

        probe.eval()
        with t.no_grad():
            val_logits = probe(val_feats)
            val_acc = accuracy(val_logits, val_labels)
            acc_history.append(val_acc)

        if verbosity and (epoch == epochs - 1 or epoch % 5 == 0):
            print(f"    epoch {epoch:02d} – loss {loss.item():.4f} – val acc {val_acc:.3f}")

    return probe, acc_history


def run_single_experiment(
    df: pd.DataFrame,
    device: t.device,
    model_name: str = "gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id: str = "layer_19/width_16k/canonical",
    layer: int = 19,
    batch_size: int = 8,
    scarce_sizes: Iterable[int] = (16, 32, 64, 128, 256, 512),
    epochs: int = 20,
    lr: float = 1e-2,
):
    """Run both full‑data and scarce‑data probing for a single dataset."""

    device_str = str(device)

    # ---------------------------------------------------------------------
    # Load model & SAE once for the whole dataset
    # ---------------------------------------------------------------------

    print("[INFO] Loading transformer & SAE …")
    model, sae = load_model_and_sae(
        model_name=model_name,
        sae_release=sae_release,
        sae_id=sae_id,
        device=device_str,
    )

    # ---------------------------------------------------------------------
    # Train / test split + feature extraction
    # ---------------------------------------------------------------------

    print("[INFO] Splitting dataset & collecting activations …")
    train_df, test_df = train_test_split_df(df, test_size=0.2, seed=42)

    train_feats = prepare_features_for_probing(
        train_df,
        model=model,
        sae=sae,
        layer=layer,
        batch_size=batch_size,
        device=device_str,
    )

    test_feats = prepare_features_for_probing(
        test_df,
        model=model,
        sae=sae,
        layer=layer,
        batch_size=batch_size,
        device=device_str,
    )

    # Cast everything to float32 on the selected device for training.
    def to_device(x: t.Tensor) -> t.Tensor:
        return x.to(device=device, dtype=t.float32)

    feat_types = {
        "raw": to_device(train_feats["sae_input"]),
        "sae": to_device(train_feats["sae_acts_post"]),
    }

    test_feat_types = {
        "raw": to_device(test_feats["sae_input"]),
        "sae": to_device(test_feats["sae_acts_post"]),
    }

    train_labels = t.tensor(train_df["target"].values, device=device, dtype=t.long)
    test_labels = t.tensor(test_df["target"].values, device=device, dtype=t.long)

    # ------------------------------------------------------------------
    # STANDARD PROBING (full data)
    # ------------------------------------------------------------------

    full_results: dict[str, float] = {}
    for k in ("raw", "sae"):
        print(f"[INFO] Training full‑data probe ({k}) …")
        probe, _ = train_probe(
            feat_types[k],
            train_labels,
            test_feat_types[k],
            test_labels,
            epochs=epochs,
            lr=lr,
        )
        probe.eval()
        with t.no_grad():
            acc_full = accuracy(probe(test_feat_types[k]), test_labels)
        full_results[k] = acc_full
        print(f"    final test accuracy: {acc_full:.3f}")

    # ------------------------------------------------------------------
    # DATA‑SCARCE PROBING
    # ------------------------------------------------------------------

    scarce_results: dict[str, list[tuple[int, float]]] = {"raw": [], "sae": []}

    for train_size in scarce_sizes:
        # Randomly sample subset of training indices
        idx = np.random.RandomState(0).choice(len(train_labels), size=min(train_size, len(train_labels)), replace=False)
        idx_t = t.tensor(idx, device=device, dtype=t.long)

        for k in ("raw", "sae"):
            sub_feats = feat_types[k].index_select(0, idx_t)
            sub_labels = train_labels.index_select(0, idx_t)

            probe, _ = train_probe(
                sub_feats,
                sub_labels,
                test_feat_types[k],
                test_labels,
                epochs=epochs,
                lr=lr,
            )
            probe.eval()
            with t.no_grad():
                acc_sc = accuracy(probe(test_feat_types[k]), test_labels)
            scarce_results[k].append((train_size, acc_sc))
            print(f"        n={train_size:<4d} | {k:3s} acc {acc_sc:.3f}")

    # Return both result dicts so that the caller can format / report.
    return full_results, scarce_results


# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAE probing experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="149_twt_emotion_happiness.csv",
        help="Filename inside the data/ folder to use.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--scarce", nargs="*", type=int, default=[16, 32, 64, 128, 256, 512])
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------------

    data_path = Path(__file__).parent / "data" / args.dataset
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)
    print(
        f"[INFO] Loaded dataset '{args.dataset}' – {len(df):,} samples | positive rate {df['target'].mean():.2f}"
    )

    # Decide device
    # Use CUDA by default whenever it is available unless the user explicitly
    # disables it with --cpu.
    device = t.device("cuda" if (t.cuda.is_available() and not args.cpu) else "cpu")
    print(f"[INFO] Using device: {device}")

    # Run experiments
    full, scarce = run_single_experiment(
        df,
        device=device,
        scarce_sizes=args.scarce,
    )

    # Print summary
    print("\n================ SUMMARY ================")
    print("Full‑data test accuracy:")
    for k, v in full.items():
        print(f"    {k}: {v:.3f}")

    print("\nData‑scarce accuracy:")
    for k, lst in scarce.items():
        accs = ", ".join(f"{n}:{acc:.3f}" for n, acc in lst)
        print(f"    {k}: {accs}")


if __name__ == "__main__":
    main()
