"""Phase 3 - Differential privacy engine (Opacus wrapper).

Why DP is needed at all: weight updates are much smaller than raw data, but a
determined adversary can still try to reconstruct training examples from them
(gradient inversion attacks). DP-SGD defends against this by:

  1. clipping each sample's gradient to L2 norm `max_grad_norm` (bounds a
     single record's influence), then
  2. adding Gaussian noise with std = `noise_multiplier * max_grad_norm`
     before the update is applied.

The noise is what makes reconstruction infeasible; `get_privacy_spent` reports
epsilon (ε), the formal budget describing *how much* information leaks.
Lower epsilon = stronger privacy. ε grows as training proceeds, because every
round spends more of the budget - that accumulation is why the dashboard plots
epsilon against rounds.

NOTE: training must stay on CPU - Opacus per-sample gradient hooks do not
support the Apple MPS backend.
"""

from opacus import PrivacyEngine

# ---- DP-SGD hyper-parameters (identical on every node) ----
NOISE_MULTIPLIER = 1.0   # noise std = 1.0 x clip norm
MAX_GRAD_NORM = 1.0      # per-sample gradient clipping bound (C)
DELTA = 1e-5             # delta: probability DP guarantee fails (~1/100,000)


def make_private(model, optimizer, data_loader, noise_multiplier=NOISE_MULTIPLIER,
                 max_grad_norm=MAX_GRAD_NORM, privacy_engine=None):
    """Wrap model/optimizer/data_loader for DP-SGD.

    Args:
        model: the plain FraudNet (must be on CPU).
        optimizer: the plain optimizer (e.g. Adam) attached to that model.
        data_loader: local training DataLoader.
        noise_multiplier: Gaussian noise std as a multiple of the clip norm.
        max_grad_norm: per-sample gradient clipping bound C.
        privacy_engine: pass an existing engine to keep its accountant alive
            across federated rounds - epsilon then accumulates for the whole
            run instead of restarting every round. None creates a fresh one.

    Returns:
        (privacy_engine, private_model, private_optimizer, private_loader)
    """
    if privacy_engine is None:
        privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        # Poisson sampling is the sampling scheme the DP accounting formulas
        # assume; Opacus enables it by default and we keep it explicit.
        poisson_sampling=True,
        # Our loss is a mean over the batch, so Opacus rescales backprops to
        # recover true per-sample gradients before clipping/noising.
        loss_reduction="mean",
        grad_sample_mode="hooks",
    )
    return privacy_engine, private_model, private_optimizer, private_loader


def unwrap_private(private_model, private_optimizer) -> None:
    """Detach Opacus after a local training round.

    The private model shares the plain model's tensors, so we only have to
    remove the hooks; the caller's original model/optimizer references stay
    valid and the next round can wrap them again with the SAME privacy engine
    (which is what makes epsilon cumulative over rounds).
    """
    private_optimizer.zero_grad()   # frees the per-sample gradient buffers
    private_model.remove_hooks()


def get_privacy_spent(privacy_engine, delta=DELTA) -> float:
    """Current cumulative epsilon (ε) for this engine at the given delta."""
    return float(privacy_engine.get_epsilon(delta))
