"""
train.py — Automated Integrity Verification & Alignment Training Loop (Stage 2)

This script serves a dual operational purpose within the training pipeline:
  1. Automated Architecture Verification (run_sanity_check): Executes a deterministic
     integrity validation of tensor topologies, gradient distribution flows, 
     and optimization mechanics utilizing synthetic tensors before data collection.
  2. Cross-Modal Optimization Loop (train): Implements the structured training 
     and validation pipeline to align cross-modal manifolds once the paired 
     tactile-visual dataset becomes accessible.

Execution Paradigms:
  # Structural Sanity Check (Offline deployment without real datasets):
      python -m src.training.train --mode sanity

  # Production Alignment Training (Requires pre-computed Stage 1 visual cache):
      python -m src.training.train --mode train --data_dir /path/to/dataset --tvl_ckpt /path/to/tvl.pth
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

# Absolute package imports mapped across the repository topology
from src.models.model import TactileProjector, D_VISUAL
from src.training.loss import FeatureAlignmentLoss, compute_alignment_metrics


# ─────────────────────────────────────────────────────────────
#  DATASET MANIFESTS & PIPELINES
# ─────────────────────────────────────────────────────────────

class TactileVisualDataset(Dataset):
    """
    Implements a highly optimized pipeline for ingesting synchronized multi-modal pairs:
    (Tactile Image, Cached Visual Embedding).

    Downstream Production Specifications:
      - tactile_images : RGB observation frames derived from the DIGIT sensor mesh.
                         Shape: (C, H, W) = (3, 224, 224), float32 tensor in [0, 1].
      - z_V_cached     : Static visual feature representations pre-computed using a 
                         frozen OpenVLA-7B encoder during Stage 1 operations.
                         Shape: (D_VISUAL,) = (2176,).

    Methodological Rationalization for Visual Latent Caching:
    ─────────────────────────────────────────────────────────
    The OpenVLA model encompasses 7.5B parameters, introducing severe VRAM footprint constraints 
    that prevent concurrent instantiation with the projection mapping layers on standard hardware. 
    Pre-computing and caching the target representations z_V onto persistent storage decouples 
    the optimization routine, transforming training into an isolated mapping operation between 
    f_theta and highly lightweight tensor structures.
    """

    def __init__(self, data_dir: str):
        import os, glob
        self.samples = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.samples) == 0:
            raise ValueError(f"No valid serialized tensor files (.pt) detected within: {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx], map_location="cpu")
        # Every verified storage payload contains a structural dictionary mapping:
        #   "tactile_image" : Tensor shape (3, 224, 224), float32 interval [0, 1]
        #   "z_V"           : Tensor shape (2176,), float32 unbounded projection
        return sample["tactile_image"], sample["z_V"]


class SyntheticDataset(Dataset):
    """
    Generates isolated, pseudo-random tensors mimicking production shapes.
    
    This module lacks any semantic representation and is exclusively utilized during 
    architectural sanity checks to verify tensor propagation, data dimensions, 
    and multi-gpu memory allocation safety mechanisms.
    """
    def __init__(self, n_samples: int = 128):
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        tactile = torch.rand(3, 224, 224)   # Synthesized raw image observation
        z_V     = torch.randn(D_VISUAL)     # Synthesized target visual latent token
        return tactile, z_V


# ─────────────────────────────────────────────────────────────
#  1. AUTOMATED ARCHITECTURE INTEGRITY VERIFICATION
# ─────────────────────────────────────────────────────────────

def run_sanity_check():
    """
    Executes an un-supervised, rigid evaluation matrix across 5 verification points:

      [1] Tensor Shape Topology : Validates structural dimension invariance along the network.
      [2] Numerical Stability   : Discovers undefined values (NaN) or infinity exceptions (Inf).
      [3] Gradient Propagation  : Verifies backward paths reach backpropagation entry layers.
      [4] Parametric Isolation  : Enforces frozen weight security within the TactileEncoder.
      [5] Parameter Update Loop : Confirms optimization step updates target projection weights.

    Any deviation from the constraints defined in this procedure triggers immediate execution 
    termination to prevent unviable resource usage.
    """
    print("=" * 60)
    print("INTEGRITY VERIFICATION — Synthetic Telemetry (Bypassing Checkpoints)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}\n")

    # Instantiate model using un-initialized weights to focus exclusively on computational mechanics
    projector = _build_projector_no_checkpoint().to(device)
    projector.train()

    criterion = FeatureAlignmentLoss(lambda_cos=1.0)
    optimizer = optim.AdamW(projector.trainable_parameters(), lr=1e-4)

    # Synthesized evaluation batch parameters
    B = 8
    tactile_images = torch.rand(B, 3, 224, 224).to(device)
    z_V_target     = torch.randn(B, D_VISUAL).to(device)

    # ── [1] Forward Pass Propagation Verification ────────────────────
    z_V_hat = projector(tactile_images)
    assert z_V_hat.shape == (B, D_VISUAL), \
        f"Structural Mismatch Exception: Expected ({B}, {D_VISUAL}), intercepted {z_V_hat.shape}"
    print(f"[1] Output Space Dimension Interface : {z_V_hat.shape}  ✓")

    # ── [2] Objective Function Numerical Verification ────────────────
    loss, info = criterion(z_V_hat, z_V_target)
    assert not torch.isnan(loss) and not torch.isinf(loss), \
        f"Numerical Instability Exception: Objective yielded out-of-bounds value {loss.item()}"
    print(f"[2] Baseline Multi-Objective Scalar  : {loss.item():.4f}  ✓")
    print(f"    MSE Component: {info['loss_mse']:.4f}  |  Angular Component: {info['loss_cosine']:.4f}")

    # ── [3] Trainable Sub-Graph Gradient Tracking Verification ────────
    optimizer.zero_grad()
    loss.backward()

    # Verify that structural optimization paths are actively propagating derivatives
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in projector.projection_head.parameters()
    )
    assert has_grad, "Gradient Routing Failure: Trainable projection layers isolated from backward pass."
    print(f"[3] Trainable Mapping Derivative Flow: Verified Active  ✓")

    # ── [4] Frozen Sub-Graph Isolation Verification ──────────────────
    no_grad_encoder = all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in projector.tactile_encoder.parameters()
    )
    assert no_grad_encoder, "Isolation Failure: Frozen backbone state mutated by backpropagation graph."
    print(f"[4] Backbone Parameter Freeze State  : Secure Isolation  ✓")

    # ── [5] Numerical Optimizer Step Verification ────────────────────
    weights_before = projector.projection_head.net[0].weight.data.clone()
    optimizer.step()
    weights_after  = projector.projection_head.net[0].weight.data

    changed = not torch.allclose(weights_before, weights_after)
    assert changed, "Optimization Failure: Weight delta remains static after parameter updates."
    print(f"[5] Numerical Parameter Adaptation   : Operational  ✓")

    print("\nINTEGRITY VERIFICATION SUCCESSFUL — Network pipeline structurally viable.")
    print("=" * 60)


def _build_projector_no_checkpoint():
    """
    Architectural factory mock utility. Instantiates the computational graphs 
    without initiating high-latency I/O operations from external parameter caches.
    """
    from src.models.model import ProjectionHead
    import torch.nn as nn

    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            import timm
            vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
            for p in vit.parameters():
                p.requires_grad = False

            self.tactile_encoder = nn.Sequential(vit)

            class _FrozenWrapper(nn.Module):
                def __init__(self, vit):
                    super().__init__()
                    self.vit = vit
                def forward(self, x):
                    from torchvision import transforms
                    x = transforms.functional.resize(x, [224, 224])
                    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(x.device)
                    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(x.device)
                    x = (x - mean) / std
                    return self.vit(x)
                def parameters(self, recurse=True):
                    return self.vit.parameters(recurse)

            self.tactile_encoder = _FrozenWrapper(vit)
            self.projection_head = ProjectionHead()

        def forward(self, x):
            with torch.no_grad():
                z_T = self.tactile_encoder(x)
            return self.projection_head(z_T)

        def trainable_parameters(self):
            return self.projection_head.parameters()

    return MockProjector()


# ─────────────────────────────────────────────────────────────
#  2. MAIN MULTI-MODAL OPTIMIZATION LOOP
# ─────────────────────────────────────────────────────────────

def train(
    data_dir:       str,
    tvl_ckpt:       str,
    output_dir:     str   = "checkpoints",
    n_epochs:       int   = 50,
    batch_size:     int   = 32,
    lr:             float = 1e-4,
    lambda_cos:     float = 1.0,
    val_split:      float = 0.1,
    log_every:      int   = 10,
    save_every:     int   = 5,
):
    """
    Orchestrates the formal cross-modal manifold alignment training and validation procedures.

    Epoch Processing Pipeline sequence:
      1. Optimization Step  : Iterative supervised learning via backpropagation.
      2. Performance Review : Deterministic assessment using un-tracked validation runs.
      3. Global Telemetry   : Formatted console reports of loss states and cosine similarity metrics.
      4. Target Persistency : Periodic serialization of top-performing mapping parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Target Training Execution Frame: {device}")

    # ── Dataset Allocation & Loader Pipelines ────────────────────────
    dataset = TactileVisualDataset(data_dir)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Manifold Cohort Strategy: Allocate {n_train} steps to training, {n_val} to validation.")

    # ── Model Instantiation ──────────────────────────────────────────
    projector = TactileProjector(tvl_ckpt).to(device)
    projector.train()

    # ── Loss Formulations & Optimization Routines ────────────────────
    criterion = FeatureAlignmentLoss(lambda_cos=lambda_cos)

    # AdamW incorporates un-decoupled weight decay constraints targeting weights exclusively over biases
    optimizer = optim.AdamW(projector.trainable_parameters(), lr=lr, weight_decay=1e-4)

    # Dynamic Learning Rate Scheduling: Reduces scale when evaluation saturation occurs
    # Monitored parameter tracking patience constraint defined as a 5-epoch envelope
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_cos_sim = -1.0   # Baseline threshold for tracking optimal structural checkpoints

    for epoch in range(1, n_epochs + 1):

        # ── OPTIMIZATION PHASE ───────────────────────────────────────
        projector.train()
        train_loss = 0.0
        train_cos  = 0.0

        for step, (tactile, z_V) in enumerate(train_loader, 1):
            tactile = tactile.to(device)
            z_V     = z_V.to(device)

            optimizer.zero_grad()
            z_V_hat       = projector(tactile)
            loss, info    = criterion(z_V_hat, z_V)
            loss.backward()

            # Gradient norm clipping constraints: Prevents geometric saturation anomalies
            torch.nn.utils.clip_grad_norm_(
                projector.projection_head.parameters(), max_norm=1.0
            )

            optimizer.step()

            train_loss += info["loss_total"]
            train_cos  += info["cos_sim_mean"]

            if step % log_every == 0:
                avg_loss = train_loss / step
                avg_cos  = train_cos  / step
                print(f"  Epoch {epoch:3d} | Step {step:4d} | "
                      f"Composite Loss {avg_loss:.4f} | Angular Congruence {avg_cos:.4f}")

        # ── EVALUATION PHASE ─────────────────────────────────────────
        projector.eval()
        val_metrics_accum = {
            "cosine_similarity_mean": 0.0,
            "mse":                    0.0,
            "relative_norm_error":    0.0,
        }
        n_val_steps = 0

        with torch.no_grad():
            for tactile, z_V in val_loader:
                tactile = tactile.to(device)
                z_V     = z_V.to(device)
                z_V_hat = projector(tactile)
                m = compute_alignment_metrics(z_V_hat, z_V)
                for k in val_metrics_accum:
                    val_metrics_accum[k] += m[k]
                n_val_steps += 1

        for k in val_metrics_accum:
            val_metrics_accum[k] /= n_val_steps

        val_cos = val_metrics_accum["cosine_similarity_mean"]
        val_mse = val_metrics_accum["mse"]
        val_nrm = val_metrics_accum["relative_norm_error"]

        print(f"Epoch {epoch:3d} | Val CosSim {val_cos:.4f} | "
              f"Val MSE {val_mse:.4f} | Val NormErr {val_nrm:.4f}")

        # Execute optimization learning rate reduction pass using Mean Squared Error criteria
        scheduler.step(val_mse)

        # ── PERSISTENCE ENGINE ───────────────────────────────────────
        if val_cos > best_cos_sim:
            best_cos_sim = val_cos
            torch.save(
                projector.projection_head.state_dict(),
                f"{output_dir}/best_projection_head.pth"
            )
            print(f"  -> Optimal alignment boundary verified (Validation Cosine Similarity: {val_cos:.4f})")

        if epoch % save_every == 0:
            torch.save(
                projector.projection_head.state_dict(),
                f"{output_dir}/projection_head_epoch{epoch:03d}.pth"
            )

    print(f"\nOptimization cycle successfully completed. Highest recorded cosine similarity: {best_cos_sim:.4f}")
    print(f"Optimal serialized checkpoint written to: {output_dir}/best_projection_head.pth")


# ─────────────────────────────────────────────────────────────
#  RUNTIME EXECUTION PARSER
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["sanity", "train"], default="sanity")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--tvl_ckpt", type=str, default="")
    parser.add_argument("--out_dir",  type=str, default="checkpoints")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=1e-4)
    args = parser.parse_args()

    if args.mode == "sanity":
        run_sanity_check()
    else:
        assert args.data_dir and args.tvl_ckpt, \
            "Production Runtime Exception: Missing absolute parameters for --data_dir or --tvl_ckpt"
        train(
            data_dir=args.data_dir,
            tvl_ckpt=args.tvl_ckpt,
            output_dir=args.out_dir,
            n_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )