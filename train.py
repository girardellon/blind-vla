
"""
train.py — Automated Integrity Verification & Alignment Training Loop (Stage 2)

Input Strategy: Option B (Channel Concatenation)
  digit_left.png and digit_right.png are concatenated channel-wise into a single
  (B, 6, H, W) tensor. The Dataset returns this fused tensor; the model handles
  the 6→3 channel projection internally via the trainable channel_adapter.

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
      python -m src.training.train --mode train --data_dir /path/to/dataset --tvl_ckpt /path/to/hf_cache
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

# Absolute package imports mapped across the repository topology
from model import TactileProjector, D_VISUAL
from loss import FeatureAlignmentLoss, compute_alignment_metrics


# ─────────────────────────────────────────────────────────────
#  DATASET MANIFESTS & PIPELINES
# ─────────────────────────────────────────────────────────────

class TactileVisualDataset(Dataset):
    """
    Ingestion pipeline for synchronized multi-modal pairs: (Dual-Finger Tactile, Visual Embedding).

    Input Strategy — Option B (Channel Concatenation):
      Each .pt file contains both digit_left and digit_right as separate (3, 224, 224)
      tensors. This Dataset concatenates them channel-wise at load time, producing a
      single (6, 224, 224) tensor that is passed directly to the TactileProjector.

    Expected .pt file structure:
      {
        "digit_left"  : Tensor (3, 224, 224), float32, [0, 1]   ← left finger DIGIT frame
        "digit_right" : Tensor (3, 224, 224), float32, [0, 1]   ← right finger DIGIT frame
        "z_V"         : Tensor (2176,),       float32           ← cached OpenVLA visual embedding
      }

    Returned batch item:
      - tactile_concat : (6, 224, 224) — channels [left_R, left_G, left_B, right_R, right_G, right_B]
      - z_V            : (2176,)
    """

    def __init__(self, data_dir: str):
        import os, glob
        self.samples = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.samples) == 0:
            raise ValueError(f"No valid serialized tensor files (.pt) detected within: {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx], map_location="cpu", weights_only=True)
        digit_left  = sample["digit_left"]    # (3, 224, 224)
        digit_right = sample["digit_right"]   # (3, 224, 224)
        z_V         = sample["z_V"].float()   # (2176,)
        # Concatenate along channel dim → (6, 224, 224)
        tactile_concat = torch.cat([digit_left, digit_right], dim=0)
        return tactile_concat, z_V


class SyntheticDataset(Dataset):
    """
    Generates pseudo-random tensors mimicking Option B production shapes.

    Exclusively used during architectural sanity checks.
    tactile shape: (6, 224, 224) — left+right DIGIT frames concatenated channel-wise.
    """
    def __init__(self, n_samples: int = 128):
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        tactile = torch.rand(6, 224, 224)   # 6 channels: 3 left + 3 right
        z_V     = torch.randn(D_VISUAL)
        return tactile, z_V


# ─────────────────────────────────────────────────────────────
#  1. AUTOMATED ARCHITECTURE INTEGRITY VERIFICATION
# ─────────────────────────────────────────────────────────────

def run_sanity_check():
    """
    Executes an un-supervised, rigid evaluation matrix across 5 verification points:

      [1] Tensor Shape Topology : Validates structural dimension invariance along the network.
      [2] Numerical Stability   : Discovers undefined values (NaN) or infinity exceptions (Inf).
      [3] Gradient Propagation  : Verifies backward paths reach both channel_adapter AND
                                  projection_head (both must receive gradients).
      [4] Parametric Isolation  : Enforces frozen weight security within the TVL ViT backbone.
      [5] Parameter Update Loop : Confirms optimizer step updates all trainable parameters.

    Any deviation from the constraints defined in this procedure triggers immediate execution 
    termination to prevent unviable resource usage.
    """
    print("=" * 60)
    print("INTEGRITY VERIFICATION — Synthetic Telemetry (Option B: 6-channel input)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}\n")

    projector = _build_projector_no_checkpoint().to(device)
    projector.train()

    criterion = FeatureAlignmentLoss(lambda_cos=1.0)
    optimizer = optim.AdamW(projector.trainable_parameters(), lr=1e-4)

    B = 8
    # Option B: 6-channel input — 3 channels left finger + 3 channels right finger
    tactile_images = torch.rand(B, 6, 224, 224).to(device)
    z_V_target     = torch.randn(B, D_VISUAL).to(device)

    # ── [1] Forward Pass Shape Verification ──────────────────────
    z_V_hat = projector(tactile_images)
    assert z_V_hat.shape == (B, D_VISUAL), \
        f"Shape mismatch: expected ({B}, {D_VISUAL}), got {z_V_hat.shape}"
    print(f"[1] Output Space Dimension Interface : {z_V_hat.shape}  ✓")

    # ── [2] Numerical Stability Verification ─────────────────────
    loss, info = criterion(z_V_hat, z_V_target)
    assert not torch.isnan(loss) and not torch.isinf(loss), \
        f"Numerical instability: loss={loss.item()}"
    print(f"[2] Baseline Multi-Objective Scalar  : {loss.item():.4f}  ✓")
    print(f"    MSE Component: {info['loss_mse']:.4f}  |  Angular Component: {info['loss_cosine']:.4f}")

    # ── [3] Gradient Flow Verification ───────────────────────────
    optimizer.zero_grad()
    loss.backward()

    adapter_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in projector.tactile_encoder.channel_adapter.parameters()
    )
    head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in projector.projection_head.parameters()
    )
    assert adapter_has_grad, "Gradient failure: channel_adapter did not receive gradients."
    assert head_has_grad,    "Gradient failure: projection_head did not receive gradients."
    print(f"[3] Gradient Flow — channel_adapter  : Active  ✓")
    print(f"    Gradient Flow — projection_head  : Active  ✓")

    # ── [4] ViT Backbone Freeze Verification ─────────────────────
    vit_no_grad = all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in projector.tactile_encoder.vit.parameters()
    )
    assert vit_no_grad, "Isolation failure: frozen ViT backbone received gradient updates."
    print(f"[4] ViT Backbone Freeze State        : Secure  ✓")

    # ── [5] Optimizer Step Verification ──────────────────────────
    adapter_w_before = projector.tactile_encoder.channel_adapter.weight.data.clone()
    head_w_before    = projector.projection_head.net[0].weight.data.clone()
    optimizer.step()
    assert not torch.allclose(adapter_w_before, projector.tactile_encoder.channel_adapter.weight.data), \
        "Optimizer failure: channel_adapter weights did not update."
    assert not torch.allclose(head_w_before, projector.projection_head.net[0].weight.data), \
        "Optimizer failure: projection_head weights did not update."
    print(f"[5] Parameter Update — channel_adapter: Updated  ✓")
    print(f"    Parameter Update — projection_head : Updated  ✓")

    print("\nINTEGRITY VERIFICATION SUCCESSFUL — Option B pipeline structurally viable.")
    print("=" * 60)

    print("\nINTEGRITY VERIFICATION SUCCESSFUL — Network pipeline structurally viable.")
    print("=" * 60)


def _build_projector_no_checkpoint():
    """
    Mock projector for sanity checks — mirrors the Option B TactileProjector architecture
    without downloading the TVL checkpoint.

    Includes a trainable channel_adapter (Conv2d 6→3) and a frozen random-weight ViT,
    matching the exact attribute names used by run_sanity_check.
    """
    from model import ProjectionHead, C_IN
    import torch.nn as nn
    from torchvision import transforms

    class _MockTactileEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            import timm

            # Trainable channel adapter: 6→3, same init as production TactileEncoder
            self.channel_adapter = nn.Conv2d(C_IN, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                w = torch.zeros(3, C_IN, 1, 1)
                for i in range(3):
                    w[i, i,     0, 0] = 0.5
                    w[i, i + 3, 0, 0] = 0.5
                self.channel_adapter.weight.copy_(w)
            # channel_adapter is trainable by default

            # Frozen random-weight ViT (no TVL checkpoint download)
            self.vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

            self._normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            )

        def forward(self, x):
            # channel_adapter: trainable, runs with gradient
            x = self.channel_adapter(x)          # (B, 6, H, W) → (B, 3, H, W)
            # ViT: frozen, no gradient
            with torch.no_grad():
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                x = self._normalize(x)
                z_T = self.vit(x)                # (B, 512)
            return z_T

    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = _MockTactileEncoder()
            self.projection_head = ProjectionHead()

        def forward(self, x):
            z_T = self.tactile_encoder(x)
            return self.projection_head(z_T)

        def trainable_parameters(self):
            return (
                list(self.tactile_encoder.channel_adapter.parameters())
                + list(self.projection_head.parameters())
            )

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
    projector = TactileProjector(cache_dir=tvl_ckpt).to(device)
    projector.train()

    # ── Loss Formulations & Optimization Routines ────────────────────
    criterion = FeatureAlignmentLoss(lambda_cos=lambda_cos)

    # AdamW incorporates un-decoupled weight decay constraints targeting weights exclusively over biases
    optimizer = optim.AdamW(projector.trainable_parameters(), lr=lr, weight_decay=1e-4)

    # Dynamic Learning Rate Scheduling: Reduces scale when evaluation saturation occurs
    # Monitored parameter tracking patience constraint defined as a 5-epoch envelope
    # verbose=False: deprecated in PyTorch >=2.2; LR changes are logged manually below
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
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
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_mse)
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr < prev_lr:
            print(f"  -> LR reduced: {prev_lr:.2e} → {curr_lr:.2e}")

        # ── PERSISTENCE ENGINE ───────────────────────────────────────
        # Save both channel_adapter and projection_head so the full trainable
        # state can be restored without re-running the frozen encoder.
        if val_cos > best_cos_sim:
            best_cos_sim = val_cos
            torch.save(
                {
                    "channel_adapter": projector.tactile_encoder.channel_adapter.state_dict(),
                    "projection_head": projector.projection_head.state_dict(),
                    "epoch":           epoch,
                    "val_cos_sim":     val_cos,
                },
                f"{output_dir}/best_projection_head.pth"
            )
            print(f"  -> New best checkpoint (Val CosSim: {val_cos:.4f})")

        if epoch % save_every == 0:
            torch.save(
                {
                    "channel_adapter": projector.tactile_encoder.channel_adapter.state_dict(),
                    "projection_head": projector.projection_head.state_dict(),
                    "epoch":           epoch,
                    "val_cos_sim":     val_cos,
                },
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