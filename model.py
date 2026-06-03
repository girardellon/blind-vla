"""
model.py — Tactile-to-Visual Projection Head (f_theta)

This module implements the cross-modal architecture for projecting high-dimensional 
tactile signals into the visual latent space of a Vision-Language-Action (VLA) policy.

Architecture Overview:
  1. TactileEncoder  : Wraps a frozen Vision Transformer (ViT-Small) pre-trained 
                       by TVL, automatically fetched from HuggingFace Hub.
                       Input  : (B, C, H_t, W_t) RGB tactile images normalized in [0, 1].
                       Output : z_T ∈ R^512 (Global [CLS] embedding).
  2. ProjectionHead  : The trainable mapping network (f_theta) that bridges modalities.
                       Input  : z_T ∈ R^512.
                       Output : z_V_hat ∈ R^2176 (Compatible with OpenVLA visual tokens).
  3. TactileProjector: End-to-end wrapper unifying both encoder and projection modules.

Automated Checkpoint Management:
  Upon initialization, TactileEncoder automatically checks the specified 'cache_dir' 
  (default: "./hf_cache") for the pre-trained weights 'tvl_enc_vits.pth' (~264 MB). 
  If not found locally, it downloads the file dynamically from the official repository.
"""

import torch
import torch.nn as nn
from torchvision import transforms


# ─────────────────────────────────────────────────────────────
#  ARCHITECTURE CONSTANTS
# ─────────────────────────────────────────────────────────────

D_TACTILE = 512    # Dimensionality of the TVL ViT-Small global embedding [CLS]
                   # (ViT-Base → 768, ViT-Tiny → 192)
D_HIDDEN  = 1024   # Intermediate hidden dimension for the projection head f_theta
D_VISUAL  = 2176   # Target OpenVLA visual patch embedding dimension (Phase 1 verified)
                   # Expected tensor shape for z_V: [1, 256, 2176]

# Remote Hub repository path definitions for automated weight synchronization
TVL_REPO_ID  = "mlfu7/Touch-Vision-Language-Models"
TVL_FILENAME = "ckpt/tvl_enc/tvl_enc_vits.pth"


# ─────────────────────────────────────────────────────────────
#  UTILITIES: Checkpoint Synchronization & Robust Parsing
# ─────────────────────────────────────────────────────────────

def download_tvl_checkpoint(cache_dir: str = "./hf_cache") -> str:
    """
    Synchronizes the remote TVL ViT-Small checkpoint with the local filesystem.

    Leverages the huggingface_hub API to automatically handle:
      - Caching mechanism: bypasses downloading if local file hash matches.
      - Integrity checks: computes cryptographic checksum validation.
      - Real-time telemetry: provides progress bars during transmission.

    Args:
        cache_dir (str): Directory path allocated for huggingface assets.
            For persistent execution environments (e.g., Kaggle), specify 
            "/kaggle/working/hf_cache".

    Returns:
        str: Absolute system path pointing to the verified .pth checkpoint file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' library is missing from the environment.\n"
            "Please resolve dependencies via: pip install huggingface_hub"
        )

    print(f"Synchronizing TVL ViT-Small checkpoint from HuggingFace Hub...")
    print(f"  Repository ID : {TVL_REPO_ID}")
    print(f"  Target File   : {TVL_FILENAME}")
    print(f"  Cache Storage : {cache_dir}")

    local_path = hf_hub_download(
        repo_id   = TVL_REPO_ID,
        filename  = TVL_FILENAME,
        cache_dir = cache_dir,
    )

    print(f"Checkpoint successfully located at: {local_path}\n")
    return local_path


def _load_state_dict_robust(checkpoint_path: str) -> dict:
    """
    A parsing utility designed to robustly ingest heterogeneous checkpoint topologies.

    The official TVL checkpoints may be distributed under varying serializations:
      (a) Flat state_dict : Directly mapping keys (e.g., "patch_embed.proj.weight") to tensors.
      (b) Nested root     : Nested inside an intermediate dictionary under the "model" key.
      (c) Custom sub-keys : Nested under alternative identifiers ("state_dict", "encoder", etc.).

    This method adaptively isolates the core parameter state_dict to prevent structural mismatches.

    Returns:
        dict: A flat dictionary mapping layer names to parameters, ready for model ingestion.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")

    # Paradigm A: Flat state_dict mapping (identified by hierarchical dot-notation)
    if isinstance(raw, dict) and any("." in k for k in raw.keys()):
        print(f"  Checkpoint topology resolved: Flat state_dict ({len(raw)} entries)")
        return raw

    # Paradigm B: Nested dictionary structures matching known sub-key identifiers
    for key in ("model", "state_dict", "encoder", "tactile_encoder", "backbone"):
        if isinstance(raw, dict) and key in raw:
            inner = raw[key]
            print(f"  Checkpoint topology resolved: Nested under key '{key}' ({len(inner)} entries)")
            return inner

    # Paradigm C: Full serialized nn.Module objects rather than state_dicts
    if hasattr(raw, "state_dict"):
        print("  Checkpoint topology resolved: Serialized nn.Module object. Extracting state_dict.")
        return raw.state_dict()

    # Fallback routine: Yields raw payload and defers structural errors to subsequent steps
    print(f"  WARNING: Unrecognized checkpoint serialization format.")
    print(f"  Root payload keys detected: {list(raw.keys())[:10]}")
    print(f"  Attempting to load payload directly (structural exception may follow).")
    return raw


# ─────────────────────────────────────────────────────────────
#  1. TACTILE ENCODER MODULE (Frozen Backbone)
# ─────────────────────────────────────────────────────────────

class TactileEncoder(nn.Module):
    """
    Tactile feature extractor built upon the pre-trained TVL ViT-Small backbone.

    During initialization, the network synchronizes weights from the remote hub,
    instantiates the vision transformer architecture, and explicitly isolates its 
    parameters from the backpropagation graph.

    Methodological Rationalization for Stage 2 Freezing:
    ────────────────────────────────────────────────────
    The TVL ViT has been pre-aligned with contrastive image-text latent spaces specifically 
    for DIGIT tactile frames. Jointly optimizing the encoder with the projection head f_theta 
    couples two separate problems: tactile feature representation learning and cross-modal 
    latent manifold projection. Keeping the encoder frozen decouples these phenomena, ensuring 
    numerical stability. Full fine-tuning may be unlocked if target cosine thresholds 
    are not achieved.

    Input  Dimensions: (B, C, H, W) float32 tensor scaled to interval [0, 1].
    Output Dimensions: (B, 512) float32 tensor representing tactile latent z_T.
    """

    def __init__(self, cache_dir: str = "./hf_cache"):
        super().__init__()

        # Synchronize model weights using local cache or remote retrieval
        checkpoint_path = download_tvl_checkpoint(cache_dir)

        # Instantiate a Vision Transformer architecture using the 'timm' registry
        # Setting num_classes=0 isolates the global [CLS] token pooling layer (512-dim output)
        try:
            import timm
        except ImportError:
            raise ImportError(
                "The 'timm' dependency is missing from the environment.\n"
                "Please resolve via: pip install timm"
            )

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,   # Disable ImageNet weights to ingest target TVL parameters
            num_classes=0,      # Isolates the global pooled embedding output
        )

        # Load optimized TVL state dictionary
        state_dict = _load_state_dict_robust(checkpoint_path)

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        # strict=False accommodates deliberate mismatches, such as the exclusion
        # of classification heads or specific TVL contrastive projectors.
        print(f"  TVL ViT-Small configuration parsed.")
        print(f"  Missing parameters    : {len(missing)}  {missing[:3] if missing else ''}")
        print(f"  Unexpected parameters : {len(unexpected)}  {unexpected[:3] if unexpected else ''}")
        if any("patch_embed" in k or "blocks" in k for k in missing):
            print("  CRITICAL WARNING: Essential backbone layers are missing from the state dict."
                  " Verify checkpoint compatibility with tvl_enc_vits.pth.")

        # Explicitly decouple backbone parameters from gradient computation graph
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()   # Enforces evaluation mode (disables dropout/batch normalization)

        # Standard ImageNet pre-processing pipeline matching pre-training distribution
        # Dynamic resizing is enforced to allow flexible downstream input configurations.
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a deterministic forward pass through the frozen tactile encoder.

        Args:
            x (torch.Tensor): Raw image tensor of shape (B, C, H, W) in [0, 1].

        Returns:
            torch.Tensor: Global feature embedding z_T of shape (B, 512).
        """
        with torch.no_grad():
            x   = self.preprocess(x)   # Output shape: (B, 3, 224, 224)
            z_T = self.vit(x)          # Output shape: (B, 512)
        return z_T


# ─────────────────────────────────────────────────────────────
#  2. MAPPING NETWORK / PROJECTION HEAD (Trainable f_theta)
# ─────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    f_theta: z_T → z_V_hat

    Architecture Pipeline (Linear Embedding + Fully Connected Network):
        Linear(512  → 1024) -> LayerNorm -> GELU  [Linear Embedding Layer]
        Linear(1024 → 1024) -> LayerNorm -> GELU  [Hidden Fully Connected Block]
        Linear(1024 → 2176)                        [Output Projection Layer]

    Design Justification Matrix:
    ────────────────────────────
    1. Layer Normalization (LayerNorm) over BatchNorm:
       BatchNorm aggregates batch-level variance, exposing it to instability when
       operating under low batch regimes. LayerNorm operates along the feature 
       dimension independently per sample, yielding deterministic stability. It 
       is also architectural standard inside OpenVLA's Transformer (Llama 2).
    2. GELU Activation over ReLU:
       GELU scales activations smoothly, avoiding the hard zero truncation of 
       standard ReLUs that leads to the "dying ReLU" gradient-saturation pathology.
    3. Unbounded Output Projection:
       The target visual space (z_V) consists of unconstrained real numbers. 
       Applying saturating or bounding activations (ReLU, Sigmoid, Tanh) would 
       artificially distort output manifold distribution matching.
    4. Xavier Uniform Initialization:
       Maintains uniform variance across sequential layer operations to protect 
       against vanishing or exploding gradient dynamics during initialization.
    """

    def __init__(
        self,
        d_in:     int = D_TACTILE,   # 512
        d_hidden: int = D_HIDDEN,    # 1024
        d_out:    int = D_VISUAL,    # 2176
    ):
        super().__init__()

        self.net = nn.Sequential(
            # ── Linear Embedding Layer ────────────────────────────────
            nn.Linear(d_in, d_hidden),    # 512  → 1024
            nn.LayerNorm(d_hidden),
            nn.GELU(),

            # ── Hidden Fully Connected Block ──────────────────────────
            nn.Linear(d_hidden, d_hidden),  # 1024 → 1024
            nn.LayerNorm(d_hidden),
            nn.GELU(),

            # ── Output Projection Layer ───────────────────────────────
            nn.Linear(d_hidden, d_out),   # 1024 → 2176
        )

        self._init_weights()

    def _init_weights(self):
        """
        Applies Xavier Uniform weight initialization across all Linear operations.
        Biases are initialized to an explicit zero state.
        """
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z_T: torch.Tensor) -> torch.Tensor:
        """
        Maps tactile features to visual tokens.

        Args:
            z_T (torch.Tensor): Feature tensor of shape (B, 512).

        Returns:
            torch.Tensor: Aligned visual prediction z_V_hat of shape (B, 2176).
        """
        return self.net(z_T)


# ─────────────────────────────────────────────────────────────
#  3. END-TO-END SYSTEM WRAPPER
# ─────────────────────────────────────────────────────────────

class TactileProjector(nn.Module):
    """
    Unified entry point mapping raw tactile inputs directly to target visual latents.

    Encapsulates both the frozen feature extractor and the trainable multi-layer 
    projection mapping network.

    Standard Training Pipeline Integration:
    ──────────────────────────────────────
        projector = TactileProjector(cache_dir="./hf_cache")

        # Explicitly isolate the optimizer to trainable parameters only:
        optimizer = torch.optim.AdamW(
            projector.trainable_parameters(), lr=1e-4
        )

        # Forward execution and parameter update:
        z_V_hat = projector(tactile_images)   # Output shape: (B, 2176)
        loss, _ = criterion(z_V_hat, z_V_cached)
        loss.backward()                       # Propagates gradients exclusively to f_theta
        optimizer.step()
    """

    def __init__(self, cache_dir: str = "./hf_cache"):
        super().__init__()
        self.tactile_encoder = TactileEncoder(cache_dir=cache_dir)
        self.projection_head = ProjectionHead()

    def forward(self, tactile_images: torch.Tensor) -> torch.Tensor:
        """
        Processes tactile frames to extract and project cross-modal embeddings.

        Args:
            tactile_images (torch.Tensor): Normalized input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted OpenVLA-compatible features z_V_hat of shape (B, 2176).
        """
        z_T     = self.tactile_encoder(tactile_images)   # Executed under frozen evaluation mode
        z_V_hat = self.projection_head(z_T)              # Tracks and propagates gradients
        return z_V_hat

    def trainable_parameters(self):
        """
        Isolates parameters tracking gradients to protect frozen modules.

        Returns:
            generator: Generator expression containing parameters belonging exclusively to f_theta.
        """
        return self.projection_head.parameters()

    def count_parameters(self) -> dict:
        """
        Computes the module network parameter distribution profile.

        Returns:
            dict: Quantified analysis of trainable vs. frozen network parameters.
        """
        trainable = sum(p.numel() for p in self.projection_head.parameters())
        frozen    = sum(p.numel() for p in self.tactile_encoder.parameters())
        return {
            "projection_head_trainable": trainable,
            "tactile_encoder_frozen":    frozen,
            "total":                     trainable + frozen,
        }