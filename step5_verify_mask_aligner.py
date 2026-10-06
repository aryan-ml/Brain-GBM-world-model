import torch
from models.segmentor import TokensToVolume
from models.misc import seg_prediction

print("=" * 60)
print("Brain-WM: Step 5 - Multi-Scale Mask Aligner & Loss Verification")
print("=" * 60)

# 1. Dimensions
spatial_size = (192, 192, 140)
comp = (4, 16, 16)
hidden_size = 1536
N_tokens = 5040  # 35 * 12 * 12

print("\nInitializing TokensToVolume segmentor (first_pow2=64 for local memory)...")
segmentor = TokensToVolume(
    spatial_size=spatial_size,
    comp=comp,
    in_channels=hidden_size,
    out_channels=4,
    first_pow2=64,
    use_ckpt=False
)

B = 1
# Multi-scale features from shallow, middle, and deep transformer layers
feat_low = torch.randn(B, N_tokens, hidden_size)
feat_mid = torch.randn(B, N_tokens, hidden_size)
feat_high = torch.randn(B, N_tokens, hidden_size)

print("Running 3D Feature Pyramid forward pass...")
logits = segmentor(feat_low, feat_mid, feat_high)
print(f"  Segmentor Output Logits shape: {tuple(logits.shape)} (B, Classes=4, D, H, W)")
assert logits.shape == (B, 4, 140, 192, 192), "Output logits shape mismatch!"

# 2. Compute Loss with Target Segmentation Mask
# Labels: 0 = Background, 1 = Necrotic Core, 2 = Peritumoral Edema, 3 = Enhancing Tumor
target_mask = torch.randint(0, 4, (B, 1, 140, 192, 192)).float()

# Unpack the 3 loss components returned by seg_prediction
total_loss, ce_loss, dice_loss = seg_prediction(logits, target_mask)

print("\n--- Anatomical Alignment Loss Breakdown ---")
print(f"  1. Focal Cross-Entropy Loss: {ce_loss.item():.4f}")
print(f"  2. Multi-Class Dice Loss:    {dice_loss.item():.4f}")
print(f"  3. Total Combined Loss:       {total_loss.item():.4f}")

print("\n[SUCCESS] Step 5 complete: Mask Aligner and loss functions verified!")
print("=" * 60)
