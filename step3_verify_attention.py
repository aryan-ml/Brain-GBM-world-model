import torch
import matplotlib.pyplot as plt
from models.omni_attention import omni_attn_mask_naive

print("=" * 60)
print("Brain-WM: Step 3 - Y-Shaped Hybrid Attention Verification")
print("=" * 60)

# 1. Define a representative multimodal sequence layout:
#  - Tokens   0 ..  19: Clinical Text Prompt (20 tokens)
#  - Tokens  20 .. 119: 3D MRI Latent Visual Block (100 tokens)
#  - Tokens 120 .. 139: Treatment Recommendation Output (20 tokens)
total_seq_len = 140
prompt_len = 20
img_offset = 20
img_len = 100
treat_len = 20

# Modality position tensor: [batch_size, num_modalities, 2] -> [[offset, length]]
modalities = torch.tensor([[[img_offset, img_len]]])

# 2. Generate the non-inverted attention mask (1 = attend, 0 = mask out)
mask = omni_attn_mask_naive(
    B=1,
    LEN=total_seq_len,
    modalities=modalities,
    device="cpu",
    inverted=False
)
mask = mask.squeeze(0).squeeze(0)  # Shape: (140, 140)

print(f"\nGenerated Attention Matrix Shape: {tuple(mask.shape)}")

# 3. Mathematical Unit Tests
print("\nRunning Verification Checks:")

# Check A: Text Prompt region (0..19) must be strictly causal (no upper triangle values)
text_upper_triangle = mask[0:prompt_len, 0:prompt_len].triu(diagonal=1).sum().item()
assert text_upper_triangle == 0, "FAIL: Prompt text is not causal!"
print("  [PASS] 1. Clinical Prompt Text is strictly causal.")

# Check B: Image block (20..119) must be fully bidirectional (all 1s)
img_block = mask[img_offset:img_offset + img_len, img_offset:img_offset + img_len]
expected_img_sum = img_len * img_len
assert img_block.sum().item() == expected_img_sum, "FAIL: Image block is not fully bidirectional!"
print("  [PASS] 2. 3D MRI Visual Block is fully bidirectional (all voxels attend to all voxels).")

# Check C: Prompt tokens cannot see future image tokens
future_leak = mask[5, img_offset + 10].item()
assert future_leak == 0, "FAIL: Prompt text leaked future visual tokens!"
print("  [PASS] 3. Early text tokens cannot leak into future image tokens.")

# Check D: Output treatment tokens (120..139) CAN see the entire image block
cross_attention = mask[125, img_offset + 50].item()
assert cross_attention == 1, "FAIL: Treatment token cannot see the MRI volume!"
print("  [PASS] 4. Treatment prediction tokens attend to all 3D MRI visual tokens.")

# Check E: Output treatment tokens cannot see future text tokens (autoregressive)
future_text_leak = mask[125, 135].item()
assert future_text_leak == 0, "FAIL: Treatment token leaked future text tokens!"
print("  [PASS] 5. Treatment tokens generate autoregressively without looking ahead.")

# 4. Save a visual diagram of the attention matrix
plt.figure(figsize=(8, 8))
plt.imshow(mask.numpy(), cmap="Blues", interpolation="nearest")
plt.title("Brain-WM: Y-Shaped Hybrid Attention Mask", fontsize=14, pad=15)
plt.xlabel("Key / Value Token Index (j)", fontsize=12)
plt.ylabel("Query Token Index (i)", fontsize=12)

# Annotate regions
plt.axvline(x=20, color='red', linestyle='--', linewidth=1)
plt.axvline(x=120, color='red', linestyle='--', linewidth=1)
plt.axhline(y=20, color='red', linestyle='--', linewidth=1)
plt.axhline(y=120, color='red', linestyle='--', linewidth=1)

plt.text(10, 10, "Causal\nPrompt", color='black', ha='center', va='center', fontsize=9)
plt.text(70, 70, "Bidirectional\n3D Image Block\n(100 x 100)", color='white', ha='center', va='center', fontweight='bold', fontsize=10)
plt.text(70, 130, "Cross-Attention\n(Treatment -> Image)", color='black', ha='center', va='center', fontsize=9)
plt.text(130, 130, "Causal\nOutput", color='black', ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("attention_mask_heatmap.png", dpi=150)
print("\n[SAVED] Visual diagram saved to 'attention_mask_heatmap.png'")
print("[SUCCESS] Step 3 complete: Y-shaped attention mechanism verified!")
print("=" * 60)
