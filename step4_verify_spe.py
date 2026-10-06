from models.modeling_showo2_qwen2_5 import SpatialAPE3D_HWD
spatial_size = (192, 192, 140)  # (H, W, D)
comp = (4, 16, 16)              # Compression: cD=4, cH=16, cW=16
hidden_size = 1536
spe = SpatialAPE3D_HWD(
    dim=hidden_size,
    spatial_size=spatial_size,
    comp=comp,
    num_freqs=8,
)
print(f"Grid dimensions (Dg, Hg, Wg): ({spe.Dg}, {spe.Hg}, {spe.Wg})")
print(f"Total tokens N: {spe.N}")  # Expected: 35 * 12 * 12 = 5040 tokens
pos_embed = spe.proj(spe.feats)
print(f"Positional embedding tensor shape: {pos_embed.shape}")  # (5040, 1536)
print("3D Positional Encoding verified")
