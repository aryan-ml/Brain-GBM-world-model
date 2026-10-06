import os
import torch
import monai.transforms as mtransforms
import matplotlib.pyplot as plt

# 1. Define paths to our linked toy dataset
data_dir = "data/toy_data/BRATS/BraTS20_Training_001/Timepoint_1"
input_dict = {
    "flair1": os.path.join(data_dir, "flair.nii"),
    "t1ce1":  os.path.join(data_dir, "t1c.nii"),
    "t2w1":   os.path.join(data_dir, "t2.nii"),
    "seg1":   os.path.join(data_dir, "seg_mask.nii"),
}

print("Running MONAI transformation pipeline...")

# 2. Replicate the exact transformation pipeline from datasets/glioma_dataset.py
pipeline = mtransforms.Compose([
    # Load 3D NIfTI images as PyTorch tensors (Channel First)
    mtransforms.LoadImaged(
        keys=["flair1", "t1ce1", "t2w1", "seg1"],
        image_only=True,
        ensure_channel_first=True
    ),
    # Ensure isotropic voxel spacing (1mm x 1mm x 1mm)
    mtransforms.Spacingd(
        keys=["flair1", "t1ce1", "t2w1"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear",) * 3
    ),
    mtransforms.Spacingd(
        keys=["seg1"],
        pixdim=(1.0, 1.0, 1.0),
        mode="nearest"  # Nearest neighbor preserves discrete segmentation class labels
    ),
    # Scale intensities: clip 99.9th percentile artifact and normalize to [-1.0, 1.0]
    mtransforms.ScaleIntensityRangePercentilesd(
        keys=["flair1", "t1ce1", "t2w1"],
        lower=0, upper=99.9,
        b_min=-1.0, b_max=1.0,
        clip=True, relative=False
    ),
    # Crop central brain volume to exactly (192, 192, 140)
    mtransforms.SpatialCropd(
        keys=["flair1", "t1ce1", "t2w1", "seg1"],
        roi_center=(120, 120, 78),
        roi_size=(192, 192, 140)
    ),
])

# Execute pipeline on our files
out = pipeline(input_dict)

print("\n--- Intermediate Modality Shapes ---")
for k in ["flair1", "t1ce1", "t2w1", "seg1"]:
    print(f"  {k:8s}: shape = {tuple(out[k].shape)}, range = [{out[k].min():.2f}, {out[k].max():.2f}]")

# 3. Stack into the 3-channel 3D volume required by Brain-WM
# MONAI outputs (Channel=1, H=192, W=192, D=140).
# permute(0, 3, 1, 2) reorders to (Channel=1, D=140, H=192, W=192).
stacked_volume = torch.cat([
    out['flair1'].permute(0, 3, 1, 2),
    out['t1ce1'].permute(0, 3, 1, 2),
    out['t2w1'].permute(0, 3, 1, 2)
], dim=0)

seg_mask = out['seg1'].permute(0, 3, 1, 2)

print("\n--- Final Stacked Model Input Tensors ---")
print(f"Stacked 3D MRI Volume shape: {tuple(stacked_volume.shape)}  (Channels=3, Depth=140, Height=192, Width=192)")
print(f"Segmentation Mask shape:    {tuple(seg_mask.shape)}  (Channels=1, Depth=140, Height=192, Width=192)")
print(f"Unique Tumor Labels:        {torch.unique(seg_mask).tolist()}")

# 4. Save a visual slice check
# Slice 70 is near the center of the brain axial plane
slice_idx = 70
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(stacked_volume[0, slice_idx, :, :].numpy(), cmap='gray')
axes[0].set_title("FLAIR (Modality 0)")
axes[0].axis('off')

axes[1].imshow(stacked_volume[1, slice_idx, :, :].numpy(), cmap='gray')
axes[1].set_title("T1-Contrast (Modality 1)")
axes[1].axis('off')

axes[2].imshow(stacked_volume[2, slice_idx, :, :].numpy(), cmap='gray')
axes[2].set_title("T2 (Modality 2)")
axes[2].axis('off')

axes[3].imshow(seg_mask[0, slice_idx, :, :].numpy(), cmap='jet')
axes[3].set_title("Tumor Mask (Ground Truth)")
axes[3].axis('off')

plt.tight_layout()
plt.savefig("axial_slice_check.png", dpi=150)
print("\n[SAVED] Visual slice check saved to 'axial_slice_check.png'")
print("[SUCCESS] Step 2 complete: Preprocessing pipeline verified!")
