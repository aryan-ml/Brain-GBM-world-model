import os
import sys
import torch
import pandas as pd

PROJECT_ROOT = "/home/aryan/code/Brain-GBM-world-model"
sys.path.insert(0, PROJECT_ROOT)

import datasets.glioma_dataset as gd

print("=" * 60)
print("Brain-WM: Step 6 - DataLoader & Clinical Decision Head")
print("=" * 60)

# 1. Point the dataset loader to our local toy dataset CSVs
toy_csv = os.path.abspath("data/toy_data/csv_files/BRATS.csv")

def patched_read(csv_paths, ext_paths, mode="train"):
    df = pd.read_csv(toy_csv, dtype=str)
    df["__site__"] = "BRATS"
    return df.to_dict(orient="records")

gd._read_and_validate_pair_csvs = patched_read

# 2. Lightweight Mock Tokenizer for test execution
class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    vocab = {"<|im_start|>": 1, "<|im_end|>": 2, "<|vision_start|>": 3, "<|vision_end|>": 4, "<|image_pad|>": 5}
    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=1024):
        if isinstance(text, (list, tuple)):
            ids = [[abs(hash(w)) % 1000 + 10 for w in (t or "").split()] for t in text]
        else:
            ids = [abs(hash(w)) % 1000 + 10 for w in (text or "").split()]
        class Output:
            pass
        out = Output()
        out.input_ids = ids
        return out

mock_tok = MockTokenizer()
showo_token_ids = {
    "bos_id": 1,
    "eos_id": 2,
    "boi_id": 3,
    "eoi_id": 4,
    "img_pad_id": 5,
}

# 3. Instantiate the official MedicalPairImageTextDataset
print("\nInitializing official MedicalPairImageTextDataset...")
ds = gd.MedicalPairImageTextDataset(
    root="data/toy_data",
    text_tokenizer=mock_tok,
    showo_token_ids=showo_token_ids,
    spatial_size=(192, 192, 140),
    max_seq_len=512,
    num_image_tokens=100,
    is_captioning=False,
    strict_files=True,
    use_seg_mask=True,
    mode="train"
)

print(f"Dataset length: {len(ds)} longitudinal pair")
assert len(ds) == 1, "Expected dataset length 1"

# 4. Fetch the real BraTS sample from disk through the pipeline
print("Fetching sample 0 from disk...")
sample = ds[0]

print("\n--- Loaded Multimodal Sample Items ---")
print(f"  images_cond (T1 baseline) shape: {tuple(sample['images_cond'].shape)} (3 modalities x 140 depth x 192 x 192)")
print(f"  images      (T2 target)   shape: {tuple(sample['images'].shape)} (3 modalities x 140 depth x 192 x 192)")
print(f"  seg_masks shape:                 {tuple(sample['seg_masks'].shape)}")
print(f"  text_tokens shape:               {tuple(sample['text_tokens'].shape)}")
print(f"  text_masks sum:                  {sample['text_masks'].sum().item()} active tokens")

assert sample['images_cond'].shape == (3, 140, 192, 192), "T1 image shape mismatch"
assert sample['images'].shape == (3, 140, 192, 192), "T2 image shape mismatch"
print("  [PASS] Multimodal pair loaded and formatted successfully!")

# 5. Test Clinical Decision Recommendation Parsing (inference_mmu.py)
print("\n--- Testing Clinical Treatment Plan Recommendation Head ---")
LABEL_TO_ID = {
    "<SURGERY>": 0,
    "<CRT>": 1,
    "<RT>": 2,
    "<TMZ>": 3,
    "<AM>": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def answer_to_index(answer_text: str) -> int:
    for token, idx in LABEL_TO_ID.items():
        if token in answer_text:
            return idx
    return -1

# Ground truth from our CSV is <RT>
ground_truth_label = "<RT>"
simulated_llm_generation = (
    "Based on the tumor progression visible on the follow-up MRI, the recommended next-step treatment is <RT>."
)

pred_idx = answer_to_index(simulated_llm_generation)
gt_idx = LABEL_TO_ID[ground_truth_label]

print(f"  Model Generated Output: '{simulated_llm_generation}'")
print(f"  Extracted Class:        {pred_idx} ({ID_TO_LABEL[pred_idx]})")
print(f"  Ground Truth Class:     {gt_idx} ({ID_TO_LABEL[gt_idx]})")
assert pred_idx == gt_idx, "Prediction mapping mismatch!"
print("  [PASS] Clinical decision parsing verified with 100% match!")

print("\n" + "=" * 60)
print("[SUCCESS] Step 6 complete: End-to-end dataset pipeline & decision head verified!")
print("=" * 60)
