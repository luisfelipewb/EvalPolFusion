"""
Split dataset into train, validation, and test sets based on the number of masks
in each experiment. The key idea is to ensure that the distribution of masks
prevents similar (consecutive) masks from being in the same set.
The list of masks is read from a text file. It does not shuffle the masks
but rather counts how many masks are in each experiment and then splits them
proportionally to the total size of the set.
"""


input_file_path = "./selected_masks_new_crop.txt"

# Define sizes
train_size = 1000
val_size = 200
test_size = 300
total_size = train_size + val_size + test_size

with open(input_file_path, 'r') as f:
    all_masks = [line.strip() for line in f if line.strip()]

print(f"Total masks: {len(all_masks)}")


# Count how many starts with exp01, exp02, etc.
exp_counts = {}
for mask in all_masks:
    if mask.startswith("exp"):
        exp_name = mask.split("_")[0]
        if exp_name not in exp_counts:
            exp_counts[exp_name] = 0
        exp_counts[exp_name] += 1

print(f"Experiment counts: {exp_counts}")

# Shuffle
# random.shuffle(all_masks)

train_masks = []
val_masks = []
test_masks = []
for i, exp_name in enumerate(exp_counts):
    n = exp_counts[exp_name]
    # Make it proportional to the total size
    if i < len(exp_counts.keys()) - 1:
        train_n = int(round(train_size * n / total_size))
        val_n = int(round(val_size * n / total_size))
        test_n = int(round(test_size * n / total_size))
        print(f"Experiment {exp_name}: {n} masks, Train: {train_n}, Val: {val_n}, Test: {test_n}")

        train_masks.extend([all_masks.pop(0) for _ in range(train_n)])
        val_masks.extend([all_masks.pop(0) for _ in range(val_n)])
        test_masks.extend([all_masks.pop(0) for _ in range(test_n)])
        print(f"Remaining masks: {len(all_masks)}")
    # In the last experiment, fill to reach round numbers
    else:
        train_n = train_size - len(train_masks)
        val_n = val_size - len(val_masks)
        test_n = test_size - len(test_masks)
        print(f"Experiment {exp_name}: {n} masks, Train: {train_n}, Val: {val_n}, Test: {test_n}")

        train_masks.extend([all_masks.pop(0) for _ in range(train_n)])
        val_masks.extend([all_masks.pop(0) for _ in range(val_n)])
        test_masks.extend([all_masks.pop(0) for _ in range(test_n)])
        print(f"Remaining masks: {len(all_masks)}")


print(f"Train masks: {len(train_masks)}")
print(f"Val masks: {len(val_masks)}")
print(f"Test masks: {len(test_masks)}")

# Shuffle
# random.suffle(train_masks)
# random.suffle(val_masks)
# random.suffle(test_masks)

# Save the sets to files
with open("train.txt", "w") as f:
    for mask in train_masks:
        f.write(f"{mask}\n")
with open("val.txt", "w") as f:
    for mask in val_masks:
        f.write(f"{mask}\n")
with open("test.txt", "w") as f:
    for mask in test_masks:
        f.write(f"{mask}\n")
