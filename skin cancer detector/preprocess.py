import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw/"
IMG_DIR1 = os.path.join(RAW_DIR, "HAM10000_images_part_1")
IMG_DIR2 = os.path.join(RAW_DIR, "HAM10000_images_part_2")
META_PATH = os.path.join(RAW_DIR, "HAM10000_metadata.csv")

metadata = pd.read_csv(META_PATH)

all_files = []

# Match image paths with labels
for idx, row in metadata.iterrows():
    file = row['image_id'] + ".jpg"
    label = row['dx']

    if os.path.exists(os.path.join(IMG_DIR1, file)):
        full_path = os.path.join(IMG_DIR1, file)
    else:
        full_path = os.path.join(IMG_DIR2, file)

    all_files.append((full_path, label))

df = pd.DataFrame(all_files, columns=["path", "label"])

# Split dataset stratified
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

def copy_files(subset_df, subset_name):
    for _, row in subset_df.iterrows():
        label = row['label']
        dst_path = os.path.join("data", subset_name, label)
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy(row['path'], dst_path)

copy_files(train_df, "train")
copy_files(val_df, "val")
copy_files(test_df, "test")

print("✔ Dataset prepared successfully!")
print("Train / Val / Test folders are ready.")
