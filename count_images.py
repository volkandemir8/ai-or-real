import os

# Kontrol edilecek ana klasor
dataset_root = "train_dataset"

splits = ["train", "validation", "test"]
classes = ["real", "fake"]

total_images = 0

print("--- Veri Seti İstatistikleri ---")

for split in splits:
    for c in classes:
        folder_path = os.path.join(dataset_root, split, c)
        if os.path.exists(folder_path):
            # Sadece resim dosyalarini say (gizli sistem dosyalarini atlamak icin)
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            count = len(images)
            total_images += count
            print(f"{split} {c} {count}")
        else:
            print(f"{split} {c} 0 (Klasör bulunamadı)")

print("--------------------------------")
print(f"Toplam Resim: {total_images}")