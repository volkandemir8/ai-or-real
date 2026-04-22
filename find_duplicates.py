import os
import hashlib
from collections import defaultdict
from tqdm import tqdm

def calculate_md5(filepath, chunk_size=8192):
    """Bir dosyanın MD5 hash değerini verimli bir şekilde hesaplar."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
    except OSError:
        # Dosya okunamıyorsa (örn. izin hatası), boş bir hash döndür
        return None
    return hash_md5.hexdigest()

def find_duplicate_images(root_folder):
    """
    Verilen klasördeki tüm resimleri tarar ve içeriği aynı olanları bulur.
    """
    hashes = defaultdict(list)
    image_paths = []

    # 1. Adım: Tüm resim dosyalarının yollarını topla
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(dirpath, filename))

    # 2. Adım: Her resmin hash'ini hesapla ve sözlükte grupla
    for path in tqdm(image_paths, desc="Resimler taranıyor"):
        file_hash = calculate_md5(path)
        if file_hash:
            hashes[file_hash].append(path)

    # 3. Adım: Birden fazla kopyası olanları filtrele
    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    return duplicates

if __name__ == "__main__":
    dataset_root = "train_dataset"
    duplicate_files = find_duplicate_images(dataset_root)

    if not duplicate_files:
        print("\nVeri setinde birbirinin kopyası olan hiçbir resim bulunamadı.")
    else:
        total_duplicate_count = sum(len(v) for v in duplicate_files.values())
        print(f"\nUYARI: {len(duplicate_files)} farklı resmin birden fazla kopyası bulundu (Toplam {total_duplicate_count} dosya).")
        print("--------------------------------------------------")
        for i, (file_hash, file_list) in enumerate(duplicate_files.items(), 1):
            print(f"Grup {i}:")
            for filepath in file_list:
                print(f"  -> {filepath}")
            print("-" * 20)