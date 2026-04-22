import os
import random
from PIL import Image, ImageFile
from tqdm import tqdm

# Bozuk/yarim kalmis ve yuksek cozunurluklu resim hatalarini engellemek icin
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# INPUT DATASET CLASORLERI
input_root = "datasets"
dataset_folders = [
    "fake-real-images-1", 
    "fake-real-images-2", 
    "fake-real-images-3", 
    "fake-real-images-4"
]

# OUTPUT DATASET
output_root = "train_dataset1"
IMG_SIZE = 256

# BÖLME ORANLARI (SPLIT RATIOS)
VAL_RATIO = 0.15 # 1, 2 ve 3. veri setlerinin %15'i validation, %85'i train olacak.

def process_and_split_multiple_datasets():
    classes = ["real", "fake"]
    
    random.seed(42)
    
    for ds_folder in dataset_folders:
        for cls in classes:
            input_path = os.path.join(input_root, ds_folder, cls)
            
            if not os.path.exists(input_path):
                print(f"Uyarı: Klasör bulunamadı, atlanıyor -> {input_path}")
                continue
                
            images = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if ds_folder == "fake-real-images-4":
                # 4. klasördeki tüm resimleri SADECE TEST seti olarak ayır
                splits = {"test": images}
            else:
                # 1, 2 ve 3. klasörlerdeki resimleri karıştır ve Train/Validation olarak böl
                random.shuffle(images)
                val_size = int(len(images) * VAL_RATIO)
                splits = {
                    "validation": images[:val_size],
                    "train": images[val_size:]
                }
            
            for split_name, img_list in splits.items():
                output_path = os.path.join(output_root, split_name, cls)
                os.makedirs(output_path, exist_ok=True)
                
                for img_name in tqdm(img_list, desc=f"Processing {ds_folder}/{cls} -> {split_name}"):
                    img_input_path = os.path.join(input_path, img_name)
                    
                    try:
                        img = Image.open(img_input_path).convert("RGB")
                        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                        
                        # İsim çakışmasını engellemek için başına klasör adını ekliyoruz
                        new_name = f"{ds_folder}_{cls}_{os.path.splitext(img_name)[0]}.jpg"
                        
                        # Sabit kalite yerine 90-100 arası rastgele yüksek kalite
                        img.save(os.path.join(output_path, new_name), "JPEG", quality=random.randint(90, 100))
                    except Exception as e:
                        print("Hata:", img_input_path, e)

if __name__ == "__main__":
    process_and_split_multiple_datasets()