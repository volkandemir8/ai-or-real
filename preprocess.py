import os
import random
from PIL import Image, ImageFile
from tqdm import tqdm

# Bozuk veya yarim kalmis (truncated) resim dosyalarinin hata vermesini engeller, yuklemeye calisir
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Cok yuksek cozunurluklu resimler icin (DecompressionBomb) piksel sinirini kaldirir
Image.MAX_IMAGE_PIXELS = None

# INPUT DATASET
# Klasor yapisi: datasets/ai-generated-images-vs-real-images -> train/test -> real/fake
input_root = "datasets/ai-generated-images-vs-real-images"

# OUTPUT DATASET
# Cikti yapisi: train_dataset -> train/validation/test -> real/fake
output_root = "train_dataset"

# IMAGE SIZE
IMG_SIZE = 256

# VALIDATION SPLIT RATIO
VAL_SPLIT_RATIO = 0.25 # Orijinal train verisinin %25'i validation'a ayrilacak

def process_single_image(img_input_path, output_path, img_name):
        try:
            img = Image.open(img_input_path)

            # RGB yap
            img = img.convert("RGB")

            # resize
            img = img.resize((IMG_SIZE,IMG_SIZE), Image.Resampling.LANCZOS) # kalite kaybını en aza indiren, LANCZOS algoritması

            # yeni isim
            new_name = os.path.splitext(img_name)[0] + ".png"

            img_output_path = os.path.join(output_path, new_name)

            # Kayıpsız (lossless) PNG olarak kaydet
            img.save(img_output_path, "PNG")

        except Exception as e:
            print("Error:", img_input_path, e)


def main():
    classes = ["real", "fake"]

    for c in classes:
        print(f"\n--- Sınıf İşleniyor: {c.upper()} ---")
        
        # --- 1. TEST SETİNİ İŞLEME ---
        test_input_dir = os.path.join(input_root, "test", c)
        test_output_dir = os.path.join(output_root, "test", c)
        os.makedirs(test_output_dir, exist_ok=True)
        
        if os.path.exists(test_input_dir):
            test_images = [f for f in os.listdir(test_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            for img_name in tqdm(test_images, desc=f"Test Seti ({c})"):
                img_input_path = os.path.join(test_input_dir, img_name)
                process_single_image(img_input_path, test_output_dir, img_name)
        else:
            print(f"Uyarı: Test klasörü bulunamadı -> {test_input_dir}")

        # --- 2. TRAIN SETİNİ İŞLEME VE VALIDATION'A BÖLME ---
        train_input_dir = os.path.join(input_root, "train", c)
        train_output_dir = os.path.join(output_root, "train", c)
        val_output_dir = os.path.join(output_root, "validation", c)
        
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)
        
        if os.path.exists(train_input_dir):
            train_images = [f for f in os.listdir(train_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            random.seed(42)
            random.shuffle(train_images)
            
            val_size = int(len(train_images) * VAL_SPLIT_RATIO)
            val_images = train_images[:val_size]
            train_images_final = train_images[val_size:]
            
            for img_name in tqdm(train_images_final, desc=f"Train Seti ({c})"):
                img_input_path = os.path.join(train_input_dir, img_name)
                process_single_image(img_input_path, train_output_dir, img_name)
                
            for img_name in tqdm(val_images, desc=f"Validation Seti ({c})"):
                img_input_path = os.path.join(train_input_dir, img_name)
                process_single_image(img_input_path, val_output_dir, img_name)
        else:
            print(f"Uyarı: Train klasörü bulunamadı -> {train_input_dir}")


if __name__ == "__main__":
    main()