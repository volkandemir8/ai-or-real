import os
from PIL import Image, ImageStat
from tqdm import tqdm

def find_corrupted_and_blank_images(dataset_root):
    problematic_images = []
    
    # Sadece resim formatlarını al
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    
    for dirpath, _, filenames in os.walk(dataset_root):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(dirpath, filename))
                
    print(f"Toplam {len(image_paths)} resim taranıyor...\n")
    
    for path in tqdm(image_paths, desc="Resimler kontrol ediliyor"):
        try:
            # 1. Dosya bütünlüğünü (Header vb.) doğrula
            with Image.open(path) as img:
                img.verify()
            
            # verify() resmi kapattığı için tekrar açıp piksel kontrolü yapıyoruz
            with Image.open(path) as img:
                img.load() # Veriyi RAM'e yükle (Bozukluk varsa burada patlar)
                
                # 2. Tamamen siyah veya tek renk (blank) resimleri tespit etme
                # Resmi gri tonlamaya çevirip en karanlık ve en parlak piksellerini (extrema) alıyoruz.
                extrema = img.convert("L").getextrema()
                
                # Eğer en karanlık piksel ile en aydınlık piksel aynıysa resim tek bir renkten oluşuyordur.
                if extrema[0] == extrema[1]:
                    color_desc = "Simsiyah" if extrema[0] == 0 else "Tamamen Düz Renk"
                    problematic_images.append((path, f"Görüntü Bozulmuş ({color_desc})"))
                    
        except Exception as e:
            problematic_images.append((path, f"Açılamayan Bozuk Dosya: {e}"))
            
    return problematic_images

if __name__ == "__main__":
    dataset_root = "train_dataset"
    bad_images = find_corrupted_and_blank_images(dataset_root)
    
    if not bad_images:
        print("\nVeri setinde bozuk veya simsiyah resim bulunamadı.")
    else:
        print(f"\nUYARI: {len(bad_images)} adet bozuk veya sorunlu resim tespit edildi!\n")
        for path, reason in bad_images:
            print(f"[{reason}] -> {path}")