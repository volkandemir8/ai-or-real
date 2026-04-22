import torch

def check_cuda():
    print("--- GPU / CUDA Kontrolü ---")
    if torch.cuda.is_available():
        print("✅ CUDA Kullanılabilir durumda! Sistem GPU kullanmaya hazır.")
        print(f"PyTorch Sürümü: {torch.__version__}")
        print(f"CUDA Sürümü: {torch.version.cuda}")
        print(f"Mevcut GPU Sayısı: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA bulunamadı. İşlemler CPU üzerinden yürütülecek.")
        print("Eğer bir NVIDIA GPU'nuz varsa, uygun CUDA sürücülerini ve PyTorch'un CUDA destekli versiyonunu yüklediğinizden emin olun.")
    print("---------------------------")

if __name__ == "__main__":
    check_cuda()