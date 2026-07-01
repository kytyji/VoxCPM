# memory_test.py
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Всего VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Свободно VRAM: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

# Пробуем загрузить safetensors вручную
from safetensors.torch import load_file
print("Загрузка весов...")
try:
    state_dict = load_file("./pretrained_models/VoxCPM2/model.safetensors", device="cpu")
    print(f"✅ Загружено на CPU, ключей: {len(state_dict)}")
    total_size = sum(v.nbytes for v in state_dict.values()) / 1024**3
    print(f"Размер весов: {total_size:.2f} GB")
except Exception as e:
    import traceback
    traceback.print_exc()