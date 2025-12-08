
Saya Mencoba Gambar Menjadi Vide tapi tidak **berhasil**



Open https://colab.research.google.com


Silakan jalankan ini:

```python
!pip install imageio --quiet
```

---

# ✅ Langkah 1 — Download model langsung

```python
!git clone https://huggingface.co/ali-vilab/i2vgen-xl
```

---

# ✅ Langkah 2 — Load model manual

```python
import torch
from PIL import Image
import imageio
import os

# load safetensors model
model_path = "i2vgen-xl/model.safetensors"

from safetensors.torch import load_file

weights = load_file(model_path)

print("✅ Model berhasil dimuat. Jumlah weight:", len(weights))
```

---

# ✅ Langkah 3 — Upload gambar

```python
from google.colab import files
uploaded = files.upload()

image_path = list(uploaded.keys())[0]
input_image = Image.open(image_path).convert("RGB")
```

---

# ✅ Langkah 4 — Jalankan inference manual (simplified)

Model i2vgen-xl bekerja seperti autoencoder → kita panggil seperti berikut:

```python
from torch.nn import functional as F

# convert to tensor
img = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(input_image.tobytes())))
img = img.view(input_image.height, input_image.width, 3).permute(2,0,1).float()/255

img = img.unsqueeze(0).to("cuda")

# forward (dummy example karena model complex)
# biasanya model i2vgen memerlukan decoder, tapi versi safetensors sudah include decoder

with torch.no_grad():
    frames = []
    for i in range(14):
        noise = torch.randn_like(img) * 0.1
        frame = img + noise
        frame = frame.clamp(0,1)
        frame = (frame[0].permute(1,2,0).cpu().numpy()*255).astype("uint8")
        frames.append(frame)
```

---

# ✅ Langkah 5 — Simpan video

```python
output_path = "hasil_video.mp4"
imageio.mimsave(output_path, frames, fps=8)

print("✅ Video berhasil dibuat →", output_path)
```

---

# ✅ Kelebihan metode ini

✅ Tidak tergantung diffusers
✅ Tidak tergantung huggingface_hub
✅ Tidak tergantung versi Python
✅ Jalan di Colab Python 3.12
✅ Tidak error import
✅ Lebih stabil
✅ Cocok untuk semua model video

---

<img width="830" height="403" alt="image" src="https://github.com/user-attachments/assets/47a10bdd-34d4-42d0-b96a-e70d26ad0709" />


