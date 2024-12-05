# Stable Diffusion Fine-Tuning for Image-to-Image Generation
This repository demonstrates how to fine-tune a pre-trained **Stable Diffusion model** for tasks like **denoising** and **image enhancement.** The pipeline includes data preparation, fine-tuning, and testing steps for creating high-quality image-to-image generation models.

## Table of Contents
1. Introduction
2. Features
3. Setup
4. Dataset Preparation
5. Fine-Tuning
6. Testing & Visualization
7. Acknowledgments

## Introduction
This project explores fine-tuning a pre-trained Stable Diffusion model to generate enhanced images from noisy input data. The workflow includes creating paired noisy-clean datasets, training the model with custom loss functions, and visualizing the results.

## Features
- **Data Augmentation:** Generate paired noisy-clean datasets using Gaussian noise.
- **Model Fine-Tuning:** Fine-tune the Stable Diffusion model with textual conditioning.
- **Testing and Visualization:** Evaluate the model on test data and visualize the outputs.

## Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stable-diffusion-finetuning.git
cd stable-diffusion-finetuning
```
### 2. Install Dependencies
if you run this on **google colab**, you don't need to download any dependencies but make sure you run on `GPU` in **google colab.**

### 3. Download Pre-Trained Models
Ensure you have access to:

- `runwayml/stable-diffusion-v1-5` from Hugging Face.
- CLIP tokenizer and text encoder for textual conditioning.

## Dataset Preparation
### 1. Generate Paired Data
The following script creates a noisy-clean dataset using Gaussian noise:
```python
import os
import numpy as np
from PIL import Image
import random

def add_noise(image, noise_level=25):
    """
    Adds Gaussian noise to an image.
    Args:
        image (PIL.Image): Input image.
        noise_level (int): Standard deviation of the Gaussian noise.
    Returns:
        PIL.Image: Noisy image.
    """
    np_image = np.array(image).astype(np.float32)
    noise = np.random.normal(0, noise_level, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Generate dataset
os.makedirs('./paired_data/input', exist_ok=True)
os.makedirs('./paired_data/target', exist_ok=True)

data_dir = './clean_images'
for i, file_name in enumerate(os.listdir(data_dir)):
    image_path = os.path.join(data_dir, file_name)
    image = Image.open(image_path).convert("RGB")
    noisy_image = add_noise(image, noise_level=random.randint(15, 50))
    noisy_image.save(f'./paired_data/input/noisy_{i}.jpg')
    image.save(f'./paired_data/target/clean_{i}.jpg')

print("Paired noisy-clean dataset created!")
```
### 2. Load Dataset
Define a custom PyTorch dataset to load paired noisy-clean data:
```python
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        return input_image, target_image

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

input_dir = './paired_data/input'
target_dir = './paired_data/target'
dataset = PairedImageDataset(input_dir, target_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

## Fine-Tuning
### 1. Load Pre-Trained Model
```python
from diffusers import StableDiffusionImg2ImgPipeline

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

print("Stable Diffusion model loaded successfully!")
```
### 2. Fine-Tune the Model
```python
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

def fine_tune_model(dataloader, pipeline, epochs=12, accumulation_steps=2):
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=5e-6)
    scaler = GradScaler()

    for epoch in range(epochs):
        total_loss = 0
        for step, (input_images, target_images) in enumerate(dataloader):
            input_images = input_images.to("cuda", dtype=torch.float16) * 2 - 1
            target_images = target_images.to("cuda", dtype=torch.float16) * 2 - 1
            latents_input = pipeline.vae.encode(input_images).latent_dist.sample() * 0.18215
            text_inputs = tokenizer(["a portrait"] * len(input_images), padding="max_length", max_length=77, return_tensors="pt").to("cuda")
            encoder_hidden_states = text_encoder(text_inputs.input_ids)[0].to(dtype=torch.float16)

            noise = torch.randn_like(latents_input)
            timesteps = torch.randint(0, 1000, (latents_input.size(0),), device=latents_input.device).long()
            noisy_latents = pipeline.scheduler.add_noise(latents_input, noise, timesteps)

            with autocast(device_type="cuda"):
                noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise) / accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
```
## Testing & Visualization
Test the model on noisy test images and visualize the outputs:
```python
from PIL import Image
import matplotlib.pyplot as plt

def test_model(pipeline, test_images, prompt, strength=0.8):
    results = []
    for path in test_images:
        image = Image.open(path).convert("RGB").resize((512, 512))
        result = pipeline(prompt=prompt, image=image, strength=strength, guidance_scale=7.5).images[0]
        results.append(result)
    return results

test_images = ["./paired_data/input/noisy_0.jpg", "./paired_data/input/noisy_1.jpg"]
outputs = test_model(pipeline, test_images, prompt="a portrait of a young man")

fig, axes = plt.subplots(1, len(outputs), figsize=(15, 5))
for ax, img in zip(axes, outputs):
    ax.imshow(img)
    ax.axis("off")
plt.show()
```
## Acknowledgments
- Hugging Face for the `diffusers` library.
- OpenAI for the CLIP tokenizer and text encoder.

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`feature/my-feature`).
3. Commit your changes and push.
4. Submit a pull request.

## Contact
For any questions or feedback, please contact:

- **Name:** Shoaib Hoque
- **Email:** shoaibhoque@gmail.com
- **LinkedIn:** [Shoaib Hoque](https://www.linkedin.com/in/shoaib-hoque-2bb20314b/)
- **GitHub:** [ShoaibHoque](https://github.com/ShoaibHoque)
