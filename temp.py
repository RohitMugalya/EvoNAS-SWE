import os
from torchvision import datasets

# Output directory
output_root = "dataset/mnist"
os.makedirs(output_root, exist_ok=True)

# Download MNIST (returns PIL Images by default)
mnist = datasets.MNIST(
    root="dataset",
    train=True,
    download=True
)

# Create class folders
for i in range(10):
    os.makedirs(os.path.join(output_root, str(i)), exist_ok=True)

# Save images
for idx, (image, label) in enumerate(mnist):
    image_path = os.path.join(output_root, str(label), f"{idx}.png")
    image.save(image_path)

print("MNIST saved successfully at:", os.path.abspath(output_root))
