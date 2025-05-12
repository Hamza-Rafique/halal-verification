from PIL import Image
import torchvision.transforms as transforms

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Must match training size
        transforms.ToTensor()
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)
