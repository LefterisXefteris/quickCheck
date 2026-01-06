import base64
from io import BytesIO
import zlib
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
from torchvision import transforms

class FootballDataset:
    def __init__(self, data_dir="../football"):
        data_dir = Path(data_dir)
        root_dir_data = [x for x in data_dir.iterdir() if x.is_dir()]
        image_dir = root_dir_data[0]
        annotations_dir = root_dir_data[1]

        
        self.images = sorted(image_dir.glob("*.png"))
        self.annotations = sorted(annotations_dir.glob("*.json"))
        self.transform = transforms.Compose([
                            transforms.Resize((720, 1280)),  
                            transforms.ToTensor(),
])
    
    def __len__(self):
        return len(self.images)
    
  
    # 1. Access ann['objects'][0]['bitmap']['data'] (the base64 string)
    # 2. Access ann['objects'][0]['bitmap']['origin'] (the x, y position)
    # 3. base64.b64decode(data) → gives compressed bytes
    # 4. zlib.decompress(compressed) → gives raw image bytes
    # 5. Load as image: Image.open(BytesIO(decompressed))
    # 6. Convert to numpy array
    # 7. Create full-size mask using origin to position it correctly
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        with open(self.annotations[idx], 'r') as f:
            annotations = json.load(f)

        access_mask = annotations['objects'][0]['bitmap']['data']
        origin = annotations['objects'][0]['bitmap']['origin']
        x, y = origin[0], origin[1]
        compressed = base64.b64decode(access_mask)
        decompressed = zlib.decompress(compressed)
        small_mask = Image.open(BytesIO(decompressed))
        small_mask = np.array(small_mask)

        h, w = annotations['size']['height'], annotations['size']['width']
        full_mask = np.zeros((h, w), dtype=np.uint8)
        mask_h, mask_w = small_mask.shape[:2]
        full_mask[y:y+mask_h, x:x+mask_w] = small_mask if small_mask.ndim == 2 else small_mask[:,:,0]

        full_mask = torch.tensor(full_mask, dtype=torch.float32)
        return image, full_mask
    
    def visualize(self, idx):
        image, mask = self[idx]
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image_np)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask_np, cmap='gray')
        plt.axis('off')

        plt.show()
    


if __name__ == "__main__":

    dataset = FootballDataset(data_dir="../football")
    dataset.visualize(20)

    


 