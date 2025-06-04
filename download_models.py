import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    """Download LeanVAE pretrained models"""
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs (these would be the actual URLs from the paper/repository)
    models = {
        "LeanVAE-chn4.ckpt": "https://example.com/LeanVAE-chn4.ckpt",  # Replace with actual URL
        "LeanVAE-chn16.ckpt": "https://example.com/LeanVAE-chn16.ckpt"  # Replace with actual URL
    }
    
    print("Note: This script contains placeholder URLs.")
    print("Please check the official LeanVAE repository for actual model download links.")
    print("Common sources for pretrained models:")
    print("- Hugging Face Hub")
    print("- Google Drive links in the repository")
    print("- Direct download links in the README")
    
    for model_name, url in models.items():
        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists, skipping...")
            continue
            
        print(f"Downloading {model_name}...")
        try:
            download_url(url, model_path)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            print(f"Please manually download from the official repository and place in {models_dir}/")

if __name__ == "__main__":
    main()
