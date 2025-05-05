import kagglehub
import os

# Download latest version to current directory
path = kagglehub.dataset_download("rupanshukapoor/harry-potter-books")

print("Path to dataset files:", path)