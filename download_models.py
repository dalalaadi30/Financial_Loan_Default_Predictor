import gdown
import os

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

# List of files to download
files = {
    "default_classifier.pkl": "1w87e_RxxGHW1x5eLYEk_jvr-3Pl-GmCL",
    "score_model.pkl": "1wrdxSypPK1XCzKenpFNMLQE1kfNJNpNK"
}

for filename, file_id in files.items():
    output_path = os.path.join("models", filename)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{filename} already exists, skipping.")