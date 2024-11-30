import shutil
import os
from pathlib import Path
import requests
import tarfile

def download_and_extract(url, extract_path):
    os.makedirs(extract_path, exist_ok=True)
    tar_path = extract_path + ".tar"

    # Force delete existing directory
    if Path(extract_path).exists():
        print(f"Removing existing directory: {extract_path}")
        shutil.rmtree(extract_path)

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)
    os.remove(tar_path)
    print(f"Downloaded and extracted to {extract_path}")

def ensure_paddleocr_models():
    base_dir = str(Path.home() / ".paddleocr/whl")
    models = {
        "det": ("https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar", f"{base_dir}/det/en/en_PP-OCRv3_det_infer"),
        "rec": ("https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar", f"{base_dir}/rec/en/en_PP-OCRv4_rec_infer"),
        "cls": ("https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar", f"{base_dir}/cls/ch_ppocr_mobile_v2.0_cls_infer"),
    }

    for model, (url, path) in models.items():
        download_and_extract(url, path)

# Ensure models are ready
ensure_paddleocr_models()
