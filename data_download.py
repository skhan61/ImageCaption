import os

import requests
from tqdm import tqdm


def download_file(url, folder):
    local_filename = folder + '/' + url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    return local_filename


# Folder to save the data
data_folder = "/media/sayem/510B93E12554BBD1/CocoData"


os.makedirs(data_folder, exist_ok=True)

# COCO dataset URLs
urls = [
    # 2017 Train images [18GB]
    "http://images.cocodataset.org/zips/train2017.zip",
    # 2017 Val images [1GB]
    "http://images.cocodataset.org/zips/val2017.zip",
    # 2017 Test images [6GB]
    "http://images.cocodataset.org/zips/test2017.zip",
    # 2017 Train/Val annotations [241MB]
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]

# Download each file
for url in urls:
    download_file(url, data_folder)
for url in urls:
    download_file(url, data_folder)
