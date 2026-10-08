#!/usr/bin/env bash
set -e

# Use current working directory / relative paths
DATA_DIR="./data/raw_images"
BUILD_DIR="./build"

echo "=================================================="
echo " [1/4] Cleaning Stale Downloads & Preparing Paths"
echo "=================================================="
mkdir -p "${DATA_DIR}"

# Remove invalid or corrupt files if any exist
rm -f ./landscape_dataset.zip*

echo "=================================================="
echo " [2/4] Downloading 1,000 Images (1280x720)"
echo "=================================================="
EXISTING_COUNT=$(find "${DATA_DIR}" -type f -name "*.jpg" | wc -l)

if [ "${EXISTING_COUNT}" -lt 800 ]; then
    python3 -c "
import os, urllib.request, concurrent.futures

data_dir = '${DATA_DIR}'
os.makedirs(data_dir, exist_ok=True)

def fetch(i):
    url = f'https://picsum.photos/id/{i}/1280/720'
    path = os.path.join(data_dir, f'img_{i:04d}.jpg')
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass

print('[INFO] Downloading missing images via 16 worker threads...')
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(fetch, range(1, 1001)))
print('[SUCCESS] Download complete!')
"
else
    echo "[INFO] Found ${EXISTING_COUNT} existing images in ${DATA_DIR}. Skipping download."
fi

echo "=================================================="
echo " [3/4] Building C++ OpenMP Pipeline"
echo "=================================================="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -G Ninja ..
ninja

echo "=================================================="
echo " [4/4] Executing Week 1 Pipeline"
echo "=================================================="
./pipeline_week1 "../${DATA_DIR}"

echo "=================================================="
echo " [SUCCESS] Week 1 Build & Test Run Complete!"
echo "=================================================="