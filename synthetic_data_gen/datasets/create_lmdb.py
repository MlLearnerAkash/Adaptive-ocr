import os
import lmdb 
import cv2
import numpy as np
import argparse

def checkImageIsValid(imageBin):
    if imageBin is None:
        print("empty imageBin")
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    length1 = len(imageBuf)
    if length1 > 0:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
    else:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                txn.put(str(k).encode(), v)
            else:
                txn.put(str(k).encode(), str(v).encode())

def createDataset(outputPath, combinedFile, checkValid=True, size=1.0):
    # Read the combined file, where each line is "image_path<TAB>label"
    with open(combinedFile, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    
    nSamples = int(len(lines) * size)
    print("Total samples to process:", nSamples)
    
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    for i in range(nSamples):
        # Each line should contain two parts separated by a tab.
        parts = lines[i].strip().split("\t")
        if len(parts) < 2:
            print("Malformed line, skipping:", lines[i])
            continue

        imagePath = parts[0].strip()
        label = parts[1].strip()

        if not os.path.exists(imagePath):
            print(f"{imagePath} does not exist")
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print(f"{imagePath} is not a valid image")
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label

        if cnt % 50000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f"Written {cnt} / {nSamples}")

        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print(f"Created dataset with {nSamples} samples")
    return nSamples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputPath', help='Path to output LMDB directory', required=True)
    parser.add_argument('--combinedFile', help='Path to txt file with image path and label separated by tab', required=True)
    parser.add_argument('--size', type=float, default=1.0, help='Fraction of samples to use (0 to 1)')
    args = parser.parse_args()

    output_path = args.outputPath
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    combinedFile = args.combinedFile
    createDataset(output_path, combinedFile, size=args.size)
