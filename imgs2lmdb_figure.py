#!/usr/bin/env python3
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
# todo: change eval_step in strhub > models > base.py

import argparse
import io
import os
import string

import lmdb
import numpy as np
from PIL import Image

from strhub.data.dataset import LmdbDataset


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    img = Image.open(io.BytesIO(imageBin)).convert('RGB')
    return np.prod(img.size) > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, outputPath, checkValid=True, dataset_name=None):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1

    print(inputPath)

    gtFile = sorted([f for f in os.listdir(inputPath) if '.png' in f])

    folder = f'/home/cindy/PycharmProjects/data/ocr/test/{args.dataset}'

    print(folder)
    charset = string.digits + string.ascii_letters  # alphanumeric Latin character set
    dataset = LmdbDataset(folder, charset, max_label_len=25)

    nSamples = len(gtFile)
    for i, line in enumerate(gtFile):
        imagePath = gtFile[i]

        num = int(imagePath.split('.')[0])
        img, label = dataset[num]
        imagePath = os.path.join(inputPath, imagePath)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                img = Image.open(io.BytesIO(imageBin)).convert('RGB')
            except IOError as e:
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('{}-th image data occured error: {}, {}\n'.format(i, imagePath, e))
                continue
            if np.prod(img.size) == 0:
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--bright', type=float, default=0.6)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--ours', action='store_true')
    parser.add_argument('--ldm', action='store_true')
    parser.add_argument('--input', action='store_true')
    parser.add_argument('--make_input', action='store_true')

    args = parser.parse_args()

    gt_set = 'IC13_1015' if 'ic' in args.dataset else args.dataset
    print(f'gt set: {gt_set}')


    if args.ours:
        createDataset(
            inputPath=f'/home/cindy/Documents/did_paper/wordqual/{args.dataset}_v{args.bright}_n{args.noise}/ours',
            outputPath=f'/home/cindy/Documents/did_paper/wordqual/test/{gt_set}',
        )

    elif args.input:
        createDataset(
            inputPath=f'/home/cindy/Documents/did_paper/wordqual/{args.dataset}_v{args.bright}_n{args.noise}/input',
            outputPath=f'/home/cindy/Documents/did_paper/wordqual/test/{gt_set}',
        )
    elif args.ldm:
        createDataset(
            inputPath=f'/home/cindy/Documents/did_paper/wordqual/{args.dataset}_v{args.bright}_n{args.noise}/ldm',
            outputPath=f'/home/cindy/Documents/did_paper/wordqual/test/{gt_set}',
        )
    else:
        createDataset(
            inputPath=f'/home/cindy/Documents/did_paper/wordqual/{args.dataset}_v{args.bright}_n{args.noise}/llflow',
            outputPath=f'/home/cindy/Documents/did_paper/wordqual/test/{gt_set}',
        )




# read lmdb -> save as padded images
# take images -> run through networks
# save outputs of networks to lmdb