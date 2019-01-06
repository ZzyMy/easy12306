# coding: utf-8
import shutil
import os
import pathlib

import numpy as np

import mlearn
from pretreatment import load_data


def learn():
    texts, imgs = load_data()
    labels = mlearn.predict(texts)
    labels = labels.argmax(axis=1)
    imgs.dtype = np.uint64
    imgs.shape = (-1, 8)
    unique_imgs = np.unique(imgs)
    print(unique_imgs.shape)
    imgs_labels = []
    for img in unique_imgs:
        idxs = np.where(imgs == img)[0]
        u, counts = np.unique(labels[idxs], return_counts=True)
        counts = np.bincount(labels[idxs], minlength=80)
        imgs_labels.append(counts)
    np.savez('images.npz', images=unique_imgs, labels=imgs_labels)


def main():
    pathlib.Path('result_images').mkdir(exist_ok=True)
    data = np.load('images.npz')
    images, labels = data['imgs'], data['labels']
    labels = labels.argmax(axis=1)
    images = list(images)
    path = 'fingerprints'
    for idx, fn in enumerate(os.listdir(path)):
        fp = int(fn[:16], base=16)
        try:
            idx = images.index(fp)
            label = labels[idx]
        except:
            label = 80
        shutil.copyfile(os.path.join(path, fn),
                        os.path.join('result_images', f'{label}.{idx}.jpg'))


if __name__ == '__main__':
    # learn()
    main()
