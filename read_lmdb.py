import string
import matplotlib.pyplot as plt
from strhub.data.dataset import LmdbDataset
import numpy as np
import skimage.io


types = ['SVTP', 'IC15_2077', 'IC13_1015', 'SVT', 'IIIT5k']
types = ['CUTE80']

for type in types:
    folder = f'/home/cindy/PycharmProjects/data/ocr/test/{type}'
    charset = string.digits + string.ascii_letters  # alphanumeric Latin character set
    dataset = LmdbDataset(folder, charset, max_label_len=25)

    num_samples = len(dataset)
    print(num_samples)

    for i in range(num_samples):
        img0, label0 = dataset[i]  # img0 is PIL.Image, label0 is str
        #
        # plt.imshow(img0)
        # plt.show()
    #     img0, label0 = dataset[i]
        img0 = np.array(img0)
        skimage.io.imsave(f'{folder}/{i:03d}.png', img0)

    # print(label0)
    # plt.imshow(img0)
    # plt.show()