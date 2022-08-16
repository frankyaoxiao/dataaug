import imageio as iio
import imgaug.augmenters as ia
import os
from pathlib import Path

target = "/home/frank/Coding/ML/wee/train"
source = "/home/frank/Coding/ML/data"
iter = ["/healthy", "/unhealthy"]
def ChannelShuffle():
    for i in iter:
        count = 0
        for file in Path(source + "/" + i).iterdir():
            img = iio.imread(file)
            count += 1
            aug = ia.ChannelShuffle(1)
            imgnew = aug.augment_image(img)
            iio.imwrite(target + '/' + i + '/' + str(count) + "_channel_shift" + '.JPG', imgnew)
            iio.imwrite(target + '/' + i + '/' + str(count) + '.JPG', img)

ChannelShuffle()







