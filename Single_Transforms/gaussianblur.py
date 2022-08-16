import imageio as iio
import imgaug.augmenters as ia
import os
from pathlib import Path

target = "/home/frank/Coding/ML/kek/bruh"
source = "/home/frank/Coding/ML/kek"
iter = ["/healthy", "/unhealthy"]
def GaussianBlur():
    for i in iter:
        count = 0
        for file in Path(source + "/" + i).iterdir():
            img = iio.imread(file)
            count += 1
            aug = ia.GaussianBlur()
            imgnew = aug.augment_image(img)
            iio.imwrite(target + '/' + i + '/' + str(count) + "_gaussian_blur" + '.JPG', imgnew)
            iio.imwrite(target + '/' + i + '/' + str(count) + '.JPG', img)
            aug = ia.Sharpen(alpha=(0,1.0), lightness=(.75, 1.5))
            imgnew = aug.augment_image(img)
            iio.imwrite(target + '/' + i + '/' + str(count) + "_sharpen" + '.JPG', imgnew)
            aug = ia.AdditiveGaussianNoise(loc=0, scale=(0.0,0.05*225), per_channel=0.5)
            iio.imwrite(target + '/' + i + '/' + str(count) + "_AdditiveNoise" + '.JPG', imgnew)


GaussianBlur();






