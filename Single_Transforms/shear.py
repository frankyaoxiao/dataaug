from wand.image import Image

import os

target = "/home/frank/Coding/ML/wee/train"
source = "/home/frank/Coding/ML/data"
iter = ["/healthy", "/unhealthy"]
def rotateImages(angle):
    for i in iter:
        count = 0
        for j in os.listdir(source + i):
            with Image(filename = (source + '/' + i + '/' + j)) as image:
                with image.clone() as shear:
                    shear.shear()


