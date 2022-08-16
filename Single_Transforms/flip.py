from PIL import Image, ImageOps

import os

target = "/home/frank/Coding/ML/wee/train"
source = "/home/frank/Coding/ML/data"
iter = ["/healthy", "/unhealthy"]
def flipImages():
    for i in iter:
        count = 0
        for j in os.listdir(source + i):
            count += 1
            img = Image.open(source + i + '/'+ j)
            im_flip = ImageOps.flip(img)
            im_mirror = ImageOps.mirror(img)
            im_flip.save(target + '/' + i + '/' + str(count) + "_flipped" + ".JPG")
            im_mirror.save(target + '/' + i + '/' + str(count) + "_mirrored" + ".JPG")
            img.save(target + '/' + i + '/' + str(count) + ".JPG")
            img.close()





