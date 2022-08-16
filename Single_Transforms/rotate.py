from PIL import Image

import os

target = "/home/frank/Coding/ML/wee/train"
source = "/home/frank/Coding/ML/data"
iter = ["/healthy", "/unhealthy"]
def rotateImages(angle):
    for i in iter:
        count = 0
        for j in os.listdir(source + i):
            count += 1
            img = Image.open(source + i + '/'+ j)
            img.rotate(angle).save(target + i + "/" + str(count) + "_" + str(angle) + ".JPG")
            img.close()

for i in range(360):
    rotateImages(i)




