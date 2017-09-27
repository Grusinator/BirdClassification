from lib.utils.image_splitter_merger import image_splitter_merger
from PIL import Image
import numpy as np

ism = image_splitter_merger((250,250))

image_path = "cat.jpg"

image = Image.open(image_path)
# devide input image into suitable prediction sizes
subimg_list = ism.image_splitter(image)
k = 1
for subimg in subimg_list:
    outputpath = "test_output%d.jpg" %k
    k += 1
    # save image in output folder
    print('Saving results to: ', outputpath)
    with open(outputpath, 'wb') as out_file:
        subimg.save(out_file)

# merge the subsections
annotated_image = ism.image_merger(subimg_list)
# construct output path
outputpath = "test_output.tif"
# save image in output folder
print('Saving results to: ', outputpath)
with open(outputpath, 'wb') as out_file:
    annotated_image.save(out_file)


image_in = np.array(Image.open(image_path)).astype(np.float32)

image_out = np.array(Image.open(outputpath)).astype(np.float32)


ism._toPILImage(image_out)

diff = image_in - image_out
if diff.sum():
    print("equal!!")
else:
    print("not equal, test using tif?? dont compare jpgs, they are lossy")


