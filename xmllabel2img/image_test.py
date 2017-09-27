from skimage.data import camera
from skimage import filters
camera = camera()
print(type(camera))
val = filters.threshold_otsu(camera)
mask = camera < val
