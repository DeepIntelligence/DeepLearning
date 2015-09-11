from Util import tile_raster_images
import numpy as np
try:
	import PIL.Image as Image
except ImportError:
	import Image


data = np.genfromtxt("../RBM/reconstruct.dat")


image = Image.fromarray(tile_raster_images(X=data, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
image.show()
image.save('reconstruct.png')


