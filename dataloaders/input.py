import torch
from PIL import Image
from torch.utils.data import Dataset
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast,RandomOrder, Hue, random_crop
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from torch.autograd import Variable
import numpy as np
import os

def custom(filename):

	IM_SCALE = 592

	image_unpadded = Image.open(filename)

	tform = [SquarePad(),Resize(IM_SCALE),ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]

	result =Compose(tform)

	#print (image_unpadded.siz
	a=result(image_unpadded)
	b=F.to_pil_image(a)
	c = Variable(a.view(-1,3,592,592))
	w, h = image_unpadded.size
	img_scale_factor = IM_SCALE/max(w,h)
	im_size = (IM_SCALE, int(w*img_scale_factor), img_scale_factor)
	return c, np.asarray([im_size])


def main():
	#custom()
	print (os.listdir('/home/saeid/KERN/data/input_images'))

if __name__ =="__main__":
	main()
