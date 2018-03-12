import numbers

import numpy as np
from PIL import Image
from torchvision.transforms import functional as transforms_f


class RandomSubset(object):
	"""Apply a random subset of transforms"""

	def __init__(self, transforms):
		self.transforms = transforms

	def get_subset(self):
		# draw how many transforms are to be applied
		nb = np.random.randint(0, len(self.transforms))
		# draw which transforms are applied
		subset = np.random.choice(self.transforms, nb, replace=False)
		return subset

	def __call__(self, img):
		for t in self.get_subset():
			img = t(img)
		return img


class RandomHorizontalFlipUnison(object):
	"""Apply a random horizontal flip on multiple images in unison"""

	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, imgs):
		if np.random.random() < self.p:
			for i in range(len(imgs)):
				imgs[i] = transforms_f.hflip(imgs[i])
		return imgs


class RandomVerticalFlipUnison(object):
	"""Apply a random horizontal flip on multiple images in unison"""

	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, imgs):
		if np.random.random() < self.p:
			for i in range(len(imgs)):
				imgs[i] = transforms_f.vflip(imgs[i])
		return imgs


class RandomResizedCropUnison(object):
	"""Apply a random resized crop on multiple images in unison"""

	# adapted from torchvision.transforms.transforms.RandomResizedCrop
	# using numpy.random instead of random because of better global seeding

	def __init__(self, size, scale, ratio, interpolation=Image.BILINEAR):
		self.size = (size, size)
		self.interpolation = interpolation
		self.scale = scale
		self.ratio = ratio

	@staticmethod
	def get_params(img, scale, ratio):
		for attempt in range(10):
			area = img.size[0] * img.size[1]
			target_area = np.random.uniform(*scale) * area
			aspect_ratio = np.random.uniform(*ratio)

			w = int(round(np.sqrt(target_area * aspect_ratio)))
			h = int(round(np.sqrt(target_area / aspect_ratio)))

			if np.random.random() < 0.5:
				w, h = h, w

			if w < img.size[0] and h < img.size[1]:
				i = np.random.randint(0, img.size[1] - h)
				j = np.random.randint(0, img.size[0] - w)
				return i, j, h, w

		# Fallback
		w = min(img.size[0], img.size[1])
		i = (img.size[1] - w) // 2
		j = (img.size[0] - w) // 2
		return i, j, w, w

	def __call__(self, imgs):
		# assume images are all the same size
		i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
		for i in range(len(imgs)):
			imgs[i] = transforms_f.resized_crop(imgs[i], i, j, h, w, self.size, self.interpolation)
		return imgs


class RandomRotationUnison(object):
	"""Apply a random rotation on multiple images in unison"""

	# adapted from torchvision.transforms.transforms.RandomRotation

	def __init__(self, degrees, resample=False, expand=False, center=None):
		if isinstance(degrees, numbers.Number):
			if degrees < 0:
				raise ValueError("If degrees is a single number, it must be positive.")
			self.degrees = (-degrees, degrees)
		else:
			if len(degrees) != 2:
				raise ValueError("If degrees is a sequence, it must be of len 2.")
			self.degrees = degrees

		self.resample = resample
		self.expand = expand
		self.center = center

	@staticmethod
	def get_params(degrees):
		return np.random.uniform(degrees[0], degrees[1])

	def __call__(self, imgs):
		angle = self.get_params(self.degrees)
		for i in range(len(imgs)):
			imgs[i] = transforms_f.rotate(imgs[i], angle, self.resample, self.expand, self.center)
		return imgs
