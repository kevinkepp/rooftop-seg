import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import functional as transforms_f

import random_trans as transforms_r


class RoofDataset(Dataset):
	img_mean = [0.28439793, 0.2864871, 0.25555506]
	img_std = [0.19537579, 0.18462174, 0.17142102]

	def __init__(self, data_dir, mode="train", random_trans=False, normalize=True, resize=None, limit=None,
				 no_labels=False):
		self.images = self.load_images(Path(f"{data_dir}/images/{mode}"), mode="RGB")
		self.labels = self.load_images(Path(f"{data_dir}/labels/{mode}"), mode="L") if not no_labels else None
		if limit:
			self.images, self.labels = self.images[:limit], self.labels[:limit]
		self.normalize = normalize
		self.resize = resize
		self.random_trans = transforms_r.RandomSubset([
			transforms_r.RandomHorizontalFlipUnison(),
			transforms_r.RandomVerticalFlipUnison(),
			transforms_r.RandomResizedCropUnison(256, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
			transforms_r.RandomRotationUnison(180)
		]) if random_trans else None

	@staticmethod
	def load_images(dir_: Path, mode=None):
		paths = sorted([p for p in dir_.iterdir() if p.is_file()])
		images = [open_image(p, mode) for p in paths]
		return images

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx] if self.labels else None
		if self.resize:
			image = transforms_f.resize(image, self.resize)
		if self.random_trans:
			image, label = self.random_trans([image, label])
		image = transforms_f.to_tensor(image)
		if self.normalize:
			image = transforms_f.normalize(image, self.img_mean, self.img_std)
		if label:
			label = transforms_f.to_tensor(label)
			return image, label
		else:
			return image


def open_image(path, mode=None):
	i = Image.open(path, 'r')
	i.load()
	if mode and i.mode != mode:
		i = i.convert(mode)
	return i


def get_mean_std():
	dataset = RoofDataset(".", mode="train", random_trans=False, normalize=False, resize=False, limit=None)
	nb_samples = len(dataset)
	loader = DataLoader(dataset, batch_size=nb_samples)
	images, _ = next(iter(loader))
	means = np.mean(images.numpy().transpose(0, 2, 3, 1), axis=(0, 1, 2))
	stds = np.std(images.numpy().transpose(0, 2, 3, 1), axis=(0, 1, 2))
	return means, stds


if __name__ == "__main__":
	# calculate mean and std for the training data
	mean, std = get_mean_std()
	print(f"Traing data mean: {mean.tolist()}, std: {std.tolist()}")
