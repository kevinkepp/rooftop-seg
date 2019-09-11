import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as transforms_f

from data import RoofDataset


def tensor_denormalize(tensor, mean, std):
	return (tensor * torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
			+ torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)) \
		.clamp(0, 1)


def plot_train_preds(inputs, outputs, preds, labels, denormalize=True):
	import matplotlib.pyplot as plt

	fig, axs = plt.subplots(inputs.size(0), 4, squeeze=False)
	for r, imgs in enumerate(zip(inputs, outputs, preds, labels)):
		imgs = [im.data for im in imgs]
		if denormalize:
			imgs[0] = tensor_denormalize(imgs[0], RoofDataset.img_mean, RoofDataset.img_std)
		imgs = [im.numpy().transpose(1, 2, 0) for im in imgs]
		imgs = [im[:, :, 0] if im.shape[2] == 1 else im for im in imgs]
		for c, img in enumerate(imgs):
			axs[r][c].imshow(img, vmin=0, vmax=1)
			axs[r][c].set_axis_off()
	for i, title in enumerate(["Input", "Output", "Pred", "Label"]):
		axs[0][i].set_title(title)
	fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
	fig.show()


def save_train_preds(dir_, name, inputs, outputs, preds, labels, denormalize=True):
	image_rows = []
	for imgs in zip(inputs, outputs, preds, labels):
		# input_ = tensor_to_image(imgs[0].data, mean=RoofDataset.img_mean, std=RoofDataset.img_std, clip=True) \
		#	if denormalize else tensor_to_image(imgs[0].data)
		# imgs = [input_] + [tensor_to_image(im.data) for im in imgs[1:]]
		# images = [Image.fromarray(img, mode) for img, mode in zip(imgs, ["RGB", "L", "L", "L"])]
		imgs = [im.data for im in imgs]
		if denormalize:
			imgs[0] = tensor_denormalize(imgs[0], RoofDataset.img_mean, RoofDataset.img_std)
		images = [transforms_f.to_pil_image(img, mode) for img, mode in zip(imgs, ["RGB", "L", "L", "L"])]
		image_rows.append(hstack_images(images))
	im = vstack_images(image_rows)
	path = Path(f"{dir_}/{name}.jpg")
	im.save(path)


def save_test_preds(dir_, preds):
	dir_ = Path(dir_)
	for i, im in enumerate(preds):
		im = transforms_f.to_pil_image(im.data, mode="L")
		im.save(dir_ / f"{i}.jpg")


def hstack_images(images):
	widths, heights = zip(*(i.size for i in images))
	stack = Image.new("RGB", (sum(widths), max(heights)))
	x_offset = 0
	for im in images:
		stack.paste(im, (x_offset, 0))
		x_offset += im.size[0]
	return stack


def vstack_images(images):
	widths, heights = zip(*(i.size for i in images))
	stack = Image.new("RGB", (max(widths), sum(heights)))
	y_offset = 0
	for im in images:
		stack.paste(im, (0, y_offset))
		y_offset += im.size[1]
	return stack
