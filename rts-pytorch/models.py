import torch
import torchvision
from torch import nn


def resnet_fc(resnet_fixed=True, cuda=False):
	model = torchvision.models.resnet50(pretrained=True)
	if resnet_fixed:
		for param in model.parameters():
			param.requires_grad = False
	# replace fully connected layer
	# use one output neuron per pixel in target image
	model.fc = nn.Linear(model.fc.in_features, 256 * 256)
	if cuda:
		model = model.cuda()
	return model, model.fc.parameters() if resnet_fixed else model.parameters()


def seg_net(num_classes, enc_fixed=True, no_out_act=False, cuda=False):
	model = SegNet(num_classes, no_out_act)
	if enc_fixed:
		for enc in [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
			for param in enc:
				param.requires_grad = False
		params = [p for dec in [model.dec5, model.dec4, model.dec3, model.dec2, model.dec1] for p in dec.parameters()]
	else:
		params = model.parameters()
	if cuda:
		model = model.cuda()
	return model, params


def duc(num_classes, cuda=False, no_out_act=False):
	model = ResNetDUC(num_classes, pretrained=True, no_out_act=no_out_act)
	if cuda:
		model = model.cuda()
	return model, model.parameters()


def duc_hdc(num_classes, cuda=False):
	model = ResNetDUCHDC(num_classes, pretrained=True)
	if cuda:
		model = model.cuda()
	return model, model.parameters()


class SegNetDecoderBlock(nn.Module):
	# adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation
	def __init__(self, in_channels, out_channels, num_conv_layers, no_act=False):
		super().__init__()
		middle_channels = in_channels // 2
		layers = [
			nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
			nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(middle_channels),
			nn.ReLU(inplace=True)
		]
		layers += [
					  nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
					  nn.BatchNorm2d(middle_channels),
					  nn.ReLU(inplace=True),
				  ] * (num_conv_layers - 2)
		layers += [
			nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
		]
		if not no_act:
			layers.append(nn.ReLU(inplace=True))
		self.decode = nn.Sequential(*layers)

	def forward(self, x):
		return self.decode(x)


class SegNet(nn.Module):
	# adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation
	def __init__(self, num_classes, no_out_act=False):
		super(SegNet, self).__init__()
		vgg = torchvision.models.vgg19_bn(pretrained=True)
		features = list(vgg.features.children())
		self.enc1 = nn.Sequential(*features[0:7])
		self.enc2 = nn.Sequential(*features[7:14])
		self.enc3 = nn.Sequential(*features[14:27])
		self.enc4 = nn.Sequential(*features[27:40])
		self.enc5 = nn.Sequential(*features[40:])

		self.dec5 = nn.Sequential(
			*([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
			  [nn.Conv2d(512, 512, kernel_size=3, padding=1),
			   nn.BatchNorm2d(512),
			   nn.ReLU(inplace=True)] * 4)
		)
		self.dec4 = SegNetDecoderBlock(1024, 256, 4)
		self.dec3 = SegNetDecoderBlock(512, 128, 4)
		self.dec2 = SegNetDecoderBlock(256, 64, 2)
		self.dec1 = SegNetDecoderBlock(128, num_classes, 2, no_out_act)
		initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

	def forward(self, x):
		enc1 = self.enc1(x)
		enc2 = self.enc2(enc1)
		enc3 = self.enc3(enc2)
		enc4 = self.enc4(enc3)
		enc5 = self.enc5(enc4)

		dec5 = self.dec5(enc5)
		dec4 = self.dec4(torch.cat([enc4, dec5], 1))
		dec3 = self.dec3(torch.cat([enc3, dec4], 1))
		dec2 = self.dec2(torch.cat([enc2, dec3], 1))
		dec1 = self.dec1(torch.cat([enc1, dec2], 1))
		return dec1


def initialize_weights(*models):
	# adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation
	for model in models:
		for module in model.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()


class _DenseUpsamplingConvModule(nn.Module):
	# adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation
	def __init__(self, down_factor, in_dim, num_classes, no_act=False):
		super(_DenseUpsamplingConvModule, self).__init__()
		self.no_act = no_act
		upsample_dim = (down_factor ** 2) * num_classes
		self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(upsample_dim)
		if not no_act:
			self.relu = nn.ReLU(inplace=True)
		self.pixel_shuffle = nn.PixelShuffle(down_factor)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		if not self.no_act:
			x = self.relu(x)
		x = self.pixel_shuffle(x)
		return x


class ResNetDUC(nn.Module):
	# adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation
	# the size of image should be multiple of 8
	def __init__(self, num_classes, pretrained=True, no_out_act=False):
		super(ResNetDUC, self).__init__()
		resnet = torchvision.models.resnet152(pretrained)
		self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

		for n, m in self.layer3.named_modules():
			if 'conv2' in n:
				m.dilation = (2, 2)
				m.padding = (2, 2)
				m.stride = (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)
		for n, m in self.layer4.named_modules():
			if 'conv2' in n:
				m.dilation = (4, 4)
				m.padding = (4, 4)
				m.stride = (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)

		self.duc = _DenseUpsamplingConvModule(8, 2048, num_classes, no_out_act)

	def forward(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.duc(x)
		return x


class ResNetDUCHDC(nn.Module):
	# adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation
	# the size of image should be multiple of 8
	def __init__(self, num_classes, pretrained=True):
		super(ResNetDUCHDC, self).__init__()
		resnet = torchvision.models.resnet152(pretrained)
		self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

		for n, m in self.layer3.named_modules():
			if 'conv2' in n or 'downsample.0' in n:
				m.stride = (1, 1)
		for n, m in self.layer4.named_modules():
			if 'conv2' in n or 'downsample.0' in n:
				m.stride = (1, 1)
		layer3_group_config = [1, 2, 5, 9]
		for idx in range(len(self.layer3)):
			self.layer3[idx].conv2.dilation = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
			self.layer3[idx].conv2.padding = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
		layer4_group_config = [5, 9, 17]
		for idx in range(len(self.layer4)):
			self.layer4[idx].conv2.dilation = (layer4_group_config[idx], layer4_group_config[idx])
			self.layer4[idx].conv2.padding = (layer4_group_config[idx], layer4_group_config[idx])

		self.duc = _DenseUpsamplingConvModule(8, 2048, num_classes)

	def forward(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.duc(x)
		return x
