import argparse
import copy
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import RoofDataset
from models import seg_net
from viz import plot_train_preds, save_train_preds, save_test_preds

parser = argparse.ArgumentParser(description='Deep roof segmentation')
parser.add_argument('--data-dir', default='data', help="Default: 'data'")
parser.add_argument('--out-dir', default='out', help="Default: 'out'")
parser.add_argument('--fixed', action="store_true")
parser.add_argument('--workers', default=1, type=int, help="Default: 1")
parser.add_argument('-e', '--epochs', default=50, type=int, help="Default: 50")
parser.add_argument('-bs', '--batch-size', default=5, type=int, help="Default: 5")
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, help="Default: 0.01")
parser.add_argument('-pat', '--lr-patience', default=10, type=int,
					help="How many epochs to wait until reducing LR when validation loss plateauing. "
						 "No LR schedule when set to 0. Default: 3")
parser.add_argument('-fac', '--lr-factor', default=0.1, type=float,
					help="Factor by which the LR is reduced by the scheduler. Default: 0.1")
parser.add_argument('-m', '--momentum', default=0.9, type=float, help="Default: 0.9")
parser.add_argument('-wd', '--weight-decay', default=0, type=float, help="Default: 0")
parser.add_argument('-s', '--seed', default=None, type=int, help="Default: None")
parser.add_argument('--no-gpu', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--no-normalize', action='store_true')
parser.add_argument('--no-random-trans', action='store_true')
parser.add_argument('--overfit', action='store_true',
					help="Uses only one sample from the training data and no validation.")
parser.add_argument('--limit', type=int, default=None,
					help="Limit the number of smaples to be used for training in each epoch. Default: None")


def main():
	print("General setup...")
	args = parser.parse_args()
	cuda = not args.no_gpu and torch.cuda.is_available()
	if args.seed:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if cuda:
			torch.cuda.manual_seed(args.seed)
	out_dir = Path(args.out_dir)
	tmp_dir = out_dir / "tmp"
	if tmp_dir.exists():
		shutil.rmtree(tmp_dir)
	tmp_dir.mkdir()
	model_dir = out_dir / "models"
	model_dir.mkdir(exist_ok=True)
	test_dir = out_dir / "test"
	test_dir.mkdir(exist_ok=True)

	print("Preparing model...")
	model, params = seg_net(1, enc_fixed=args.fixed, no_out_act=True, cuda=cuda)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	scheduler = lr_scheduler.ReduceLROnPlateau(
		optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True) \
		if args.lr_patience > 0 and 1 > args.lr_factor > 0 else None

	print("Loading data...")
	normalize = not args.no_normalize
	random_trans = not args.overfit and not args.no_random_trans
	shuffle = not args.overfit
	limit = args.limit if not args.overfit else 1
	batch_size = args.batch_size if not limit or args.batch_size <= limit else limit
	train_loader = DataLoader(
		dataset=RoofDataset(args.data_dir, mode="train", random_trans=random_trans, normalize=normalize, limit=limit),
		batch_size=batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=cuda)
	val_loader = DataLoader(
		dataset=RoofDataset(args.data_dir, mode="val", random_trans=False, normalize=normalize),
		batch_size=batch_size, num_workers=args.workers, pin_memory=cuda)
	test_loader = DataLoader(
		dataset=RoofDataset(args.data_dir, mode="test", random_trans=False, normalize=normalize, no_labels=True),
		batch_size=batch_size, num_workers=args.workers, pin_memory=cuda)
	phases = ["train"]
	dataloaders = {"train": train_loader, "test": test_loader}
	if not args.overfit:
		phases.append("val")
		dataloaders["val"] = val_loader

	print("Training and evaluating...")
	t_start = time.time()
	it_durs = []  # iteration durations
	val_losses, val_accs = [], []
	best_acc, best_weights = -1, None
	for epoch in range(args.epochs):
		print(f"Epoch {epoch + 1}/{args.epochs}\n{'-' * 10}")

		for phase in phases:
			model.train(phase == "train")

			# loop through data batches
			for i, (inputs, labels) in enumerate(dataloaders[phase]):
				t_it_start = time.time()
				inputs, labels = Variable(inputs), Variable(labels)
				if cuda:
					inputs, labels = inputs.cuda(), labels.cuda()
				# predict
				optimizer.zero_grad()
				logits = model(inputs)
				loss = criterion(logits, labels)
				outputs = torch.sigmoid(logits)
				preds = torch.round(outputs)
				# eval
				nb_wrong = (labels - preds).nonzero().size(0)
				nb_total = labels.size(0) * labels.size(1) * labels.size(2) * labels.size(3)
				acc = (1 - nb_wrong / nb_total) * 100
				if phase == "train":
					print(f"{phase} loss: {loss.data[0]:.4f}, acc: {acc:5.2f}%")
				elif phase == "val":
					val_losses.append(loss.data[0])
					val_accs.append(acc)
				# learn
				loss.backward()
				optimizer.step()
				t_it_dur = time.time() - t_it_start
				if phase == "train":
					it_durs.append(t_it_dur)  # only count training iterations

				images = inputs, outputs, preds, labels
				if cuda:
					images = [i.cpu() for i in images]
				if args.plot and phase == "train":
					plot_train_preds(*images, denormalize=normalize)
				save_train_preds(tmp_dir, f"{epoch}_{phase}_{i}", *images, denormalize=normalize)

			if phase == "val":
				val_loss, val_acc = np.mean(val_losses), np.mean(val_accs)
				its_per_sec = 1 / np.mean(it_durs)
				print(f"{phase} loss: {val_loss:.4f}, acc: {val_acc:5.2f}%, it/s: {its_per_sec:.2f}")
				if scheduler:
					scheduler.step(val_loss)
				if val_acc > best_acc:
					best_acc = val_acc
					best_weights = copy.deepcopy(model.state_dict())
					print("new best acc!")
				val_losses, val_accs = [], []

	t_dur = time.time() - t_start
	print(f"Training complete in {t_dur // 60:.0f}m {t_dur % 60:.0f}s")
	print(f"Best model val acc: {best_acc:5.2f}%")
	model.load_state_dict(best_weights)

	print("Storing model...")
	torch.save(model.state_dict(), model_dir / "best.pkl")

	print("Predicting test labels...")
	for inputs in dataloaders["test"]:
		inputs = Variable(inputs)
		if cuda:
			inputs = inputs.cuda()
		preds = torch.round(torch.sigmoid(model(inputs)))
		save_test_preds(test_dir, preds.cpu())

	print("Done")


if __name__ == "__main__":
	main()
