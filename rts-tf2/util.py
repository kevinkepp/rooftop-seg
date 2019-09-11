import matplotlib.pyplot as plt
import tensorflow as tf
import PIL


class DisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self, data, model, intv=1):
        self.data = data
        self.model = model
        self.intv = intv

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.intv == 0:
            show_predictions(self.data, self.model)
        # track LR
        logs.update({"lr": self.model.optimizer.lr.numpy()})


def show_predictions(data, model, num=1):
    nb_shown = 0
    ds_iter = iter(data)
    display_list = []
    while nb_shown < num:
        batch = next(ds_iter)
        images, masks = batch if isinstance(batch, (list, tuple)) else (batch, None)
        preds = model.predict(images)
        pred_masks = tf.round(tf.math.sigmoid(preds))
        i = 0
        while nb_shown < num and i < len(images):
            if masks is not None:
                dl = [images[i], masks[i], pred_masks[i]]  # create_mask
            else:
                dl = [images[i], pred_masks[i]]  # create_mask
            display_list.append(dl)
            i += 1
            nb_shown += 1
    show_images(display_list)


def show_images(display_list):
    rows, cols = len(display_list), len(display_list[0])
    plt.figure(figsize=(5 * cols, 3 * rows))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c + 1
            plt.subplot(rows, cols, i)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[r][c]))
            plt.axis('off')
    plt.show()


def show_model_history(history, nb_epochs, show_lr=False):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    lr = history.history["lr"]
    epochs = range(nb_epochs)
    fig, ax1 = plt.subplots()
    plt.title('Training and Validation Loss')
    ax2 = ax1.twinx()
    l1 = ax1.plot(epochs, loss, 'r', label='Training loss')
    l2 = ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss value')
    # plt.ylim([0, 1])
    lines = [l1[0], l2[0]]
    if show_lr:
        l3 = ax2.plot(epochs, lr, label="Learning rate")
        ax2.set_ylabel("Learning rate")
        lines.append(l3[0])
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels)
    plt.show()
