from datetime import datetime
import os
from pathlib import Path
import tensorflow as tf
import plac


def main(
        # (help, kind, abbrev, type, choices, metavar)
        data_dir: ("Input data directory", "positional", None, Path) = Path("data/"),
        log_dir: ("Logs directory", "positional", None, Path) = Path("logs/"),
        batch_size: ("Batch size", "option", "bs", int) = 5,
        epochs: ("Epochs", "option", "ep", int) = 100,
        cpu: ("CPU only", "flag", "cpu") = False,
):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import data
    import unet

    # tf.config.experimental_run_functions_eagerly(True)

    train, nb_train = data.get_dataset(data_dir, "train", batch_size)
    val, nb_val = data.get_dataset(data_dir, "val", batch_size)
    # test, nb_test = data.get_test_dataset(data_dir)

    model = unet.get_model(input_shape=(224, 224, 3), output_channels=1, output_act=None)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    log_dir = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)]

    history = model.fit(train,
                        epochs=epochs,
                        steps_per_epoch=nb_train // batch_size,
                        validation_data=val,
                        validation_steps=nb_val // batch_size,
                        callbacks=callbacks)


if __name__ == "__main__":
    plac.call(main)
