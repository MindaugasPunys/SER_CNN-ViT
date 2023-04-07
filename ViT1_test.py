import numpy as np
from argparse import Namespace

import tensorflow as tf
# import tensorflow_datasets as tfds
from keras import layers
from keras import models

import keras_cv as kcv
from keras_cv.models import ViTTiny16
from keras_cv.layers import preprocessing

"""# Hyperparameters"""
configs = Namespace(
    learning_rate = 1e-4,
    batch_size = 64,
    num_epochs = 10,
    image_size = 224,
    num_classes = 120,
    num_steps = 1.0,
)

"""# Dataset and Dataloaders"""
AUTOTUNE = tf.data.AUTOTUNE


def parse_data(example):
    "Apply preprocessing to one data sample at a time."
    image = example["image"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (configs.image_size, configs.image_size))

    label = example["label"]
    label = tf.one_hot(label, configs.num_classes)

    return image, label


base_augmentations = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="base_augmentation",
)

mixup = preprocessing.MixUp(alpha=0.8)


def apply_base_augmentations(images, labels):
    images = base_augmentations(images)
    return images, labels


ds_train, ds_test = tfds.load('stanford_dogs', split=['train', 'test'])

trainloader = (
    ds_train
    .map(parse_data, num_parallel_calls=AUTOTUNE)
    .batch(configs.batch_size)
    .map(apply_base_augmentations, num_parallel_calls=AUTOTUNE)
    .map(lambda images, labels: mixup({"images": images, "labels": labels}), num_parallel_calls=AUTOTUNE)
    .map(lambda x: (x["images"], x["labels"]), num_parallel_calls=AUTOTUNE)
    .shuffle(1024)
    .prefetch(AUTOTUNE)
)

testloader = (
    ds_test
    .map(parse_data, num_parallel_calls=AUTOTUNE)
    .batch(configs.batch_size)
    .prefetch(AUTOTUNE)
)

def get_model():
    inputs = tf.keras.layers.Input(shape=(configs.image_size, configs.image_size, 3))

    vit = ViTTiny16(
        include_rescaling=False,
        include_top=False,
        name="ViTTiny32",
        weights="imagenet",
        input_tensor=inputs,
        pooling="token_pooling",
        activation=tf.keras.activations.gelu,
    )
    
    vit.trainable = True

    outputs = tf.keras.layers.Dense(configs.num_classes, activation="softmax")(vit.output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

model = get_model()
model.summary()

"""# Compile the Model

We will use `CosineDecay` learning rate scheduler.
"""

total_steps = len(trainloader)*configs.num_epochs
decay_steps = total_steps * configs.num_steps

cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    configs.learning_rate, decay_steps, alpha=0.1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay_scheduler),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

"""# [OPTIONAL] Model Prediction Visualization

We will build a custom Keras callback by subclassing `WandbEvalCallback` for model prediction visualization.
"""

class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validloader, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.val_data = validloader.unbatch().take(num_samples)

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            self.data_table.add_data(
                idx,
                wandb.Image(image),
                np.argmax(label, axis=-1)
            )

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred
            )

    def _inference(self):
      preds = []
      for image, label in self.val_data:
          pred = self.model(tf.expand_dims(image, axis=0))
          argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
          preds.append(argmax_pred)

      return preds

"""# Train the model with W&B"""

# Initialize a W&B run
run = wandb.init(
    project="keras_cv_vit",
    save_code=False,
    config=vars(configs),
)

# Fine-tune the model
model.fit(
    trainloader,
    epochs=configs.num_epochs,
    validation_data=testloader,
    callbacks=[
        WandbMetricsLogger(log_freq=2),
        WandbClfEvalCallback(
            validloader = testloader,
            data_table_columns = ["idx", "image", "label"],
            pred_table_columns = ["epoch", "idx", "image", "label", "pred"]
        )
    ],
)

"""# Model Evaluation"""

eval_loss, eval_acc = model.evaluate(testloader)
wandb.log({
    "eval_loss": eval_loss,
    "eval_acc": eval_acc
})

# Close the W&B run
wandb.finish()

