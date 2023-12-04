import tensorflow as tf
import tensorflow_datasets as tfds

@tf.function
def kdst_aug(images):
    # randomly select an augmentation technique
    choice = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    # apply the selected augmentation technique
    images = tf.case(
        [
            (tf.equal(choice, 0), lambda: tf.image.random_crop(value=images, size=[images.shape[0], 30, 30, 3])),
            (tf.equal(choice, 1), lambda: tf.image.random_flip_left_right(images)),
            (tf.equal(choice, 2), lambda: tf.image.random_flip_up_down(images))
        ],
        default=lambda: images)
    return images

@tf.function
def aug_images(images):
    # randomly select an augmentation technique
    choice = tf.random.uniform(shape=[], minval=0, maxval=7, dtype=tf.int32)
    # apply the selected augmentation technique
    images = tf.case([
        (tf.equal(choice, 0), lambda: tf.image.random_brightness(images, max_delta=0.3)),
        (tf.equal(choice, 1), lambda: tf.image.random_contrast(images, lower=0.8, upper=1.2)),
        (tf.equal(choice, 2), lambda: tf.image.random_saturation(images, lower=0.8, upper=1.2)),
        (tf.equal(choice, 3), lambda: tf.image.random_hue(images, max_delta=0.2)),
        (tf.equal(choice, 4), lambda: tf.image.random_flip_left_right(images)),
        (tf.equal(choice, 5), lambda: tf.image.random_flip_up_down(images)),
    ], default=lambda: images)
    return images


def setup_data(batch_size=None, dataset_id="cifar10"):
    
    CIFAR_X, CIFAR_X_info = tfds.load(
        dataset_id,
        split=["train", "test"],
        as_supervised=True,
        with_info=True)

    k = CIFAR_X_info.features["label"].num_classes
    h, w, c = CIFAR_X_info.features["image"].shape
    h = 224
    w = 224
    img_class_labels = CIFAR_X_info.features["label"].names
    id2label = {str(i): label for i, label in enumerate(img_class_labels)}
    label2id = {v: k for k, v in id2label.items()}

    D_train, D_test = CIFAR_X

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 32 if batch_size == None else batch_size

    # Build training pipeline
    D_train = D_train.shuffle(D_train.cardinality())
    D_train = D_train.batch(BATCH_SIZE)
    D_train = D_train.cache()
    D_train = D_train.prefetch(AUTOTUNE)

    # Build D_test
    D_test = D_test.shuffle(D_test.cardinality())
    D_test = D_test.batch(BATCH_SIZE)

    # Build D_val
    p = int(0.2*D_test.cardinality().numpy())
    D_val = D_test.take(p)
    D_test = D_test.skip(p)
    # Build evaluation pipeline
    D_val = D_val.cache()
    D_val = D_val.prefetch(AUTOTUNE)
    # Build testing pipeline
    D_test = D_test.prefetch(AUTOTUNE)

    return D_train, D_val, D_test, k, h, w, c, id2label, label2id
