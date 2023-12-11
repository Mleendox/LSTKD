import re
import sys
import argparse
from data.CIFAR_X.setup import setup_data, aug_images
from model_definitions.resnet import ResNet
import tensorflow as tf
import wandb


def __init_accelerators__():
    
    # Avoid Out-Of-Memory Error by limiting GPU memory consumption growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    strategy = tf.distribute.MirroredStrategy()
    print("Num GPUs Available: {}".format(len(tf.config.list_physical_devices("GPU"))))

    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')

    return strategy


if __name__ == "__main__":

    strategy = __init_accelerators__()

    parser = argparse.ArgumentParser()
    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--architecture", type=str, default="")
    parser.add_argument("--dataset_id", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    # get args
    args, _ = parser.parse_known_args()

    # build data pipelines
    D_train, D_val, D_test, k, h, w, c, _, _ = setup_data(batch_size=args.batch_size, dataset_id=args.dataset_id)

    # init output directory
    architecture_dir_map = {
        "ResNet18": "resnet18",
        "ResNet50": "resnet50s",
        "ResNet152": "resnet152",
    }

    student_model_dir = architecture_dir_map.get(args.architecture)

    if "ResNet" in args.architecture:
        # Extract the numbers at the end of the string
        numbers = re.findall(r"\d+$", args.architecture)
        if numbers:
            # resnet_variant = "101" if numbers[0]=="152" else numbers[0]
            resnet_variant = numbers[0]
    
    output_dir = "models/ConvNets/{}/{}/".format(student_model_dir, args.dataset_id)

    # model creation
    preprocessing = tf.keras.Sequential(
        layers = [
            tf.keras.layers.Resizing(height=h, width=w, interpolation="bicubic"),
            tf.keras.layers.Rescaling(scale=(1.0/255.0))
        ],
        name = "preprocessing"
    )

    if args.architecture == "ResNet18" or args.architecture == "ResNet50" or args.architecture == "ResNet152":
        core_model = ResNet(resnet_variant)

    l2_reg = tf.keras.regularizers.l2(l2=0.000025)
    classification_head = tf.keras.Sequential(
        layers = [
            tf.keras.layers.Dense(1024, kernel_initializer="lecun_normal", activation="selu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, kernel_initializer="lecun_normal", activation="selu", kernel_regularizer=l2_reg),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, kernel_initializer="lecun_normal", activation="selu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(k)
        ],
        name = "classification_head"
    )

    model = tf.keras.Sequential(
        layers=[
            preprocessing,
            core_model,
            classification_head
        ],
        name="model"
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=args.learning_rate) ,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseTopKCategoricalAccuracy(1, name='Top1CategoricalAccuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='Top5CategoricalAccuracy'),
        ]
    )

    # start a new wandb run to track this script
    project_id = "ResNetX" if args.architecture.startswith("ResNet") else args.architecture
    run = wandb.init(project=project_id, dir=output_dir, config={
            "learning_rate": args.learning_rate,
            "architecture": args.architecture,
            "dataset": args.dataset_id,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "l2_reg": 0.000025,
            "input_shape": "(224, 224, 3)",
            "rescaling_and_resizing": "keras preprocessing layers"}
    )
    
    # define the training loop
    for epoch in range(args.epochs):
        print("E P O C H  {}/{}".format((epoch+1), args.epochs))
        D_train = D_train.shuffle(D_train.cardinality())
        for batch in D_train:
            # compute the loss and gradients for this batch of data
            with tf.GradientTape() as tape:
                # augment images
                images = aug_images(batch[0])
                logits = model(images, training=True)
                loss_value = model.compiled_loss(batch[1], logits)
            # get the gradients of the trainable variables with respect to the loss
            grads = tape.gradient(loss_value, model.trainable_variables)
            # update the weights of the model using the optimizer
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # update the metrics
            model.compiled_metrics.update_state(batch[1], logits)
        # show and log training metrcis
        for metric in model.metrics:
            metric_name = metric.name
            metric_value = metric.result().numpy()
            print(f"    training_{metric_name}: {metric_value}")
            wandb.log({f"training_{metric_name}": metric_value})
        # reset the metrics for the next epoch
        model.reset_metrics()
        # Calculate validation loss at end of epoch
        val_loss = 0.0
        D_val = D_val.shuffle(D_val.cardinality())
        for batch in D_val:
            logits = model(batch[0], training=False)
            val_loss += model.compiled_loss(batch[1], logits)
            # update the metrics
            model.compiled_metrics.update_state(batch[1], logits)
        val_loss /= D_val.cardinality().numpy()
        val_loss = val_loss.numpy()
        # Log validation loss to wandb
        wandb.log({"val_loss": val_loss})
        print(f"    val_loss: {val_loss}")
        # show and log val metrics
        for metric in model.metrics:
            metric_name = metric.name
            if metric_name == "loss": continue
            metric_value = metric.result().numpy()
            print(f"    val_{metric_name}: {metric_value}")
            # Log validation metrics to wandb
            wandb.log({f"val_{metric_name}": metric_value})
        # reset the metrics for the next epoch
        model.reset_metrics()
        
    # define the evaluation loop
    eval_loss = 0.0
    for batch in D_test:
        logits = model(batch[0], training=False)
        eval_loss += model.compiled_loss(batch[1], logits)
        # update the metrics
        model.compiled_metrics.update_state(batch[1], logits)
    eval_loss /= D_test.cardinality().numpy()
    eval_loss = eval_loss.numpy()
    # Log evaluation loss to wandb
    print()
    wandb.log({"eval_loss": eval_loss})
    print("eval_loss: {}".format(eval_loss))
    # show and log evaluation metrics
    for metric in model.metrics:
        metric_name = metric.name
        if metric_name == "loss": continue
        metric_value = metric.result().numpy()
        print(f"eval_{metric_name}: {metric_value}")
        # Log validation metrics to wandb
        wandb.log({"eval_{}".format(metric_name): metric_value})
    # reset the metrics for the next epoch
    model.reset_metrics()

    # model.save(filepath="{}model/model.keras".format(output_dir))
    model.save_weights(filepath="{}weights/".format(output_dir))
