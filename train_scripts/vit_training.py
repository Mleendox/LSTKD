import sys
from data.CIFAR_X.setup import setup_data, aug_images
from transformers import TFViTForImageClassification , ViTImageProcessor
import tensorflow as tf
import argparse
import wandb

def __init_accelerators__():
    
    # Avoid Out-Of-Memory Error by limiting GPU memory consumption growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--dataset_id", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    args, _ = parser.parse_known_args()

    # init data
    D_train, D_val, D_test, k, h, w, c, id2label, label2id = setup_data(args.batch_size, args.dataset_id)

    # init vit weights directory
    if args.model_id == "google/vit-base-patch16-224-in21k":
        model_dir_name = "google_vit_base"
    elif args.model_id == "google/vit-large-patch16-224-in21k":
        model_dir_name = "google_vit_large"
    elif args.model_id == "google/vit-huge-patch16-224-in21k":
        model_dir_name = "google_vit_huge"
    output_dir = "models/ViTs/{}/{}/".format(model_dir_name, args.dataset_id)

    # init ViT model 
    vit = TFViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=args.model_id,
        num_labels=k,
        id2label=id2label,
        label2id=label2id
    )

    # compile model
    vit.compile(
        optimizer = tf.keras.optimizers.AdamW(learning_rate=args.learning_rate),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [
            tf.keras.metrics.SparseTopKCategoricalAccuracy(1, name="Top1CategoricalAccuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="Top5CategoricalAccuracy"),
        ]
    )

    # Initialize wandb
    if args.model_id == "google/vit-base-patch16-224-in21k": project_id = "google-vit-base-16-224-in21k"
    elif args.model_id == "google/vit-large-patch16-224-in21k": project_id = "google-vit-large-16-224-in21k"
    elif args.model_id == "google/vit-huge-patch16-224-in21k": project_id = "google-vit-huge-16-224-in21k"
    run = wandb.init(project=project_id, dir=output_dir, config={
            "learning_rate": args.learning_rate,
            "architecture": "ViTForImageClassification",
            "dataset": args.dataset_id,
            "epochs": args.epochs,
            "optimizer": "AdamW"
        }
    )

    vip = ViTImageProcessor.from_pretrained(args.model_id)
    # define the training loop
    for epoch in range(args.epochs):
        print("E P O C H  {}/{}".format((epoch+1), args.epochs))
        # apply gradients
        D_train = D_train.shuffle(D_train.cardinality())
        for batch in D_train:
            # augment images
            images = aug_images(batch[0])
            # compute the loss and gradients for this batch of data
            with tf.GradientTape() as tape:
                images = tf.convert_to_tensor(vip(images)["pixel_values"])
                logits = vit(images, training=True)["logits"]
                loss_value = vit.compiled_loss(batch[1], logits)
            # get the gradients of the trainable variables with respect to the loss
            grads = tape.gradient(loss_value, vit.trainable_variables)
            # update the weights of the model using the optimizer
            vit.optimizer.apply_gradients(zip(grads, vit.trainable_variables))
            # update the metrics
            vit.compiled_metrics.update_state(batch[1], logits)
        # show and log training metrcis
        for metric in vit.metrics:
            metric_name = metric.name
            metric_value = metric.result().numpy()
            print(f"    training_{metric_name}: {metric_value}")
            wandb.log({f"training_{metric_name}": metric_value})
        # reset the metrics for the next epoch
        vit.reset_metrics()

        # Calculate validation loss at end of epoch
        val_loss = 0.0
        D_val = D_val.shuffle(D_val.cardinality())
        for batch in D_val:
            images = tf.convert_to_tensor(vip(batch[0])["pixel_values"])
            logits = vit(images, training=False)["logits"]
            val_loss += vit.compiled_loss(batch[1], logits)
            # update the metrics
            vit.compiled_metrics.update_state(batch[1], logits)
        val_loss /= D_val.cardinality().numpy()
        val_loss = val_loss.numpy()
        # Log validation loss to wandb
        wandb.log({"val_loss": val_loss})
        print(f"    val_loss: {val_loss}")
        # show and log val metrics
        for metric in vit.metrics:
            metric_name = metric.name
            if metric_name == "loss": continue
            metric_value = metric.result().numpy()
            print(f"    val_{metric_name}: {metric_value}")
            # Log validation metrics to wandb
            wandb.log({f"val_{metric_name}": metric_value})
        # reset the metrics for the next epoch
        vit.reset_metrics()
    
    # define the evaluation loop
    eval_loss = 0.0
    for batch in D_test:
        images = tf.convert_to_tensor(vip(batch[0])["pixel_values"])
        logits = vit(images, training=False)["logits"]
        eval_loss += vit.compiled_loss(batch[1], logits)
        # update the metrics
        vit.compiled_metrics.update_state(batch[1], logits)
    eval_loss /= D_test.cardinality().numpy()
    eval_loss = eval_loss.numpy()
    # Log evaluation loss to wandb
    print()
    wandb.log({'eval_loss': eval_loss})
    print(f'    eval_loss: {eval_loss}')
    # show and log evaluation metrics
    for metric in vit.metrics:
        metric_name = metric.name
        if metric_name == "loss": continue
        metric_value = metric.result().numpy()
        print(f"    eval_{metric_name}: {metric_value}")
        # Log validation metrics to wandb
        wandb.log({f"eval_{metric_name}": metric_value})
    # reset the metrics for the next epoch
    vit.reset_metrics()
    
    # vit.save(filepath="{}model/model.keras".format(output_dir))
    vit.save_weights(filepath="{}weights/".format(output_dir))
