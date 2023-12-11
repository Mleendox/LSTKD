import re
import sys
from data.CIFAR_X.setup import setup_data, kdst_aug
from model_definitions.resnet import ResNet
from transformers import ViTImageProcessor, TFViTForImageClassification
import tensorflow as tf
import wandb
import argparse

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

class DIST(tf.keras.losses.Loss):

    def __init__(self, beta=1.0, gamma=1.0, tau=1.0, **kwargs):
        super(DIST, self).__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    @tf.function
    def cosine_similarity(self, a, b, eps=1e-8):
        return tf.reduce_sum(a * b, axis=1) / (tf.norm(a, axis=1) * tf.norm(b, axis=1) + eps)

    @tf.function
    def pearson_correlation(self, a, b, eps=1e-8):
        return self.cosine_similarity(a - tf.reduce_mean(a, axis=1, keepdims=True),
                                      b - tf.reduce_mean(b, axis=1, keepdims=True), eps)

    @tf.function
    def inter_class_relation(self, y_s, y_t):
        return 1 - tf.reduce_mean(self.pearson_correlation(y_s, y_t))

    @tf.function
    def intra_class_relation(self, y_s, y_t):
        return self.inter_class_relation(tf.transpose(y_s), tf.transpose(y_t))

    @tf.function
    def call(self, y_true, y_pred):
            y_s = tf.nn.softmax(y_pred / self.tau, axis=1)
            y_t = tf.nn.softmax(y_true / self.tau, axis=1)
            inter_loss = self.tau**2 * self.inter_class_relation(y_s, y_t)
            intra_loss = self.tau**2 * self.intra_class_relation(y_s, y_t)
            kd_loss = self.beta*inter_loss + self.gamma*intra_loss
            return kd_loss
        
def KDST(T, vit_model_id, S, learning_rate, alpha, beta, gamma, tau, D_train, D_val, D_test, epochs=1):

    # init vit image processor
    vip = ViTImageProcessor.from_pretrained(vit_model_id)
    
    S.compile(
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5")
        ]
    )
    
    dist = DIST(beta, gamma, tau)
    
    # begin KDST
    for e in range(1, (epochs+1)):

        print("E P O C H  {}/{}".format(e, epochs))
        
        D_train = D_train.shuffle(D_train.cardinality())
        
        loss = 0.0
        loss_kd = 0.0
        loss_classification = 0.0
        for batch in D_train:
            # pass the images through the vit_model and get the output dictionary
            images = tf.convert_to_tensor(vip(batch[0])["pixel_values"])
            z_t = T(images, training=False)["logits"]
            # observe gradients with tape
            with tf.GradientTape() as tape:
                # augment and preprocess images
                images = kdst_aug(batch[0])
                z_s = S(images, training=True)
                # compute classification loss for this batch
                loss_classification_b = S.compiled_loss(batch[1], z_s)
                # compute kd loss for this batch
                loss_kd_b = dist(z_t, z_s)
                # compute toal training loss for this batch
                loss_b = alpha*loss_classification_b + loss_kd_b
            # compute the gradients of the loss with respect to the student model and projection functions
            grads = tape.gradient(loss_b, S.trainable_variables)
            # update the student model using the gradients
            S.optimizer.apply_gradients(zip(grads, S.trainable_variables))
            # update the metrics
            S.compiled_metrics.update_state(batch[1], z_s)
            # accumulate losses
            loss += loss_b.numpy()
            loss_kd += loss_kd_b.numpy()
            loss_classification += loss_classification_b.numpy()
        # aggregate losses
        loss /= D_train.cardinality().numpy()
        loss_kd /= D_train.cardinality().numpy()
        loss_classification /= D_train.cardinality().numpy()

        # show and log losses to wandb
        print("    training_loss_classification: {}".format(loss_classification))
        print("    training_loss_kd: {}".format(loss_kd))
        print("    training_loss: {}".format(loss))
        wandb.log({"training_loss_classification": loss_classification})
        wandb.log({"training_loss_kd": loss_kd})
        wandb.log({"training_loss": loss})

        # show and log training metrcis
        for metric in S.metrics:
            if metric.name == "loss": continue
            metric_name = metric.name
            metric_value = metric.result().numpy()
            print(f"    training_{metric_name}: {metric_value}")
            wandb.log({f"training_{metric_name}": metric_value})
        # reset the metrics for the next epoch
        S.reset_metrics()

        # validate student
        val_loss_classification = 0.0
        D_val = D_val.shuffle(D_val.cardinality())
        for batch in D_val:
            # pass the images through the student
            logits = S(batch[0], training=False)
            # compute classification loss for this batch
            loss_classification_b = S.compiled_loss(batch[1], logits)
            # update the metrics
            S.compiled_metrics.update_state(batch[1], logits)
            # accumulate losses
            val_loss_classification += loss_classification_b.numpy()
        # aggregate losses
        val_loss_classification /= D_train.cardinality().numpy()

        # show and log losses to wandb
        print("    val_loss_classification: {}".format(val_loss_classification))
        wandb.log({"val_loss_classification": val_loss_classification})

        # show and log val metrics
        for metric in S.metrics:
            if metric.name == "loss": continue
            metric_name = metric.name
            metric_value = metric.result().numpy()
            print(f"    val_{metric_name}: {metric_value}")
            # Log validation metrics to wandb
            wandb.log({f"val_{metric_name}": metric_value})
        # reset the metrics for the next epoch
        S.reset_metrics()
        print()

    # finally, evaluate student
    print("\nevaluating...")
    D_test = D_test.shuffle(D_test.cardinality())

    eval_loss = 0.0
    for batch in D_test:
        logits = S(batch[0], training=False)
        eval_loss += (S.compiled_loss(batch[1], logits)).numpy()
        # update the metrics
        S.compiled_metrics.update_state(batch[1], logits)
    eval_loss /= D_test.cardinality().numpy()
    
    # show and log eval metrics
    for metric in S.metrics:
        metric_name = metric.name
        metric_value = eval_loss if metric_name=="loss" else metric.result().numpy()
        print(f"eval_{metric_name}: {metric_value}")
        # Log evaluation metrics to wandb
        wandb.log({f"eval_{metric_name}": metric_value})
    # reset the metrics
    S.reset_metrics()
    print()

if __name__ == "__main__":

    strategy = __init_accelerators__()

    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_model_id", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--student_name", type=str, default="Resnet18")
    parser.add_argument("--dataset_id", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    args, _ = parser.parse_known_args()

    # init dirs

    # init vit weights directory
    if args.vit_model_id == "google/vit-base-patch16-224-in21k":
        vit_model_dir = "google_vit_base"
    elif args.vit_model_id == "google/vit-large-patch16-224-in21k":
        vit_model_dir = "google_vit_large"
    vit_weights_dir = "models/ViTs/{}/{}/weights/".format(vit_model_dir, args.dataset_id)

    # init output directory
    architecture_dir_map = {
        "ResNet18": "resnet18",
        "ResNet50": "resnet50s",
        "ResNet152": "resnet152",
    }

    student_model_dir = architecture_dir_map.get(args.student_name)

    if "ResNet" in args.student_name:
        # Extract the numbers at the end of the string
        numbers = re.findall(r"\d+$", args.student_name)
        if numbers:
            resnet_variant = numbers[0]

    student_weights_dir = "models/ConvNets/{}/{}/weights/".format(student_model_dir, args.dataset_id)

    # init output directory
    output_dir = "experiments/kdst/models/{}/{}/{}/".format(student_model_dir, args.dataset_id, vit_model_dir)

    # init wandb
    run = wandb.init(
        project="KDST",
        dir=output_dir,
        config={
            "teacher": args.vit_model_id,
            "student": args.student_name,
            "dataset": args.dataset_id,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "aplha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "tau": args.tau,
            "epochs": args.epochs,
            "optimizer": "AdamW"
        }
    )

    # init datasets
    D_train, D_val, D_test, k, h, w, c, id2label, label2id = setup_data(
        batch_size=args.batch_size,
        dataset_id=args.dataset_id)
    
    # init teacher ViT model

    vit = TFViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=args.vit_model_id,
        num_labels=k,
        id2label=id2label,
        label2id=label2id
    )

    status = vit.load_weights(vit_weights_dir)
    status.expect_partial()
    print("\nteacher vit ready!")
    print()

    # model creation
    preprocessing = tf.keras.Sequential(
        layers = [
            tf.keras.layers.Resizing(height=h, width=w, interpolation="bicubic"),
            tf.keras.layers.Rescaling(scale=(1.0/255.0))
        ],
        name = "preprocessing"
    )

    if args.student_name == "ResNet18" or args.student_name == "ResNet50" or args.student_name == "ResNet152":
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

    model.load_weights(filepath=student_weights_dir)

    KDST(vit,
         args.vit_model_id,
         model,
         args.learning_rate,
         args.alpha,
         args.beta,
         args.gamma,
         args.tau,
         D_train,
         D_val,
         D_test,
         epochs=args.epochs)
    
    # model.save("{}model/model.keras".format(output_dir))
    model.save_weights("{}weights/".format(output_dir))
