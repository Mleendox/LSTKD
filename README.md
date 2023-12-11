# LSTKD
 Our focus is on knowledge distillation (KD), a method that aims to create smaller yet efficient models. In this research, we specifically explore transferring knowledge from powerful vision transformers (ViTs) to convolutional neural networks (CNNs). Specifically, the ResNet variants 18, 50, and 152 were used.

 Firstly, the teacher ViT's and student CNN's are trained on CIFAR-10 and CIFAR-100 independently to obtain their optimal performance. This can be found in the `train_scripts` directory. The trained model weights are saved in the `models` directory under their respective names and the dataset on which they were trained.

 Next, the KD is performed on the students. This can be found under the `experiments` directory. The training hyperparameters can be found in the paper. The trained model weights are saved in the `models` directory in the same `experiments` directory. They are named according to the student model name, the training dataset used and the teacher ViT that trained the student.
