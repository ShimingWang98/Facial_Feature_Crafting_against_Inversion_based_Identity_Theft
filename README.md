# Crafter: Facial Feature Crafting against Inversion-based Identity Theft on Deep Models

*by Shiming Wang, Zhe Ji, Liyao Xiang, Hao Zhang, Xinbing Wang, Chenghu Zhou, Bo Li* 

## Overview

This repository is the code implementation and supplementary material of our [NDSS '24 paper](https://www.ndss-symposium.org/wp-content/uploads/2024-326-paper.pdf).
We provide a defense to protect identity information in facial images from model inversion attacks. 
It can be deployed as a plug-in to general edge-cloud computing frameworks without any change in the backend models.

It is compatible with general deep learning tasks including both inference and training:

- Inference: The server itself trains both feature extractor and downstream classifier without our defense. Then the feature extractor and our defense run on edge devices. The downstream classifier on cloud receives the protected feature as input and completes the inference.

- Training: The server deploys a pretrained feature extractor to edge devices. Edge devices run our defense and upload the protected features to the cloud, which serve as the training data of the downstream classifier.

Contributions: high utility on general ML tasks; robustness against adaptive attackers.

## Setup

### Prerequisites

Make sure you have Pytorch 1.10 with cuda version installed on your machine. See [Pytorch official websites ](https://pytorch.org/get-started/locally/) for a detailed installation guide.

In addition, install python dependencies by running:

```shell
$ pip3 install -r requirements.txt
```

## Quick Start

### Dataset Preprocessing

The experiments run on cropped and resized CelebA dataset with image size 64 * 64,  and LFW dataset with image size 128 * 128. Due to the limited size of repository, the preprocessed datasets are not provided here. The dataset dictionary should have the structure shown as follows.

```
.
+-- public
|   +-- image files (jpg)
+-- private
|   +-- image files (jpg)
+-- pub_attri.csv
+-- pub_id.csv
+-- pvt_attri.csv
+-- pvt_id.csv
+-- eval_train.csv
+-- eval_test.csv
+-- eval_train_attri.csv
+-- eval_test_attri.csv
```

The `public` dictionary and `private` dictionary are a split of the whole dataset, with no ID overlapping.  Images in `private` are considered as faces of users, which should not be exposed to an attacker, while images in `public` are considered accessible to anyone.  The information about face IDs of `public` and `private` is stored in `pub_id.csv` and `pvt_id.csv` respectively. The information of attributes of faces, such as "Big nose" and "Eyeglasses", of `public` and `private` is stored in `pub_attri.csv` and `pvt_attri.csv` respectively.

The `eval_train.csv` and `eval_test.csv` are a split of `pvt_id.csv`, with completely ID overlapping. `eval_train.csv` is used to train ID classifier to evaluate the performance of protection algorithm. `eval_train_attri.csv` and `eval_test_attri.csv` are corresponding split of `pvt_attri.csv`, used to evaluate utility later.

### Construct an White-box Attacker

The crafter protection contains an adversarial training process, meaning our protection uses fictitious attack runs.

#### Train GAN

The generator of a GAN is the prior knowledge of the fictitious attacker, trained by `public` images.

Change the work dictionary to `WhiteBoxAttack` and run

````shell
$ python train_gan.py --latent_dim 500 --img_size 64
````

 to train the GAN. You can modify the value of `latent_dim`, which is the dimension of the latent vector (the input of the generator). We use 500 for CelebA and 3000 for LFW. The `img_size` should match the images in previous section. 

The result will be saved in `repo_root/Gan/`.

#### Train Amortize Net

The white-box attacker will solve a optimization problem on the domain of latent vector, the input of the generator. To solve the problem, the attacker need to initialize the value of latent vector $z$, to use gradient descent method. To shorten the time required to attack, we train an amortize net which receive a feature as input and output a $z$ value to approximate the solution of the optimization problem. 

```shell
$ python trainAmor.py
```

Some hyperparameters in this code may need to be adjusted.

#### White-box Attack

The function that run a white-box attack is implemented in `Crafter_protection/attacker.py`, which will be imported later.

### Inference Scenario

The whole inference task will take an image as the input, feed it into an encoder to get the feature, then put the feature into a classifier to get the prediction result.

Here we assume you have a trained encoder and a downstream classifier. Otherwise please run the following code to get one. 

```shell
$ python trainTargetNet.py -whole
```

In our experiments, the structures of the encoder and classifier are a segmentation of ResNet18. The combination of the both forms a complete ResNet18. The definitions of these nets are shown in `Crafter_protection/models.py`.

Once the target net is prepared, the crafter protection algorithm is ready to run.

```shell
$ python featureCrafter.py
```

The code will save the released protected features as files. The released protected features are exactly what the edge devices need to send to the cloud server.

For z crafter algorithm, run `python zCrafter.py`. 

### Training Scenario

The steps for training scenario is similar to inference one. The difference is that the protected features will be the training dataset for the downstream classifier.

Change the work dictionary to `Craft_protection` and run the following code to train the downstream classifier.

```shell
$ python trainClassifier.py
```

## Evaluate Privacy and Utility

The evaluation is done by code `evaluate.py`. 

### Privacy

**Step1.**  Use both black-box and white-box attack to reconstruct images of `dataset/eval_test.csv` given protected features. 

The black-box attack is implemented in `BlackBoxAttack` dictionary. The white-box attack is the one we used in Crafter protection.

**Step2.**  Feed the reconstructed images into an ID classifier trained on `dataset/eval_train.csv`

The training code is `Image2ID/evalTrain.py`. The structure of ID classifier is defined in `models.py`. 

**Step3.**  Calculate the accuracy of the ID classifier as the fictitious attacker's success rate. Calculate the SSIM and FSIM metrics between (reconstructed image, original image) pair as an indicator of the  quality of the reconstructed images.

There are more choices for the ID classifier:

1. Python library `facenet_pytorch` provide some pretrained identification network. Limitations: Image size 64 * 64 is too small for these provided network. This choice is only used in case of LFW 128 * 128. Training code is `Image2ID/facenetTrain.py`.
2. Commercial Face Identification Service. In the experiments, we upload the `eval_train` set to Azure server and upload reconstructed images to query their IDs. The same limitation as `facenet_pytorch`. 
3. Instead of predicting ID given the reconstructed images, an attacker can also predict ID directly given the protected feature. The feature to ID attack is implemented in `Feature2ID`. 

### Utility

We use multiple binary classification tasks as the evaluation of downstream utility. For CelebA dataset, we choose all of the 40 attributes to classify, while we choose 10 of 73 attributes of LFW dataset. The metric to evaluate the performance of the downstream classification is **Area Under Curve** (AUC), implemented by python lib `sklearn.metrics`. 
