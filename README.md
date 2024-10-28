# Crafter: Facial Feature Crafting against Inversion-based Identity Theft on Deep Models

*by Shiming Wang, Zhe Ji, Liyao Xiang, Hao Zhang, Xinbing Wang, Chenghu Zhou, Bo Li* 

## Overview

This repository is the code implementation and supplementary material of our [NDSS '24 paper](https://www.ndss-symposium.org/wp-content/uploads/2024-326-paper.pdf).
We provide a defense to protect identity information in facial images from model inversion attacks. 
It can be deployed as a plug-in to general edge-cloud computing frameworks without any change in the backend models.

The defense is compatible with general deep learning tasks including both inference and training:

- Inference: The server itself trains both feature extractor and downstream classifier without our defense. Then the feature extractor and our defense run on edge devices. The downstream classifier on cloud receives the protected feature as input and completes the inference.

- Training: The server deploys a pretrained feature extractor to edge devices. Edge devices run our defense and upload the protected features to the cloud, which then serve as the training data of the downstream classifier.

Contributions: high utility on general ML tasks; robustness against adaptive attackers.

## Setup

Make sure you have Pytorch 1.10 with cuda version installed on your machine. See [Pytorch official websites ](https://pytorch.org/get-started/locally/) for a detailed installation guide.

In addition, install python dependencies by running:

```shell
$ pip3 install -r requirements.txt
```

## Quick Start

### Dataset Preprocessing

The experiments run on cropped and resized CelebA dataset with image size 64 * 64,  and LFW with image size 128 * 128. The dataset directory should look like this:

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

The `public` and `private` directories are a split of the entire dataset, with no ID overlapping.  Images in `private` are faces of private users, which should not be exposed to an attacker. Images in `public` are considered accessible to anyone, eg. celebrity faces crawled from the Internet.  Face IDs are in `pub_id.csv` and `pvt_id.csv`. Facials attributes (e.g. "Big nose", "Eyeglasses" etc.) are  in `pub_attri.csv` and `pvt_attri.csv`.

The `eval_train.csv` and `eval_test.csv` are a split of `pvt_id.csv`, with complete ID overlapping. `eval_train.csv` is used to train an ID classifier that evaluates the privacy of our defense as an oracle. `eval_train_attri.csv` and `eval_test_attri.csv` are a split of `pvt_attri.csv`, used to evaluate the utility of our defense.


### Construct an White-box Attacker

Our crafter protection is based on adversarial training --- it uses fictitious white-box attack runs. You may simulate a white-box attacker as follows.

#### Train GAN

You will first need a GAN generator trained on `public` images, representing the prior knowledge of the fictitious attacker.

Download a pretrained public GAN parameter, or train your own GAN with
````shell
$ cd WhiteBoxAttack/
$ python train_gan.py --latent_dim 500 --img_size 64
````
You can modify  `latent_dim`, the dimension of the latent vector (the input of the generator). In our test, setting it to 500 for CelebA and 3000 for LFW does the trick. The `img_size` should match your image dimension. 

The result will be saved in `repo_root/Gan/`.

#### Train Amortize Net (optional)

White-box attackers solve an optimization problem on the latent vector, starting at a random initialization point. This optimization process can take quite a while to finish. To speed it up, you may train an amortizor model --- it receives a feature as input and outputs a latent vector $z$, which will be the new initialization point. To train one, run

```shell
$ python trainAmor.py
$ cd ../
```


#### White-box Attack

The complete white-box attack is implemented in `WhiteBoxAttack/attacker.py`.

### Inference Scenario

The inference task passes an input image through an encoder and gets the feature, which is then fed into a classifier to generate the final prediction.

Here you should have a trained encoder and a downstream classifier, with models defined in `Crafter_protection/models.py` and parameters stored under `params/`. It not, try  
```shell
$ cd Crafter_protection/
$ python trainTargetNet.py -whole
```
It will train a splitted ResNet18 as a demo.

You are now ready to run the crafter protection with
```shell
$ python featureCrafter.py
$ cd ../
```

The protected features will be saved to `../Crafter_result/`, ready for cloud server inference.

We also implement the inferior z-crafter protection in `zCrafter.py`. It runs much faster than featureCrafter, but comes at the cost of weaker protection.



### Training Scenario

The training scenario is pretty much the same as the inference one. The only difference is that the encoder and cloud models are not trained apriori, and the protected features serve as the training data for the cloud model. Run

```shell
$ python trainClassifier.py
$ cd ../
```

## Evaluate Privacy and Utility

### Privacy

To see how well the features are protected, you may use either black-box or white-box attack to reconstruct images of `dataset/eval_test.csv`. 

**Step1. Build the attackers**

The white-box attack is the one we used in Crafter protection. 
 If you wish to simulate an adaptive white-box inversion attacker, run
```shell
$ cd WhiteBoxAttack
$ python adaptiveUpdateG.py
$ cd ../
```
To train a black-box attack, run
```shell
$ cd BlackBoxAttack
$ python trainDecoder.py
```
and the attacker model will be saved to `params/dec.pkl`. If you wish to simulate an adaptive blackbox inversion attacker, run
```shell
$ python adaptiveTrainDecoder.py
```
and the attacker model will be saved to `params/dec_adaptive.pkl`.

**Step2. Train the evaluating networks.** 
The oracle evaluating network (an ID classification network) is defined in `models.py`. To train it, run
```shell
$ cd Image2ID/
$ python evalTrain.py
```
You can also define your own oracle and train it on `dataset/eval_train.csv`.
Some otther choices for the oracle are:

1. Python library `facenet_pytorch` provide some pretrained identification network. Limitations: Image size 64 * 64 is too small for these provided network. This choice is only implemented on LFW of size 128 * 128. The training code is `Image2ID/facenetTrain.py`.
2. Commercial Face Identification Service. You can upload the `eval_train` set to Azure server, and then upload the reconstructed images to query their IDs. Same as `facenet_pytorch`, Azure does not work on small images either. 
3. Instead of predicting ID from the reconstructed images, an attacker can also predict the ID from the protected feature directly. This attack is implemented in `Feature2ID`. 



**Step3. Reconstruct the images and test their privacy.**  

Run 
```shell
$ python evaluate.py
```
to check the ID classification accuracy, SSIM and FSIM metrics as indicators of the  privacy of the reconstructed images.

### Utility

To simulate a challenging utility task, we choose the multiple binary classification tasks to evaluate the protected feature's utility. For CelebA dataset, we classify all 40 attributes all at once. For the LFW dataset, we choose 10 out of 73 attributes. lay around with these options in `evaluate.py` and `metrics.py` to customize as you like!
