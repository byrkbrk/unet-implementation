# UNet implementation using PyTorch on Google Colab

We implement the well-known image segmentatation architecture, [U-net](https://arxiv.org/abs/1505.04597) for the segmentation of neural structures.
Because [the segmentation challenge website](brainiac2.mit.edu/isbi_challenge/) indicated in [the authors' website](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
is (apparently) no longer accessible, we use the datasets (i.e., volumes-train.tif, labels-train.tif, volumes-test.tif) shared in [the repository](https://github.com/zhixuhao/unet/tree/master/data/membrane).

Our UNet architecture is inspired by the Coursera course [Apply GANs](https://www.coursera.org/learn/apply-generative-adversarial-networks-gans/home/week/2).

---

## Requirements
This repository is designed to train the model and make inference entirely on Google Colab. So, for successful training and inference, it suffices to

- open the notebook `unet_cell_data.ipynb` on Colab (by either using this [link](https://colab.research.google.com/github/byrkbrk/unet-implementation/blob/main/unet_cell_data.ipynb) or the link *Open in Colab* at top left of the notebook)
- sign in your Google account (if you haven't yet)
- run the cells (with short comments) one by one
