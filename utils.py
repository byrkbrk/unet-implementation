import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def crop(tensor_images, new_shape):
    h = new_shape[2]
    w = new_shape[3]
    h_start = int((tensor_images.shape[2] - h) / 2)
    w_start = int((tensor_images.shape[3] - w) / 2)
    cropped_images = tensor_images[:, :, h_start:h_start+h, w_start:w_start+w]
    return cropped_images


def show_tensor_images(images, n_images, nrow=4, size=(1, 32, 32)):
    unflatted_images = images.detach().cpu().reshape(-1, *size)
    gridded_images = make_grid(unflatted_images[:n_images], nrow=nrow)
    plt.imshow(gridded_images.permute(1, 2, 0))
    plt.show()
