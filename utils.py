


def crop(tensor_images, new_shape):
    h = new_shape[2]
    w = new_shape[3]
    h_start = int((tensor_images.shape[2] - h) / 2)
    w_start = int((tensor_images.shape[3] - w) / 2)
    cropped_images = tensor_images[:, :, h_start:h_start+h, w_start:w_start+w]

    return cropped_images
