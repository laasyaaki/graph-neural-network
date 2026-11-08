from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, "rb") as f:
        magic, num_images, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8)
        X = X.reshape(num_images, r * c)
        X = X.astype(np.float32) / 255.0

    with gzip.open(label_filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int8)

    return X, y
    ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        self.images, self.labels = parse_mnist(
            image_filename, label_filename
        )
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = np.asarray(self.images[index])
        labels = self.labels[index]

        if images.ndim == 1:
            image = images.reshape((28, 28, 1))
            image = self.apply_transforms(image)
            return np.asarray(image), labels

        transformed = []
        for image in images:
            image = image.reshape((28, 28, 1))
            image = self.apply_transforms(image)
            transformed.append(np.asarray(image))

        return np.stack(transformed), labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION