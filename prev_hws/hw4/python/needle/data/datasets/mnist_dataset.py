from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        import gzip
        import struct
        with gzip.open(image_filename, "rb") as f:
            _, count, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape((count, rows * cols)).astype(np.float32) / 255.0
        with gzip.open(label_filename, "rb") as f:
            f.read(8)
            self.labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int8)
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
