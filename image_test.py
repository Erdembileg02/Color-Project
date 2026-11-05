
import numpy as np

image = np.array([
    [[0, 0, 255], [0, 255, 0]],
    [[255, 0, 0], [0, 255, 255]]
], dtype=np.uint8)

print(image.shape)
print(image[0, 0])
