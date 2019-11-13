import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# # add scalar
# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# # add image
# img = np.zeros((3, 100, 100))
# img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

# img_HWC = np.zeros((100, 100, 3))
# img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

# writer.add_image('my_image', img, 0)

# # If you have non-default dimension setting, set the dataformats argument.
# writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')


vertices_tensor = torch.as_tensor([
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, -1],
    [-1, 1, -1],
], dtype=torch.float).unsqueeze(0)
colors_tensor = torch.as_tensor([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
], dtype=torch.int).unsqueeze(0)
faces_tensor = torch.as_tensor([
    [0, 2, 3],
    [0, 3, 1],
    [0, 1, 2],
    [1, 3, 2],
], dtype=torch.int).unsqueeze(0)

writer = SummaryWriter()
writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

writer.close()