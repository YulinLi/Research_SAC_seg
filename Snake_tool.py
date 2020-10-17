import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from PIL import Image

grid_sample = False
gaussian = True
dot = False


def get_grid(w, h, x, y, crop_w, crop_h, grid_w, grid_h):
    ax = 1 / (w/2)
    bx = -1
    ay = 1 / (h/2)
    by = -1
    left_x = x - (crop_w/2)
    right_x = x + (crop_w/2)
    left_y = y - (crop_h/2)
    right_y = y + (crop_h/2)
    left_x = left_x*ax + bx
    right_x = right_x*ax + bx
    left_y = left_y*ay + by
    right_y = right_y*ay + by
    grid_x = torch.linspace(float(left_x), float(right_x), grid_w)
    grid_y = torch.linspace(float(left_y), float(right_y), grid_h)
    meshy, meshx = torch.meshgrid((grid_y, grid_x))
    grid = torch.stack((meshx, meshy), 2)
    grid = grid.unsqueeze(0)  # add batch dim
    return grid


def get_gaussian(x, y, var, kernel_size, image_size):
    img_gaus = np.zeros((image_size, image_size))

    for i in range(image_size):
        for j in range(image_size):
            # if(((i-x)**2+(j-y)**2) < kernel_size**2):
            img_gaus[j][i] = np.exp(-(((i-x)**2+(j-y)**2)/(2*(var**2))))

    img_gaus = np.expand_dims(img_gaus, 2)

    return img_gaus


def orientation_dot(v1, v2):
    normalize_action1 = v1/np.linalg.norm(v1)
    normalize_target1 = v2/np.linalg.norm(v2)
    dot_orien1 = np.dot(normalize_action1, normalize_target1)
    return dot_orien1


if __name__ == "__main__":
    path = "image/example.png"
    # grid create function test
    if(grid_sample):
        img = cv2.imread(path)
        img = np.asarray(img).astype(float)
        img = torch.FloatTensor(np.expand_dims(img, 0))
        img = img.permute(0, 3, 1, 2)

        grid1 = get_grid(128, 128, 25, 25, 32, 32, 32, 32)

        cropped1 = F.grid_sample(img, grid1)
        cropped1 = torch.squeeze(cropped1)
        cropped1 = cropped1.permute(1, 2, 0)

        # print(cropped1)
        img = img.permute(0, 2, 3, 1)
        img = torch.squeeze(img)
        print(img.shape)
        img = cropped1.cpu().detach().numpy()
        cv2.imshow("image", img)
        cv2.waitKey()
    # gaussian create function test
    if(gaussian):
        img = get_gaussian(133, 75, 16, 32, 128)
        img = np.asarray(img).astype(float)
        print(img.shape)
        cv2.imshow("image", img)
        cv2.waitKey()

    if(dot):
        v1 = [2, 3]
        v2 = [-1, 1]
        print(orientation_dot(v1, v2))
