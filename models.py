import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import cv2


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(23, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        mu = torch.tanh(self.fc4(h_fc3))
        return mu


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class PolicyNetGaussian(nn.Module):
    def __init__(self):
        super(PolicyNetGaussian, self).__init__()
        # full image
        self.full_img_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Flatten(),
            nn.Linear(6272, 64)
        )
        # cropped image
        self.cropped_img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Conv2d(16, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Flatten(),
            nn.Linear(64, 16)
        )
        # coordinate s[1:8]

        # TODO1 : shape???
        self.fc1 = nn.Linear(80, 128)  # TODO1 shape
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        # none change
        self.fc4_mean = nn.Linear(128, 2)
        self.fc4_logstd = nn.Linear(128, 2)

    def forward(self, s_img, s_cropped):
        # full image conv
        img_cov1 = self.full_img_encoder(s_img)
        # print(img_cov1.shape)
        # cropped images conv four times
        # 2.cropped image conv
        img_cov2 = self.cropped_img_encoder(s_cropped)
        """
        img_cov3 = self.cropped_img_encoder(cropped2)
        img_cov4 = self.cropped_img_encoder(cropped3)
        img_cov5 = self.cropped_img_encoder(cropped4)
        """
        # concat all
        s_full = torch.cat((img_cov1, img_cov2), 1)
        # print(s_full)
        # print(s_full.shape)
        # mirror to Q
        h_fc1 = F.relu(self.fc1(s_full))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        # none change
        a_mean = self.fc4_mean(h_fc3)
        a_logstd = self.fc4_logstd(h_fc3)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return a_mean, a_logstd

    def sample(self, s_img, s_coord):
        a_mean, a_logstd = self.forward(s_img, s_coord)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        # full image
        self.full_img_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Flatten(),
            nn.Linear(6272, 64)
        )
        # cropped image
        self.cropped_img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Conv2d(16, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(
                4, 4), dilation=(1, 1), ceil_mode=False),

            nn.Flatten(),
            nn.Linear(64, 16)
        )

        self.fc1 = nn.Linear(80, 128)
        self.fc2 = nn.Linear(128+2, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, s_img, s_cropped, a):
        # mirror to policy gradient
        # full image conv
        img_cov1 = self.full_img_encoder(s_img)

        # 2.cropped image conv
        img_cov2 = self.cropped_img_encoder(s_cropped)
        """
        img_cov3 = self.cropped_img_encoder(cropped2)
        img_cov4 = self.cropped_img_encoder(cropped3)
        img_cov5 = self.cropped_img_encoder(cropped4)
        """
        # concat all
        s_full = torch.cat((img_cov1, img_cov2), 1)
        # coordinate
        h_fc1 = F.relu(self.fc1(s_full))
        # concat action, have multiple
        h_fc1_a = torch.cat((h_fc1, a), 1)
        h_fc2 = F.relu(self.fc2(h_fc1_a))
        h_fc3 = F.relu(self.fc3(h_fc2))
        # mirror
        q_out = self.fc4(h_fc3)
        return q_out
