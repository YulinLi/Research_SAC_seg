import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Snake_tool as tool

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC():
    def __init__(
        self,
        model,
        n_actions,
        learning_rate=[1e-4, 2e-4],
        reward_decay=0.98,
        replace_target_iter=300,
        memory_size=5000,
        batch_size=64,
        tau=0.01,
        alpha=0.5,
        auto_entropy_tuning=True,
        criterion=nn.MSELoss()
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.criterion = criterion
        self._build_net(model[0], model[1])
        self.init_memory()

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(device)
        self.critic_target.eval()

        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.Tensor(self.n_actions).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0001)

    def save_load_model(self, op, path):
        anet_path = path + "sac_anet.pt"
        cnet_path = path + "sac_cnet.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif op == "load":
            self.critic.load_state_dict(
                torch.load(cnet_path, map_location=device))
            self.critic_target.load_state_dict(
                torch.load(cnet_path, map_location=device))
            self.actor.load_state_dict(
                torch.load(anet_path, map_location=device))

    def choose_action(self, s, eval=False):
        # current coordinate
        s_coord = torch.FloatTensor(np.expand_dims(s[1:], 0)).to(device)
        # full image
        s_img = torch.FloatTensor(np.expand_dims(s[0], 0)).to(device)
        s_img = s_img.permute(0, 3, 1, 2)
        # gaussian image
        s_g_img = tool.get_gaussian(
            s_coord[0, 0].cpu().detach().numpy(), s_coord[0, 1].cpu().detach().numpy(), 16, 32, 128)
        """
        # Test block start-gaussian image
        img = np.asarray(s_g_img).astype(float)
        cv2.imshow("image", img)
        cv2.waitKey()
        # Test block end
        """
        s_g_img = torch.FloatTensor(np.expand_dims(s_g_img, 0)).cuda()
        s_g_img = s_g_img.permute(0, 3, 1, 2)
        # concat full image and gaussian image
        s_h_img = torch.cat((s_img, s_g_img), dim=1)
        # cropped image
        # 1.convert coordinate to grid
        B_s_coord, _ = s_coord.shape
        grid1 = tool.get_grid(
            128, 128, s_coord[0, 0], s_coord[0, 1], 32, 32, 32, 32).cuda()
        if B_s_coord > 1:
            for i in range(1, B_s_coord):
                grid_temp = tool.get_grid(
                    128, 128, s_coord[i, 0], s_coord[i, 1], 32, 32, 32, 32).cuda()
                grid1 = torch.cat((grid1, grid_temp), dim=0)
        s_cropped1 = F.grid_sample(s_img, grid1)
        # Test block start-cropped image
        """
        cropped1 = torch.squeeze(s_cropped1)
        cropped1 = cropped1.permute(1, 2, 0)
        img = cropped1.cpu().detach().numpy()
        cv2.imshow("image", img)
        cv2.waitKey()
        """
        # Test block end
        """
        grid2 = tool.get_grid(
            128, 128, s_coord[0, 2], s_coord[0, 3], 32, 32, 32, 32).cuda()
        if B_s_coord > 1:
            for i in range(1, B_s_coord):
                grid_temp = tool.get_grid(
                    128, 128, s_coord[i, 2], s_coord[i, 3], 32, 32, 32, 32).cuda()
                grid2 = torch.cat((grid2, grid_temp), dim=0)
        cropped2 = F.grid_sample(s_img, grid2)

        grid3 = tool.get_grid(
            128, 128, s_coord[0, 4], s_coord[0, 5], 32, 32, 32, 32).cuda()
        if B_s_coord > 1:
            for i in range(1, B_s_coord):
                grid_temp = tool.get_grid(
                    128, 128, s_coord[i, 4], s_coord[i, 5], 32, 32, 32, 32).cuda()
                grid3 = torch.cat((grid3, grid_temp), dim=0)
        cropped3 = F.grid_sample(s_img, grid3)

        grid4 = tool.get_grid(
            128, 128, s_coord[0, 6], s_coord[0, 7], 32, 32, 32, 32).cuda()
        if B_s_coord > 1:
            for i in range(1, B_s_coord):
                grid_temp = tool.get_grid(
                    128, 128, s_coord[i, 6], s_coord[i, 7], 32, 32, 32, 32).cuda()
                grid4 = torch.cat((grid4, grid_temp), dim=0)
        cropped4 = F.grid_sample(s_img, grid4)
        """

        if eval == False:
            action, _, _ = self.actor.sample(s_h_img, s_cropped1)
        else:
            _, _, action = self.actor.sample(s_h_img, s_cropped1)

        action = action.cpu().detach().numpy()[0]
        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s": [], "a": [], "r": [], "sn": [], "end": []}

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            self.memory["end"][index] = end

        self.memory_counter += 1

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_(
                    (1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)

        s_batch = [self.memory["s"][index] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        sn_batch = [self.memory["sn"][index] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]

        # Construct torch tensor
        s_batch_np = np.array(s_batch)
        s_coord = torch.FloatTensor(
            np.array(s_batch_np[:, 1:]).astype(float)).to(device)
        # current image state
        s_img = torch.FloatTensor(np.stack(np.array(s_batch)[:, 0])).to(device)
        """
        # Test block start-gaussian image
        for i in range(16):
            img = s_img[i, :, :, :].cpu().detach().numpy()
            cv2.imshow("image", img)
            cv2.waitKey()
        # Test block end
        """
        s_img = s_img.permute(0, 3, 1, 2)
        B, _, _, _ = s_img.shape
        # gaussian image
        s_g_img = tool.get_gaussian(
            s_coord[0, 0].cpu().detach().numpy(), s_coord[0, 1].cpu().detach().numpy(), 16, 32, 128)
        s_g_img = torch.FloatTensor(np.expand_dims(s_g_img, 0)).cuda()
        s_g_img = s_g_img.permute(0, 3, 1, 2)
        """
        print(s_coord[0, 0].cpu().detach().numpy())
        print(s_coord[0, 1].cpu().detach().numpy())
        g_img = torch.squeeze(s_g_img).cpu().detach().numpy()
        print(g_img.shape)
        cv2.imshow("image", g_img)
        cv2.waitKey()
        """
        s_g_img_batch = s_g_img
        for i in range(1, B):
            s_g_img = tool.get_gaussian(s_coord[i, 0].cpu().detach(
            ).numpy(), s_coord[i, 1].cpu().detach().numpy(), 16, 32, 128)
            s_g_img = torch.FloatTensor(np.expand_dims(s_g_img, 0)).cuda()
            s_g_img = s_g_img.permute(0, 3, 1, 2)
            """
            print(s_coord[i, 0].cpu().detach().numpy())
            print(s_coord[i, 1].cpu().detach().numpy())
            g_img = torch.squeeze(s_g_img).cpu().detach().numpy()
            cv2.imshow("image", g_img)
            cv2.waitKey()
            """
            s_g_img_batch = torch.cat((s_g_img_batch, s_g_img), dim=0)
        # concat full image and gaussian image
        s_h_img = torch.cat((s_img, s_g_img_batch), dim=1)
        print(s_h_img.shape)
        # print(np.array(s_batch_np[:, 1:]).astype(float).dtype)
        # cropped image
        # 1.convert coordinate to grid
        B_s_coord, _ = s_coord.shape
        grid1 = tool.get_grid(
            128, 128, s_coord[0, 0], s_coord[0, 1], 32, 32, 32, 32).cuda()
        if B_s_coord > 1:
            for i in range(1, B_s_coord):
                grid_temp = tool.get_grid(
                    128, 128, s_coord[i, 0], s_coord[i, 1], 32, 32, 32, 32).cuda()
                grid1 = torch.cat((grid1, grid_temp), dim=0)
        s_cropped1 = F.grid_sample(s_img, grid1)
        # action
        a_ts = torch.FloatTensor(np.array(a_batch)).to(device)
        # reward
        r_ts = torch.FloatTensor(np.array(r_batch)).to(
            device).view(self.batch_size, 1)
        # next img state
        sn_batch_np = np.array(sn_batch)
        sn_coord = torch.FloatTensor(
            np.array(sn_batch_np[:, 1:]).astype(float)).to(device)
        # next state image
        sn_img = torch.FloatTensor(
            np.stack(np.array(sn_batch)[:, 0])).to(device)
        sn_img = sn_img.permute(0, 3, 1, 2)
        B, _, _, _ = sn_img.shape
        # next gaussian image
        sn_g_img = tool.get_gaussian(
            sn_coord[0, 0].cpu().detach().numpy(), sn_coord[0, 1].cpu().detach().numpy(), 16, 128, 128)
        sn_g_img = torch.FloatTensor(np.expand_dims(sn_g_img, 0)).cuda()
        sn_g_img = sn_g_img.permute(0, 3, 1, 2)

        sn_g_img_batch = sn_g_img
        for i in range(1, B):
            sn_g_img = tool.get_gaussian(sn_coord[i, 0].cpu().detach(
            ).numpy(), sn_coord[i, 1].cpu().detach().numpy(), 16, 32, 128)

            sn_g_img = torch.FloatTensor(np.expand_dims(sn_g_img, 0)).cuda()

            sn_g_img = sn_g_img.permute(0, 3, 1, 2)
            sn_g_img_batch = torch.cat((sn_g_img_batch, sn_g_img), dim=0)
        # concat full image and gaussian image
        sn_h_img = torch.cat((sn_img, sn_g_img_batch), dim=1)

        # cropped image
        # 1.convert coordinate to grid
        B_sn_coord, _sn_coordN = sn_coord.shape
        grid1 = tool.get_grid(
            128, 128, sn_coord[0, 0], sn_coord[0, 1], 32, 32, 32, 32).cuda()
        if B_sn_coord > 1:
            for i in range(1, B_sn_coord):
                grid_temp = tool.get_grid(
                    128, 128, sn_coord[i, 0], sn_coord[i, 1], 32, 32, 32, 32).cuda()
                grid1 = torch.cat((grid1, grid_temp), dim=0)
        # 2.grid sample
        sn_cropped1 = F.grid_sample(sn_img, grid1)

        end_ts = torch.FloatTensor(np.array(end_batch)).to(
            device).view(self.batch_size, 1)

        # TD-target
        with torch.no_grad():
            a_next, logpi_next, _ = self.actor.sample(sn_h_img, sn_cropped1)
            q_next_target = self.critic_target(
                sn_h_img, sn_cropped1, a_next) - self.alpha * logpi_next
            q_target = r_ts + end_ts * self.gamma * q_next_target

        # Critic loss
        q_eval = self.critic(s_h_img, s_cropped1, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        a_curr, logpi_curr, _ = self.actor.sample(s_h_img, s_cropped1)
        q_current = self.critic(s_h_img, s_cropped1, a_curr)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()

        # Adaptive entropy adjustment
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr +
                                             self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())

        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())
