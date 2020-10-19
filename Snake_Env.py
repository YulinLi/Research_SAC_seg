import cv2
import numpy as np
import Snake_tool as tool


class SnakeContourPoint:
    def __init__(self, idx):
        self.idx = idx

    def init_state(self, pos):
        self.x = pos[0]
        self.y = pos[1]


class SnakeEnv:
    def __init__(self, path="image/example.png"):
        self.img = cv2.imread(path)
        self.img = np.asarray(self.img/255).astype(float)
        self.w = 128  # self.width

    def initialize(self):
        # Set Init Snake Point
        # Snake_point_list = []
        sx = np.random.randint(0, 125/2)
        sy = np.random.randint(0, 50/2)
        self.snake1 = SnakeContourPoint(1)
        self.snake1.init_state((10, 2))
        # self.snake1.init_state((self.w/2, 0))

        """
        self.snake2 = SnakeContourPoint(2)
        self.snake2.init_state((self.w/2, 0))
        self.snake3 = SnakeContourPoint(3)
        self.snake3.init_state((self.w, self.w/2))
        self.snake4 = SnakeContourPoint(4)
        self.snake4.init_state((self.w/2, self.w))

        self.snake5 = SnakeContourPoint(5)
        self.snake5.init_state((self.w, self.w/2))
        self.snake6 = SnakeContourPoint(6)
        self.snake6.init_state((self.w*3/4, self.w*3/4))
        self.snake7 = SnakeContourPoint(7)
        self.snake7.init_state((self.w/2, self.w))
        self.snake8 = SnakeContourPoint(8)
        self.snake8.init_state((self.w/4, self.w*3/4))
        """
        self.target1 = SnakeContourPoint(-1)
        self.target1.init_state((25, 25))
        self.target2 = SnakeContourPoint(-2)
        self.target2.init_state((100, 25))
        self.target3 = SnakeContourPoint(-3)
        self.target3.init_state((100, 75))
        self.target4 = SnakeContourPoint(-4)
        self.target4.init_state((25, 75))
        # Set Target and distance
        if(self.snake1.x < 125/2 and self.snake1.y < 50/2):
            # self.target1 = (25, 25)
            self.target = self.target1
            self.target_dist = np.sqrt(
                (self.snake1.x - self.target.x)**2 + (self.snake1.y - self.target.y)**2)
        elif(self.snake1.x > 125/2 and self.snake1.y < 50/2):
            # self.target1 = (100, 25)
            self.target = self.target2
            self.target_dist = np.sqrt(
                (self.snake1.x - self.target.x)**2 + (self.snake1.y - self.target.y)**2)
        elif(self.snake1.x > 125/2 and self.snake1.y > 50/2):
            # self.target1 = (100, 75)
            self.target = self.target3
            self.target_dist = np.sqrt(
                (self.snake1.x - self.target.x)**2 + (self.snake1.y - self.target.y)**2)
        elif(self.snake1.x < 125/2 and self.snake1.y > 50/2):
            self.target = self.target4
            self.target_dist = np.sqrt(
                (self.snake1.x - self.target.x)**2 + (self.snake1.y - self.target.y)**2)
        # return state
        state = self._construct_state(self.img,  self.snake1)
        """, self.snake2, self.snake3, self.snake4"""
        return state

    def step(self, action):
        # Update snake point position
        self.snake1.x += 10*action[0]
        self.snake1.y += 10*action[1]
        """
        self.snake2.x += 5*action[2]
        self.snake2.y += 5*action[3]
        self.snake3.x += 5*action[4]
        self.snake3.y += 5*action[5]
        self.snake4.x += 5*action[6]
        self.snake4.y += 5*action[7]
        """
        # Constrain action range
        """
        if(self.snake1.x > self.w):
            self.snake1.x = self.w
        elif(self.snake1.x < 0):
            self.snake1.x = 0

        if(self.snake1.y > self.w):
            self.snake1.y = self.w
        elif(self.snake1.y < 0):
            self.snake1.y = 0
        """
        # Change target if near than origin one
        """
        if(self.snake1.x < 125/2 and self.snake1.y < 50/2):
            # self.target1 = (25, 25)
            self.target = self.target1
        elif(self.snake1.x > 125/2 and self.snake1.y < 50/2):
            # self.target1 = (100, 25)
            self.target = self.target2
        elif(self.snake1.x > 125/2 and self.snake1.y > 50/2):
            # self.target1 = (100, 75)
            self.target = self.target3
        elif(self.snake1.x < 125/2 and self.snake1.y > 50/2):
            self.target = self.target4
        """
        """
        if(self.snake2.x > self.w):
            self.snake2.x = self.w
        elif(self.snake2.x < 0):
            self.snake2.x = 0

        if(self.snake2.y > self.w):
            self.snake2.y = self.w
        elif(self.snake2.y < 0):
            self.snake2.y = 0

        if(self.snake3.x > self.w):
            self.snake3.x = self.w
        elif(self.snake3.x < 0):
            self.snake3.x = 0

        if(self.snake3.y > self.w):
            self.snake3.y = self.w
        elif(self.snake3.y < 0):
            self.snake3.y = 0

        if(self.snake4.x > self.w):
            self.snake4.x = self.w
        elif(self.snake4.x < 0):
            self.snake4.x = 0

        if(self.snake4.y > self.w):
            self.snake4.y = self.w
        elif(self.snake4.y < 0):
            self.snake4.y = 0

        """
        """
        self.snake5.x += action[8]
        self.snake5.y += action[9]
        self.snake6.x += action[10]
        self.snake6.y += action[11]
        self.snake7.x += action[12]
        self.snake7.y += action[13]
        self.snake8.x += action[14]
        self.snake8.y += action[15]
        """
        # Distance Reward
        # 1
        curr_target_dist = np.sqrt(
            (self.snake1.x - self.target.x)**2 + (self.snake1.y - self.target.y)**2)
        reward_dist1 = self.target_dist - curr_target_dist
        """
        # 2
        curr_target_dist2 = np.sqrt(
            (self.snake2.x - self.target2[0])**2 + (self.snake2.y - self.target2[1])**2)
        reward_dist2 = self.target_dist2 - curr_target_dist2
        # 3
        curr_target_dist3 = np.sqrt(
            (self.snake3.x - self.target3[0])**2 + (self.snake3.y - self.target3[1])**2)
        reward_dist3 = self.target_dist3 - curr_target_dist3
        # 4
        curr_target_dist4 = np.sqrt(
            (self.snake4.x - self.target4[0])**2 + (self.snake4.y - self.target4[1])**2)
        reward_dist4 = self.target_dist4 - curr_target_dist4

        # 5
        curr_target_dist5 = np.sqrt(
            (self.snake5.x - self.target5[0])**2 + (self.snake5.y - self.target5[5])**2)
        reward_dist5 = self.target_dist5 - curr_target_dist5
        # 6M
        curr_target_dist6 = np.sqrt(
            (self.snake6.x - self.target6[0])**2 + (self.snake6.y - self.target6[6])**2)
        reward_dist6 = self.target_dist6 - curr_target_dist6
        # 7
        curr_target_dist7 = np.sqrt(
            (self.snake7.x - self.target7[0])**2 + (self.snake7.y - self.target7[7])**2)
        reward_dist7 = self.target_dist7 - curr_target_dist7
        # 8
        curr_target_dist8 = np.sqrt(
            (self.snake8.x - self.target8[0])**2 + (self.snake8.y - self.target8[8])**2)
        reward_dist8 = self.target_dist8 - curr_target_dist8
        """
        reward_dist = reward_dist1  # + reward_dist2 + reward_dist3 + reward_dist4

        # Orientation Reward
        # 1
        reward_orien1 = tool.orientation_dot(
            action, (self.target.x - self.snake1.x, self.target.y - self.snake1.y))
        """
        # 2
        len2 = np.sqrt((action[2]**2 + action[3]**2))
        reward_orien2 = (action[0]/len2) * (self.snake2.x - self.target2[0])/curr_target_dist2 + \
            action[1]/len2 * (self.snake2.y - self.target2[1]
                              )/curr_target_dist2

        # 3
        len3 = np.sqrt((action[4]**2 + action[5]**2))
        reward_orien3 = (action[0]/len3) * (self.snake3.x - self.target3[0])/curr_target_dist3 + \
            action[1]/len3 * (self.snake3.y - self.target3[1]
                              )/curr_target_dist3

        # 4
        len4 = np.sqrt((action[6]**2 + action[7]**2))
        reward_orien4 = (action[0]/len4) * (self.snake4.x - self.target4[0])/curr_target_dist4 + \
            action[1]/len4 * (self.snake4.y - self.target4[1]
                              )/curr_target_dist4
        """
        reward_orien = reward_orien1  # + reward_orien2 + reward_orien3 + reward_orien4

        # Action Penalty
        reward_act = -0.1 if np.sqrt(action[0]**2 + action[1]**2) < 0.5 else 0

        # Total Reward

        print('reward_dist:{:.2f}|reward_orien:{:.2f}|reward_act:{:.2f}'.format(
            0.5*reward_dist, reward_orien, reward_act))

        print('x:{:.2f}|y:{:.2f}'.format(self.snake1.x, self.snake1.y))

        reward = 0.5*reward_dist + reward_orien + reward_act
        # Terminal State
        done = False

        if curr_target_dist < 10:
            """and curr_target_dist2 < 20 and curr_target_dist3 < 20 and curr_target_dist4 < 20"""
            reward = 20
            done = True
        # out of bound
        if self.snake1.x < -10 or self.snake1.x > 72 or self.snake1.y < -10 or self.snake1.y > 60:
            reward = -20
            done = True

        # State Constructor
        self.target_dist = curr_target_dist
        """
        print('target:({:.2f},{:.2f})|SCP:({:.2f},{:.2f})|curr_target_dist:{:.2f}'.format(
            self.target.x, self.target.y, self.snake1.x, self.snake1.y, self.target_dist))
        print('reward_dist:{:.2f}|reward_orien:{:.2f}|reward_act:{:.2f}'.format(
            0.5*reward_dist, reward_orien, reward_act))
        """
        state_next = self._construct_state(self.img, self.snake1)
        """, self.snake2, self.snake3, self.snake4"""
        return state_next, reward, done

    def render(self, gui=True):
        img_ = self.img.copy()*255

        if(self.snake1.x > self.w):
            draw_x = self.w
        elif(self.snake1.x < 0):
            draw_x = 0
        else:
            draw_x = self.snake1.x

        if(self.snake1.y > self.w):
            draw_y = self.w
        elif(self.snake1.y < 0):
            draw_y = 0
        else:
            draw_y = self.snake1.y

        # draw SCP1
        cv2.circle(img_, (int(draw_x), int(
            draw_y)), 5, (0, 0, 255), 3)
        cv2.putText(img_, "SCP:1", (int(draw_x), int(draw_y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.1, (0, 0, 255), 1, cv2.LINE_AA)
        """
        # draw SCP2
        cv2.circle(img_, (int(self.snake2.x), int(
            self.snake2.y)), 5, (0, 255, 255), 3)
        cv2.putText(img_, "SCP:2", (int(self.snake2.x), int(self.snake2.y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.1, (0, 0, 255), 1, cv2.LINE_AA)

        # draw SCP3
        cv2.circle(img_, (int(self.snake3.x), int(
            self.snake3.y)), 5, (255, 0, 255), 3)
        cv2.putText(img_, "SCP:3", (int(self.snake3.x), int(self.snake3.y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.1, (0, 0, 255), 1, cv2.LINE_AA)

        # draw SCP4
        cv2.circle(img_, (int(self.snake4.x), int(
            self.snake4.y)), 5, (0, 0, 0), 3)
        cv2.putText(img_, "SCP:4", (int(self.snake4.x), int(self.snake4.y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.1, (0, 0, 255), 1, cv2.LINE_AA)
        """
        # draw targer1
        cv2.circle(img_, (int(self.target.x), int(
            self.target.y)), 5, (0, 0, 255), 3)
        # cv2.putText(img_, "target:1", (self.target1.x, self.target1.y), cv2.FONT_HERSHEY_SIMPLEX,
        # 0.1, (0, 0, 0), 1, cv2.LINE_AA)
        """
        # draw targer2
        cv2.circle(img_, (int(self.target2[0]), int(
            self.target2[1])), 5, (0, 255, 255), 3)
        # cv2.putText(img_, "target:2", (self.target2[0], self.target2[1]), cv2.FONT_HERSHEY_SIMPLEX,
        # 0.1, (0, 0, 0), 1, cv2.LINE_AA)

        # draw targer3
        cv2.circle(img_, (int(self.target3[0]), int(
            self.target3[1])), 5, (255, 0, 255), 3)
        # cv2.putText(img_, "target:3", (self.target3[0], self.target3[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 0), 1, cv2.LINE_AA)

        # draw targer4
        cv2.circle(img_, (int(self.target4[0]), int(
            self.target4[1])), 5, (0, 0, 0), 3)
        # cv2.putText(img_, "target:4", (self.target4[0], self.target4[1]), cv2.FONT_HERSHEY_SIMPLEX,0.1, (0, 0, 0), 1, cv2.LINE_AA)
        """
        if gui:
            cv2.imshow("Snake Segmentation", img_)
            k = cv2.waitKey(1)
        return img_

    def _construct_state(self, img, SCP1):
        state = []
        state.append(img)
        state.append(SCP1.x)
        state.append(SCP1.y)
        """
        state.append(SCP2.x/self.w)
        state.append(SCP2.y/self.w)
        state.append(SCP3.x/self.w)
        state.append(SCP3.y/self.w)
        state.append(SCP4.x/self.w)
        state.append(SCP4.y/self.w)
        """
        return state


if __name__ == "__main__":
    env = SnakeEnv()
    for i in range(10):
        env.initialize()
        while(True):
            action = 2*np.random.random(2)-1
            sn, r, end = env.step(action)
            print(sn[20:23], r, end)
            print(len(sn))
            env.render()
            if end:
                break
