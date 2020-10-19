from Snake_Env import SnakeEnv
import sac
import models
import numpy as np
import os
import rl_eval

batch_size = 8
eval_eps = 2
rl_core = sac.SAC(
    model=[models.PolicyNetGaussian, models.QNet],
    n_actions=2,
    learning_rate=[0.0001, 0.0001],
    reward_decay=0.99,
    memory_size=10000,
    batch_size=batch_size,
    alpha=0.1,
    auto_entropy_tuning=True)

is_train = True
render = True
load_model = False
'''
is_train = False
render = True
load_model = True
'''
img_path = "image/example.png"
gif_path = "out/"
model_path = "save/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

if load_model:
    print("Load model ...", model_path)
    rl_core.save_load_model("load", model_path)

if __name__ == "__main__":
    env = SnakeEnv(path=img_path)
    total_step = 0
    max_success_rate = 0
    success_count = 0
    for eps in range(1001):
        state = env.initialize()
        step = 0
        loss_a = loss_c = 0.
        acc_reward = 0.

        while(True):
            # Choose action and run
            if is_train:
                action = rl_core.choose_action(state, eval=False)
            else:
                action = rl_core.choose_action(state, eval=True)
            state_next, reward, done = env.step(action)
            end = 0 if done else 1
            rl_core.store_transition(state, action, reward, state_next, end)

            # Render environment
            im = env.render(gui=render)
            # Learn the model
            loss_a = loss_c = 0.
            if total_step > batch_size and is_train:
                loss_a, loss_c = rl_core.learn()
            step += 1
            total_step += 1

            # Print information
            acc_reward += reward
            print('\rEps:{:3d} /{:4d} /{:6d}| action_x:{:+.2f},action_y:{:+.2f}| R:{:+.2f}| Loss:[A>{:+.2f} C>{:+.2f}]| Alpha: {:.3f}| R_total:{:.2f}  '
                  .format(eps, step, total_step, action[0], action[1], reward, loss_a, loss_c, rl_core.alpha, acc_reward), end='')

            state = state_next.copy()
            if done or step > 400:
                # Count the successful times
                if reward > 5:
                    success_count += 1
                break

        if eps > 0 and eps % eval_eps == 0:
            # Sucess rate
            success_rate = success_count / eval_eps
            success_count = 0
            # Save the best model
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                if is_train:
                    print("Save model to " + model_path)
                    rl_core.save_load_model("save", model_path)
            print("Success Rate (current/max):",
                  success_rate, "/", max_success_rate)
            # output GIF
            rl_eval.run(rl_core, total_eps=4, img_path=img_path,
                        gif_path=gif_path, gif_name="sac_"+str(eps).zfill(4)+".gif")
