import random
import time

import numpy as np

import Metaworld.metaworld as metaworld
# from metaworld.policies.sawyer_door_lock_v2_policy import (
#     SawyerDoorLockV2Policy as policy,
# )

from Metaworld.metaworld.policies.sawyer_coffee_button_v2_policy import (
    SawyerCoffeeButtonV2Policy as policy,
)

np.set_printoptions(suppress=True)

seed = 42
env_name = "door-lock-v2"

random.seed(seed)
ml1 = metaworld.MT50(seed=seed)
env = ml1.train_classes[env_name]()
task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
env.set_task(task)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
obs, _ = env.reset()

p = policy()
count = 0
done = False

info = {}

while count < 500 and not done:
    action = p.get_action(obs)
    next_obs, reward, _, _, info = env.step(action)
    # env.render()
    print('next_obs', next_obs)
    print('reward', reward)
    if int(info["success"]) == 1:
        done = True
    obs = next_obs
    time.sleep(0.02)
    count += 1

print(info)
