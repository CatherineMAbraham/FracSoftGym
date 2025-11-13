import numpy as np 
from gymnasium import spaces
from gym_fracture.envs import utils

def set_observation_space(self):
    if self.action_type == 'ori_only':
        obs_shape = 33  # Reduced from 31
        goal_shape = 5
    elif self.action_type == "pos_only":
        obs_shape = 33  # Reduced from 31
        goal_shape = 4
    else:
        obs_shape = 35
        goal_shape = 8
    if self.obs_type == 'dict':
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32)
        })
    else:
        self.observation_space = spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32)

def set__action_space(self):
    if self.action_type == 'quat':
            self.action_space = spaces.Box(
                low=np.array([-0.005, -0.005, -0.005, -0.05, -0.05, -0.05, 0.9]),
                high=np.array([0.005, 0.005, 0.005, 0.03, 0.03, 0.03, 1]),
                shape=(7,)
            )
    elif self.action_type == 'joint':
        mean_diff = np.array([
            5.344368461085916e-06, -1.7254826772920976e-06, -5.652514741346842e-06,
            -1.7459863701878556e-06, 3.352876319352558e-05, 2.7700700928288767e-06,
            -3.7586469804521616e-05, -1.1339311907951961e-06, 1.4051723622599417e-06
        ])
        low = mean_diff - 0.005
        high = mean_diff + 0.005
        self.action_space = spaces.Box(low=low, high=high, shape=(9,))
    elif self.action_type == 'ori_only':
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # low=np.array([-0.0007, -0.0013, -0.0014]),
            #  high=np.array([0.001, 0.0015, 0.0014]),
    elif self.action_type == 'pos_only':
        self.action_space = spaces.Box(low =-1, high=1, shape=(3,))
            #low=np.array([-0.005, -0.005, -0.005]),
            #high=np.array([0.005, 0.005, 0.005]),
            #shape=(3,)
        #)
    elif self.action_type == 'fiveactions':
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))
    elif self.action_type == 'fouractions':
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
    else:
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,))


def compute_reward_sparse_pos(self, achieved_goal, desired_goal, info):
    if achieved_goal.ndim == 1:
                    self.pos_distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
                    self.isHolding = achieved_goal[3]
                    reward = 0 if (self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1) else -1
    else:
        pos_achieved = achieved_goal[:, :3]
        pos_desired = desired_goal[:, :3]
        self.isHolding = achieved_goal[:, 3]
        self.pos_distance = np.linalg.norm(pos_achieved - pos_desired, axis=1)
        reward = np.where(
            (self.pos_distance <= self.distance_threshold_pos) & (self.isHolding == 1),
            0, -1
        )
    return np.array(reward)

def compute_reward_sparse_ori(self, achieved_goal, desired_goal, info):
    if self.action_type == 'ori_only':
            if achieved_goal.ndim == 1:
                new_ori = achieved_goal[:4]
                goal_ori = desired_goal[:4]
                self.isHolding = achieved_goal[4]
                dot_product = np.clip(np.abs(np.sum(new_ori * goal_ori)), -1.0, 1.0)
                self.angle = 2 * np.arccos(dot_product)
                reward = 0 if (self.angle <= self.distance_threshold_ori and self.isHolding == 1) else -1
            else:
                new_ori = achieved_goal[:, :4]
                goal_ori = desired_goal[:, :4]
                self.isHolding = achieved_goal[:, 4]
                dot_product = np.clip(np.abs(np.sum(new_ori * goal_ori, axis=-1)), -1.0, 1.0)
                self.angle = 2 * np.arccos(dot_product)
                reward = np.where(
                    (self.angle <= self.distance_threshold_ori) & (self.isHolding == 1),
                    0, -1
                )
            return np.array(reward)
    
def compute_reward_sparse_euler(self, achieved_goal, desired_goal, info):
    if achieved_goal.ndim == 1:
            pos_achieved, angle_achieved = achieved_goal[:3], achieved_goal[3:7]
            pos_desired, angle_desired = desired_goal[:3], desired_goal[3:7]
            self.pos_distance, self.angle = utils.calculate_distances(self, pos_achieved, angle_achieved, pos_desired, angle_desired)
            self.isHolding = achieved_goal[7]
            reward = 0 if (
                self.pos_distance <= self.distance_threshold_pos and
                self.angle <= self.distance_threshold_ori and
                self.isHolding == 1
            ) else -1
    else:
        pos_achieved, angle_achieved = achieved_goal[:, :3], achieved_goal[:, 3:7]
        pos_desired, angle_desired = desired_goal[:, :3], desired_goal[:, 3:7]
        self.pos_distance, self.angle = utils.calculate_distances(self, pos_achieved, angle_achieved, pos_desired, angle_desired)
        self.isHolding = achieved_goal[:, 7]
        reward = np.where(
            (self.pos_distance <= self.distance_threshold_pos) &
            (self.angle <= self.distance_threshold_ori) &
            (self.isHolding == 1),
            0, -1
        )
    return np.array(reward)

def compute_reward_dense(self, achieved_goal, desired_goal, info):
    d1 = self.pos_distance + self.angle
    d2 = self.pos_distance + self.angle
    d_pos = np.float32(self.pos_distance)
    rewardDistance = np.exp(-0.1 * self.pos_distance)
    rewardOrientation = np.exp(-0.1 * self.angle)
    e = rewardDistance + rewardOrientation
    if self.reward_type == 'dense' and self.action_type == 'pos_only':
        return -d_pos
    elif self.reward_type == 'dense_1' and self.horizon == 'variable':
        return -d1
    elif self.reward_type == 'dense_1' and self.horizon == 'fixed':
        return -d1 + e
    
def set_observation(self, pos, ori, vel, jointPoses, jointVelocities, left_contact,position, angle, right_contact, dist, isHolding):
    if self.action_type == 'ori_only':
        observation = np.concatenate([
        np.array(pos),
        np.array(ori),
        np.array(vel),
        np.array(jointPoses),
        np.array(jointVelocities),
        np.array([self.angle]),
        np.array([self.left_contact]),
        np.array([self.right_contact]),
        np.array([self.dist]),
        np.array([isHolding])
    ])  # Total: 31 elements
    elif self.action_type == 'pos_only':
          observation = np.concatenate([
                np.array(pos),
                np.array(ori),
                np.array(vel),
                np.array(jointPoses),
                np.array(jointVelocities),
                np.array([position]),
                np.array([left_contact]),
                np.array([right_contact]),
                np.array([dist]),
                np.array([isHolding])
            ])  # Total: 31 elements
    else: 
        observation = np.concatenate([
            np.array(pos),
            np.array(ori),
            np.array(vel),
            np.array(jointPoses),
            np.array(jointVelocities),
            np.array([position]),
            np.array([angle]),
            np.array([left_contact]),
            np.array([right_contact]),
            np.array([dist]),
            np.array([isHolding])
        ])    

    if self.action_type == 'ori_only':
        self.achieved_goal = np.array(list(ori) +[isHolding])
        self.desired_goal = np.array(list(self.goal_ori) + [1])
    elif self.action_type == 'pos_only':
        self.achieved_goal = np.array(list(pos) + [isHolding])
        self.desired_goal = np.array(list(self.goal_pos) + [1])
    else:
        self.achieved_goal = np.array(list(pos) + list(ori) + [isHolding])
        self.desired_goal = np.array(list(self.target_position) + [1])

    if self.obs_type == 'dict':
        observation_dict = {
            "observation": observation.astype(np.float32),
            "achieved_goal": self.achieved_goal.astype(np.float32),
            "desired_goal": self.desired_goal.astype(np.float32),
        }
        self.state = observation_dict
    else:
        self.state = observation.astype(np.float32)
    
def check_done(self):
        if self.horizon == 'variable' and self.action_type not in ['ori_only', 'pos_only']:
            return self.pos_distance <= self.distance_threshold_pos and self.angle <= self.distance_threshold_ori and self.isHolding == 1
        elif self.horizon == 'fixed' and self.action_type == 'ori_only':
            return self.angle <= self.distance_threshold_ori and self.isHolding == 1 and self.current_step >= self.max_steps
        elif self.horizon == 'fixed' and self.action_type == 'pos_only':
            return self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1 and self.current_step >= self.max_steps
        elif self.action_type == 'ori_only':
            return self.angle <= self.distance_threshold_ori and self.isHolding == 1
        elif self.action_type == 'pos_only':
            return self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1
        else:
            return self.pos_distance <= self.distance_threshold_pos and self.angle <= self.distance_threshold_ori and self.isHolding == 1