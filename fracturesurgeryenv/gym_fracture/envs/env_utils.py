import numpy as np 
from gymnasium import spaces
from gym_fracture.envs import utils

def set_observation_space(self):
    if self.action_type == 'ori_only':
        obs_shape = 35  
        goal_shape = 5
    elif self.action_type == "pos_only":
        obs_shape = 35  
        goal_shape = 4
    else:
        obs_shape = 36
        goal_shape = 10
    if self.obs_type == 'dict':
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32)
        })
    else:
        self.observation_space = spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32)

def set__action_space(self):
    if self.action_type == 'ori_only':
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # low=np.array([-0.0007, -0.0013, -0.0014]),
            #  high=np.array([0.001, 0.0015, 0.0014]),
    elif self.action_type == 'pos_only':
        self.action_space = spaces.Box(low =-1, high=1, shape=(3,))
            #low=np.array([-0.005, -0.005, -0.005]),
            #high=np.array([0.005, 0.005, 0.005]),
            #shape=(3,)
        #)
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
            self.force = achieved_goal[8]
            self.contact = achieved_goal[9]
            reward = 0 if (
                self.pos_distance <= self.distance_threshold_pos and
                self.angle <= self.distance_threshold_ori and 
                self.isHolding == 1 and
                self.force <= self.maxforce and
                self.contact == 0
            ) else -1
    else:
        pos_achieved, angle_achieved = achieved_goal[:, :3], achieved_goal[:, 3:7]
        pos_desired, angle_desired = desired_goal[:, :3], desired_goal[:, 3:7]
        self.pos_distance, self.angle = utils.calculate_distances(self, pos_achieved, angle_achieved, pos_desired, angle_desired)
        self.isHolding = achieved_goal[:, 7]
        self.force = achieved_goal[:, 8]
        self.contact = achieved_goal[:, 9]
        reward = np.where(
            (self.pos_distance <= self.distance_threshold_pos) &
            (self.angle <= self.distance_threshold_ori) &
            (self.isHolding == 1) & 
            (self.force <= self.maxforce) &
            (self.contact == 0),
            0, -1
        )
    return np.array(reward)

def compute_reward_dense(self, achieved_goal, desired_goal, info):
    hold = 0.1 if self.isHolding == 0 else 0
    d1 = self.pos_distance + self.angle
    d2 = self.pos_distance + self.angle
    d_pos = np.float32(self.pos_distance)
    rewardDistance = np.exp(-0.1 * self.pos_distance)
    rewardOrientation = np.exp(-0.1 * self.angle)
    e = rewardDistance + rewardOrientation
    if self.reward_type == 'dense' and self.action_type == 'pos_only':
        return -d_pos
    elif self.reward_type == 'dense_1' and self.horizon == 'variable':
        print(f'Pos Distance: {self.pos_distance}, Angle: {self.angle}, Holding Penalty: {hold}, Reward: {-d1}')
        return -d1
    elif self.reward_type == 'dense_2':
        return -(d2 + hold)
    elif self.reward_type == 'dense_1' and self.horizon == 'fixed':
        return -d1 + e
    
def set_observation(self, pos, ori, vel, jointPoses, jointVelocities, force,contact,left_contact,position, angle, right_contact, dist, isHolding):
    if self.action_type == 'ori_only':
        observation = np.concatenate([
        np.array(pos),
        np.array(ori),
        np.array(vel),
        np.array(jointPoses),
        np.array(jointVelocities),
        np.array([force]),
        np.array([contact]),
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
                np.array([force]),
                np.array([contact]),
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
            np.array([force]),
            np.array([contact]),
            np.array([position]),
            np.array([angle]),
            np.array([left_contact]),
            np.array([right_contact]),
            np.array([dist]),
            np.array([isHolding])
        ])    

    desired_force = [2.5]
    object_contact = [0] 
    if self.action_type == 'ori_only':
        self.achieved_goal = np.array(list(ori) +[isHolding]+[force]+[self.contact])
        self.desired_goal = np.array(list(self.goal_ori) + [1]+desired_force+object_contact)
    elif self.action_type == 'pos_only':
        self.achieved_goal = np.array(list(pos) + [isHolding]+[force]+[self.contact])
        self.desired_goal = np.array(list(self.goal_pos) + [1]+desired_force+object_contact)
    else:
        self.achieved_goal = np.array(list(pos) + list(ori) + [isHolding]+[force]+[self.contact])
        self.desired_goal = np.array(list(self.target_position) + [1]+desired_force+object_contact)

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
            return self.pos_distance <= self.distance_threshold_pos and self.angle <= self.distance_threshold_ori and self.isHolding == 1 and self.output_force <=self.maxforce and self.contact == 0
        elif self.horizon == 'fixed' and self.action_type == 'ori_only':
            return self.angle <= self.distance_threshold_ori and self.isHolding == 1 and self.current_step >= self.max_steps
        elif self.horizon == 'fixed' and self.action_type == 'pos_only':
            return self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1 and self.current_step >= self.max_steps
        elif self.action_type == 'ori_only':
            return self.angle <= self.distance_threshold_ori and self.isHolding == 1 and self.output_force <=self.maxforce
        elif self.action_type == 'pos_only':
            return self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1 and self.output_force <=self.maxforce
        else:
            return self.pos_distance <= self.distance_threshold_pos and self.angle <= self.distance_threshold_ori and self.isHolding == 1 and self.output_force <=self.maxforce and self.contact == 0