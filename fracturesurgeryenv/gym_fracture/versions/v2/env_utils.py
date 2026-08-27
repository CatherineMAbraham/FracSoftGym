import numpy as np 
from gymnasium import spaces
from gym_fracture.versions.v2 import utils

def set_observation_space(env):
    if env.action_type == 'ori_only':
        obs_shape = 37 
        goal_shape = 5
    elif env.action_type == "pos_only":
        obs_shape = 37  
        goal_shape = 4
    elif env.contact_type == True:
        obs_shape = 38
        goal_shape = 10
    else:
        obs_shape = 38
        goal_shape = 9 ## now we're going to add contact to the goal anyway as a 'dummy' variable, so we can keep the goal shape the same for both contact and non-contact environments
    if env.obs_type == 'dict':
        env.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32)
        })
    else:
        env.observation_space = spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32)

def set__action_space(env):
    if env.action_type == 'ori_only':
        env.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # low=np.array([-0.0007, -0.0013, -0.0014]),
            #  high=np.array([0.001, 0.0015, 0.0014]),
    elif env.action_type == 'pos_only':
        env.action_space = spaces.Box(low =-1, high=1, shape=(3,))
            #low=np.array([-0.005, -0.005, -0.005]),
            #high=np.array([0.005, 0.005, 0.005]),
            #shape=(3,)
        #)
    elif env.action_type == 'fouractions':
        env.action_space = spaces.Box(low=-1, high=1, shape=(4,))
    else:
        env.action_space = spaces.Box(low=-1, high=1, shape=(6,))


def compute_reward_sparse_pos(env, achieved_goal, desired_goal, info):
    if achieved_goal.ndim == 1:
                    env.pos_distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
                    env.isHolding = achieved_goal[3]
                    reward = 0 if (env.pos_distance <= env.distance_threshold_pos and env.isHolding == 1) else -1
    else:
        pos_achieved = achieved_goal[:, :3]
        pos_desired = desired_goal[:, :3]
        env.isHolding = achieved_goal[:, 3]
        env.pos_distance = np.linalg.norm(pos_achieved - pos_desired, axis=1)
        reward = np.where(
            (env.pos_distance <= env.distance_threshold_pos) & (env.isHolding == 1),
            0, -1
        )
    return np.array(reward)

def compute_reward_sparse_ori(env, achieved_goal, desired_goal, info):
    if env.action_type == 'ori_only':
            if achieved_goal.ndim == 1:
                new_ori = achieved_goal[:4]
                goal_ori = desired_goal[:4]
                env.isHolding = achieved_goal[4]
                dot_product = np.clip(np.abs(np.sum(new_ori * goal_ori)), -1.0, 1.0)
                env.angle = 2 * np.arccos(dot_product)
                reward = 0 if (env.angle <= env.distance_threshold_ori and env.isHolding == 1) else -1
            else:
                new_ori = achieved_goal[:, :4]
                goal_ori = desired_goal[:, :4]
                env.isHolding = achieved_goal[:, 4]
                dot_product = np.clip(np.abs(np.sum(new_ori * goal_ori, axis=-1)), -1.0, 1.0)
                env.angle = 2 * np.arccos(dot_product)
                reward = np.where(
                    (env.angle <= env.distance_threshold_ori) & (env.isHolding == 1),
                    0, -1
                )
            return np.array(reward)
    
def compute_reward_sparse_euler(env, achieved_goal, desired_goal, info):
    if achieved_goal.ndim == 1:   
            pos_achieved, angle_achieved = achieved_goal[:3], achieved_goal[3:7]
            pos_desired, angle_desired = desired_goal[:3], desired_goal[3:7]
            env.pos_distance, env.angle = utils.calculate_distances(env, pos_achieved, angle_achieved, pos_desired, angle_desired)
            env.isHolding = achieved_goal[7]
            env.force = achieved_goal[8]
            #env.contact = achieved_goal[9]
            reward = 0 if (
                env.pos_distance <= env.distance_threshold_pos and
                env.angle <= env.distance_threshold_ori and 
                env.isHolding == 1 and
                env.force <= env.max_force# and
               # env.contact == 0
            ) else -1
    else:
        pos_achieved, angle_achieved = achieved_goal[:, :3], achieved_goal[:, 3:7]
        pos_desired, angle_desired = desired_goal[:, :3], desired_goal[:, 3:7]
        env.pos_distance, env.angle = utils.calculate_distances(env, pos_achieved, angle_achieved, pos_desired, angle_desired)
        env.isHolding = achieved_goal[:, 7]
        env.force = achieved_goal[:, 8]
        
        #env.contact = achieved_goal[:, 9]
        reward = np.where(
            (env.pos_distance <= env.distance_threshold_pos) &
            (env.angle <= env.distance_threshold_ori) &
            (env.isHolding == 1) & 
            (env.force <= env.max_force),# &
            #(env.contact == 0),
            0, -1)
        
    return np.array(reward)

def compute_reward_sparse_euler_contact(env, achieved_goal, desired_goal, info):
    if achieved_goal.ndim == 1:   
            pos_achieved, angle_achieved = achieved_goal[:3], achieved_goal[3:7]
            pos_desired, angle_desired = desired_goal[:3], desired_goal[3:7]
            env.pos_distance, env.angle = utils.calculate_distances(env, pos_achieved, angle_achieved, pos_desired, angle_desired)
            env.isHolding = achieved_goal[7]
            env.force = achieved_goal[8]
            env.contact = achieved_goal[9]
            #contact_reward = env.contact_alpha * env.contact
            reward = 0 if (
                env.pos_distance <= env.distance_threshold_pos and
                env.angle <= env.distance_threshold_ori and 
                env.isHolding == 1 and
                env.force <= env.max_force and
                env.contact == 0
            ) else -1
    else:
        pos_achieved, angle_achieved = achieved_goal[:, :3], achieved_goal[:, 3:7]
        pos_desired, angle_desired = desired_goal[:, :3], desired_goal[:, 3:7]
        env.pos_distance, env.angle = utils.calculate_distances(env, pos_achieved, angle_achieved, pos_desired, angle_desired)
        env.isHolding = achieved_goal[:, 7]
        env.force = achieved_goal[:, 8]
        env.contact = achieved_goal[:, 9]
        reward = np.where(
            (env.pos_distance <= env.distance_threshold_pos) &
            (env.angle <= env.distance_threshold_ori) &
            (env.isHolding == 1) & 
            (env.force <= env.max_force) &
            (env.contact == 0),
            0, -1)
        
    return np.array(reward)

def compute_reward_dense(env, achieved_goal, desired_goal, info):
    hold = 0.1 if env.isHolding == 0 else 0
    d1 = env.pos_distance + env.angle
    d2 = env.pos_distance + env.angle
    d_pos = np.float32(env.pos_distance)
    rewardDistance = np.exp(-0.1 * env.pos_distance)
    rewardOrientation = np.exp(-0.1 * env.angle)
    e = rewardDistance + rewardOrientation
    if env.reward_type == 'dense' and env.action_type == 'pos_only':
        return -d_pos
    elif env.reward_type == 'dense_1' and env.horizon == 'variable':
        #print(f'Pos Distance: {env.pos_distance}, Angle: {env.angle}, Holding Penalty: {hold}, Reward: {-d1}')
        return -d1
    elif env.reward_type == 'dense_2':
        return -(d2 + hold)
    elif env.reward_type == 'dense_1' and env.horizon == 'fixed':
        return -d1 + e
    
def set_observation(env, pos, ori, vel, jointPoses, jointVelocities, 
                    force,contact,contact_distance,max_contact_force,position, angle,left_contact, right_contact, dist, isHolding):
    if env.action_type == 'ori_only':
        observation = np.concatenate([
        np.array(pos),
        np.array(ori),
        np.array(vel),
        np.array(jointPoses),
        np.array(jointVelocities),
        np.array([force]),
        np.array([contact]),
        np.array([contact_distance]),
        np.array([max_contact_force]),
        np.array([angle]),
        np.array([left_contact]),
        np.array([right_contact]),
        np.array([dist]),
        np.array([isHolding])
    ])  # Total: 31 elements
    elif env.action_type == 'pos_only':
          observation = np.concatenate([
                np.array(pos),
                np.array(ori),
                np.array(vel),
                np.array(jointPoses),
                np.array(jointVelocities),
                np.array([force]),
                np.array([contact]),
                np.array([contact_distance]),
                np.array([max_contact_force]),
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
            np.array([contact_distance]),
            np.array([max_contact_force]),
            np.array([angle]),
            np.array([left_contact]),
            np.array([right_contact]),
            np.array([dist]),
            np.array([isHolding])
        ])    

    desired_force = [2.5]
    object_contact = [0] 
    if env.action_type == 'ori_only':
        env.achieved_goal = np.array(list(ori) +[isHolding]+[force])#+[env.contact])
        env.desired_goal = np.array(list(env.goal_ori) + [1]+desired_force)#+object_contact)
    elif env.action_type == 'pos_only':
        env.achieved_goal = np.array(list(pos) + [isHolding]+[force])#+[env.contact])
        env.desired_goal = np.array(list(env.goal_pos) + [1]+desired_force)#+object_contact)
    elif env.contact_type == 1:
        env.achieved_goal = np.array(list(pos) + list(ori) + [isHolding]+[force]+[env.anycontact])#+[env.contact])
        env.desired_goal = np.array(list(env.target_position) + [1]+desired_force +object_contact)
    else:
        env.achieved_goal = np.array(list(pos) + list(ori) + [isHolding]+[force])#we're going to set this as 'success' so we don't really look for it but it keeps the shape the same.
        env.desired_goal = np.array(list(env.target_position) + [1]+desired_force)

    if env.obs_type == 'dict':
        observation_dict = {
            "observation": observation.astype(np.float32),
            "achieved_goal": env.achieved_goal.astype(np.float32),
            "desired_goal": env.desired_goal.astype(np.float32),
        }
        env.state = observation_dict
    else:
        env.state = observation.astype(np.float32)
    
def check_done(env):
        if env.horizon == 'variable' and env.action_type not in ['ori_only', 'pos_only'] and env.contact_type == 0:
            #print('checking done',{env.maximum_force},{env.filerted_force}, {env.max_force})
            return env.pos_distance <= env.distance_threshold_pos and env.angle <= env.distance_threshold_ori and env.isHolding == 1 and env.maximum_force <=env.max_force #and env.anycontact == 0
        elif env.horizon == 'variable' and env.action_type not in ['ori_only', 'pos_only'] and env.contact_type == 1:
            return env.pos_distance <= env.distance_threshold_pos and env.angle <= env.distance_threshold_ori and env.isHolding == 1 and env.maximum_force <=env.max_force and env.anycontact == 0 
        elif env.horizon == 'fixed' and env.action_type == 'ori_only':
            return env.angle <= env.distance_threshold_ori and env.isHolding == 1 and env.current_step >= env.max_steps
        elif env.horizon == 'fixed' and env.action_type == 'pos_only':
            return env.pos_distance <= env.distance_threshold_pos and env.isHolding == 1 and env.current_step >= env.max_steps
        elif env.action_type == 'ori_only':
            return env.angle <= env.distance_threshold_ori and env.isHolding == 1 and env.maximum_force <=env.max_force
        elif env.action_type == 'pos_only':
            return env.pos_distance <= env.distance_threshold_pos and env.isHolding == 1 and env.maximum_force <=env.max_force
        else:
            return env.pos_distance <= env.distance_threshold_pos and env.angle <= env.distance_threshold_ori and env.isHolding == 1 and env.maximum_force <=env.max_force and env.anycontact == 0