## Position and Orientation with Dictionary Observation

## Modules to Import
import gymnasium as gym
from gymnasium import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
from gym_fracture.envs import utils #calculate_distances, make_scene, getStarts, getGoal, check_done, get_new_pose, unpack_action,fingertip_distance, visualize_contact_forces, world_to_local
from gym_fracture.envs import env_utils
from scipy.spatial.transform import Rotation as R
import wandb
#from gym_fracture.envs.spring_damper import SpringDamper
from gym_fracture.envs.createligament import make_ligament

class fracturesurgery_env(gym.Env):
    def __init__(
        self,
        render_mode=None,
        reward_type='sparse',
        distance_threshold_pos=0.005,
        distance_threshold_ori=0.05,
        max_steps=50,
        obs_type='dict',
        goal_type='random',
        dv=0.05,
        action_type='rot_vec',
        horizon='variable',
    ):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.dv = dv
        self.max_steps = max_steps
        self.action_type = action_type
        self.horizon = horizon
        self.success_threshold = 0.6
        self.episodes_done = 0
        self.output_force = np.float32(0)
        self.goal_range_low = np.zeros(3)
        self.goal_range_high = np.zeros(3)
        self.goal_ori_low = np.zeros(3)
        self.goal_ori_high = np.zeros(3)
        self.current_step = 0
        self.pos_distance = 0.0
        self.angle = 0.0
        self.distance_threshold_pos = distance_threshold_pos
        self.distance_threshold_ori = distance_threshold_ori
        self.pitch = 0.0
        self.n = 0

        metadata = {"render_modes": ["human", "direct"]}
        if self.render_mode == 'human':
            p.connect(p.GUI, options="--background_color_red=0.9686--background_color_blue=0.79216--background_color_green=0.7882")
        else:
            p.connect(p.DIRECT)
        self.connected = True
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=87, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
        #p.getCameraImage(1000, 800)
        
        env_utils.set_observation_space(self)

        # Action space
        env_utils.set__action_space(self)
        # --- Contact helpers -------------------------------------------------
    

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == 'sparse':
            # Handle ori_only case
            if self.action_type == 'ori_only':
                reward = env_utils.compute_reward_sparse_ori(self, achieved_goal, desired_goal, info)

            # Handle pos_only case
            elif self.action_type == 'pos_only':
                reward = env_utils.compute_reward_sparse_pos(self, achieved_goal, desired_goal, info)

            # Handle general case (position + orientation)
            elif self.action_type == 'euler':
                reward = env_utils.compute_reward_sparse_euler(self, achieved_goal, desired_goal, info)

        elif self.reward_type != 'sparse':
            reward = env_utils.compute_reward_dense(self, achieved_goal, desired_goal, info)
        return reward
       

    def reset(self, seed=None, options=None):
        self.n += 1
        self.output_force = np.float32(0)
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        

        p.setTimeStep(1./5000.)
        #while p.isConnected():
        
        self.current_step = 0
        utils.make_scene(self)
        fracturestart, fractureorientationDeg, legstartpos = utils.getStarts(self)
        if isinstance(self.goal_type, str):
            utils.getGoal(self, fracturestart, fractureorientationDeg)
        else:
            self.goal_pos = np.array(fracturestart.copy())
            self.goal_ori = np.array(self.goal_type)
            self.goal_range_low = fracturestart - [0.0125, 0.01, 0.003]
            self.goal_range_high = fracturestart + [0.0125, 0.02, 0.003]
            self.goal_ori_low = np.radians(fractureorientationDeg - [15, 5, 15])
            self.goal_ori_high = np.radians(fractureorientationDeg + [15, 5, 15])

        currentDir = os.path.dirname(os.path.abspath(__file__))
        leg_path = os.path.join(currentDir, "Assets/legankle.urdf")
        foot_path = os.path.join(currentDir, "Assets/footpin.urdf")
        footorientation = p.getQuaternionFromEuler([0, 0, 90/180*np.pi])
        for _i in range(100):
            p.stepSimulation()
        legorientation = p.getQuaternionFromEuler([-90/180*np.pi, 0, 0])
        self.leg = p.loadURDF(leg_path,
                        basePosition =legstartpos,
                        baseOrientation = legorientation,
                        globalScaling = 1.0,
                        useFixedBase = 1)
        p.changeDynamics(self.leg, 1, mass = 0.5, lateralFriction=0.1)
        self.objectUid = p.loadURDF(foot_path, basePosition=fracturestart, 
                                    baseOrientation=footorientation, useFixedBase=1,
                                     globalScaling=1)
        p.changeDynamics(self.objectUid, -1, mass=0.1, lateralFriction=5)
        p.changeDynamics(self.objectUid, 0, mass=0.1, lateralFriction=0.5)
        #print(p.getAABB(self.objectUid,-1))       
        p.addUserDebugText('b', p.getAABB(self.objectUid,-1)[0], textColorRGB=[1, 0, 0], textSize=1) 
        
        
        p.stepSimulation()
        target_positions = np.array([0.0, 0.0])
        forces = [1, 1]
        for _ in range(500):
            p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
            p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
            p.stepSimulation()
            time.sleep(1./500)  # Remove for speed
        #p.resetJointState(self.pandaUid, 9, target_positions[0])
        #p.resetJointState(self.pandaUid, 10, target_positions[1])
            #time.sleep(1.)  # Remove for speed
        p.changeDynamics(self.pandaUid, 9, lateralFriction=5.0)
        p.changeDynamics(self.pandaUid, 10, lateralFriction=5.0)
        #time.sleep(10)
        child_9in_parent_pos, child_9in_parent_orn = utils.local_coords(self,9)
        child_10in_parent_pos, child_10in_parent_orn = utils.local_coords(self,10)

        # c9id = p.createConstraint(
        #     parentBodyUniqueId=self.pandaUid,
        #     parentLinkIndex=9,
        #     childBodyUniqueId=self.objectUid,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=child_9in_parent_pos,
        #     parentFrameOrientation=child_9in_parent_orn,
        #     childFramePosition=[0.0,0.0,0],
        #     childFrameOrientation=[0.0,0.0,0,1]
        # )
        # c10id = p.createConstraint(
        #     parentBodyUniqueId=self.pandaUid,
        #     parentLinkIndex=10,
        #     childBodyUniqueId=self.objectUid,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=child_10in_parent_pos,
        #     parentFrameOrientation=child_10in_parent_orn,
        #     childFramePosition=[0.0,0.0,0],
        #     childFrameOrientation=[0.0,0.0,0,1]
        # )

        for _ in range(10):
            p.stepSimulation()
        p.setGravity(0, 0, -9.8)
        self.target_position = np.concatenate((self.goal_pos, self.goal_ori))
        
        # Dummy visual shape for goal marker
        
        
        # goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
        #                   basePosition=self.goal_pos, baseOrientation=self.goal_ori)

        point_a = [0.019166, 0.051439, 0.003369]
        point_b= [-0.068064,-0.03,-0.007159]
        point_c = [ 0.027233,-0.040717,-0.011423]
        point_d = [-0.065358,-0.023195,-0.005576]
        
        make_ligament(self,"cloth_Id1", self.objectUid, self.leg, point_c, point_d,orientation=p.getQuaternionFromEuler([0, 0, 90/180*np.pi]), scale =0.9)
        make_ligament(self, "cloth_Id2", self.objectUid, self.leg, point_a, point_b,orientation=p.getQuaternionFromEuler([0, 0, 305/180*np.pi]), scale =0.9)
        
        [p.enableJointForceTorqueSensor(self.pandaUid, joint, enableSensor=True) for joint in range(p.getNumJoints(self.pandaUid))]
        p.enableJointForceTorqueSensor(self.objectUid, 0, enableSensor=True)
        for _ in range(50):
            p.stepSimulation()
        #print('On to Stepping')
        initialpos = p.getLinkState(self.pandaUid, 11)[0]
        initialor = p.getLinkState(self.pandaUid, 11)[1]
        #initialholdObject = len(p.getContactPoints(self.pandaUid, self.objectUid))
        self.dist = utils.fingertip_distance(self.pandaUid, 9, 10)
        # use helper to get 0/1 contact flags
        left_contact = utils.contact_flag(self, 9)
        right_contact = utils.contact_flag(self, 10)

        initialisHolding = utils.is_holding(self, left_contact, right_contact, self.dist)
        initialvel = p.getLinkState(self.pandaUid, 11, 1)[6]
        initialJointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
        initialJointVelocities = [p.getJointState(self.pandaUid, i)[1] for i in range(9)]
        self.pos_distance, self.angle = utils.calculate_distances(self, initialpos, initialor, self.goal_pos, self.goal_ori)
        initialisHolding = int(initialisHolding)
        initial_force = utils.visualize_contact_forces(self,self.pandaUid, self.objectUid)

        env_utils.set_observation(self, initialpos, initialor, 
                                               initialvel, initialJointPoses, 
                                               initialJointVelocities, initial_force,left_contact,
                                               self.dist, self.angle, right_contact, 
                                               self.dist, initialisHolding)
        
        
        return self.state, {}

    

    def step(self, action):
        #print(action)
        self.current_step += 1
        #print(f"Step: {self.current_step}")
        dx, dy, dz, qx, qy, qz, qw, x, y, z = utils.unpack_action(self,action, self.dv)
        mode_map = {
            'rot_vec': 'rot_vec',
            'euler': 'euler',
            'fouractions': 'fouractions',
            'fiveactions': 'fiveactions',
            'quat': 'quat',
            'joint': 'joint',
            'ori_only': 'ori_only',
            'pos_only': 'pos_only'
        }
        mode = mode_map.get(self.action_type, None)

        if self.action_type == 'joint':
            jointPoses, _ = utils.get_new_pose(self,dx, dy, dz, qx, qy, qz, None, mode)
        else:
            newPosition, newOrientation = utils.get_new_pose(self,dx, dy, dz, qx, qy, qz, qw, mode)
            if self.action_type == 'pos_only':
                p.addUserDebugText('NP', newPosition, textColorRGB=[0, 0, 1], textSize=1, lifeTime=0.5)
                jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, maxNumIterations=100, residualThreshold=1e-4)
            else:
                jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, targetOrientation=newOrientation, maxNumIterations=100, residualThreshold=1e-4)
            if np.any(np.isnan(jointPoses)) or np.any(np.abs(jointPoses) > 10):
                print("IK failure, skipping step")
        max_force = [87,87,87,87,12,12,12,20,20]
        #max_force = [8.7,8.7,8.7,8.7,1.2,1.2,1.2,20,20]
        #max_vel = [2.1750,2.1750,2.1750,2.1750,2.6100,2.6100,2.6100,0.2,0.2]
        # for i in range(20):
        #     alpha = (i + 1) / 20.0
        #     jointpose = np.array([p.getJointState(self.pandaUid, j)[0] for j in range(9)])
        #     step = jointpose + alpha * (np.array(jointPoses) - jointpose)
        p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL, list(jointPoses), forces=max_force)#, maxVelocities=max_vel)
        #     p.setJointMotorControlMultiDofArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL, list(step), forces=max_force, maxVelocities=max_vel)
        #     p.stepSimulation()

        #joint_torques = p.calculateInverseDynamics(self.pandaUid, jointPoses, [0.2]*9, [1.0]*9)
        # print(f'Joint Torques Computed: {joint_torques}')
        # #print(f"Joint Torques: {joint_torques}")
        #p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.VELOCITY_CONTROL, targetVelocities=np.zeros(9), forces=np.zeros(9))
        #p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.TORQUE_CONTROL, forces=joint_torques)
        
        for i in range(1000):
            current_joint_positions = [p.getJointState(self.pandaUid, j)[0] for j in range(9)]
            position_errors = np.array(jointPoses) - np.array(current_joint_positions)
            #print('Position Errors:', position_errors)
            position_errors = position_errors[0:7]
            if position_errors.max() < 0.001:
                joint_pose = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
                #print(f'reached at iteration {i}') if i>0 else None
                break
            p.stepSimulation()
            if i == 999:
                print("Max iterations reached without convergence")
        #for _ in range(20):
       # p.stepSimulation()
       # print('Position Errors:', position_errors)
        
            #time.sleep(1./500)  # Remove for speed
        # after stepSimulation
        # current_pos = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
        # current_vel = [p.getJointState(self.pandaUid, i)[1] for i in range(9)]
        # kp = 500
        # kd = 50
        # force_limit = max_force
        # for j in range(9):
        #     err = jointPoses[j] - current_pos[j]            # per-joint
        #     cmd_torque = kp * err - kd * current_vel[j]    # what your controller computes (example)
        #     print(f"j{j}: err={err:.4f}, vel={current_vel[j]:.4f}, cmd_torque={cmd_torque:.1f}, applied_limit={force_limit[j]}")
        #     is_saturated = abs(cmd_torque) > force_limit[j]
        #     print(f"    saturated: {is_saturated}")
        force = utils.visualize_contact_forces(self,self.pandaUid, self.objectUid, scale=0.01, lifeTime=5)
        #print(f'object reaction force {p.getJointState(self.objectUid, 0)[2]}')  # contact force on object
        #print(f'finger 9 {p.getJointState(self.pandaUid, 9)[2]}')  # contact force on left finger
        #print(f'finger 10 {p.getJointState(self.pandaUid, 10)[2]}')  # contact
        # if (force is not None) and force > self.output_force:
        #     self.output_force = force
        self.output_force+=force 
        actualNewPosition = p.getLinkState(self.pandaUid, 11)[0]
        actualNewOrientation = p.getLinkState(self.pandaUid, 11)[1]
        actualNewVelocity = p.getLinkState(self.pandaUid, 11, 1)[6]
        # use helper to get 0/1 contact flags (keeps behaviour identical but centralised)
        left_contact = utils.contact_flag(self, 9)
        right_contact = utils.contact_flag(self, 10)
        dist = utils.fingertip_distance(self.pandaUid, 9, 10)
        
        self.isHolding = utils.is_holding(self, left_contact, right_contact, dist)
        joint_states = [p.getJointState(self.pandaUid, i) for i in range(9)]
        jointPoses = np.array([js[0] for js in joint_states])        # positions
        jointVelocities = np.array([js[1] for js in joint_states])   # velocities
        self.pos_distance, self.angle = utils.calculate_distances(self, actualNewPosition, actualNewOrientation, self.goal_pos, self.goal_ori)
            
        env_utils.set_observation(self, actualNewPosition, actualNewOrientation, 
                                               actualNewVelocity, jointPoses, 
                                               jointVelocities,force, left_contact, 
                                               dist, self.angle, right_contact, 
                                               dist, self.isHolding)
        
        done = env_utils.check_done(self)
        truncated = self.current_step >= self.max_steps and not done
        if done:
            print('yay')
        
        if done or truncated:
            self.output_force = self.output_force / self.current_step
        info = {'is_success': done, 'current_step': self.current_step, 'pos_distance': self.pos_distance, 'angle': self.angle, 'avg_force': self.output_force, 'Holding': self.isHolding}
        
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, info)
        
        return self.state, reward, done, truncated, info

    

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False
