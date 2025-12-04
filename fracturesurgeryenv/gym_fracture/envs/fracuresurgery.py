## Position and Orientation with Dictionary Observation

## Modules to Import
import gymnasium as gym
from gymnasium import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
from gym_fracture.envs import spring_system, utils #calculate_distances, make_scene, getStarts, getGoal, check_done, get_new_pose, unpack_action,fingertip_distance, visualize_contact_forces, world_to_local
from gym_fracture.envs import env_utils
from scipy.spatial.transform import Rotation as R
import wandb
#from gym_fracture.envs.spring_damper import SpringDamper
from gym_fracture.envs.createligament import make_ligament, make_ligament_rod

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
        softtissue=False,
    ):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.dv = dv
        self.max_steps = max_steps
        self.action_type = action_type
        self.horizon = horizon
        self.softtissue = softtissue
        self.success_threshold = 0.6
        self.episodes_done = 0
        self.force = np.float32(0)
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
        self.max_force = np.float32(0)
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
        

        #p.setTimeStep(1./5000.)
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
        leg_path = os.path.join(currentDir, "Assets/legankle_orig.urdf")
        foot_path = os.path.join(currentDir, "Assets/footpin_orig.urdf")
        footorientation = p.getQuaternionFromEuler([0, 0, 90/180*np.pi])
        # for _i in range(100):
        #     p.stepSimulation()
        legorientation = p.getQuaternionFromEuler([-90/180*np.pi, 0, 0])
        
        self.objectUid = p.loadURDF(foot_path, basePosition=fracturestart, 
                                    baseOrientation=footorientation, 
                                    useFixedBase=0,
                                     globalScaling=1)
        
        p.changeDynamics(self.objectUid, -1, mass=0.276, lateralFriction=5)
        #p.changeDynamics(self.objectUid, 0, mass=0.1, lateralFriction=0.5)
        #print(p.getAABB(self.objectUid,-1))       
        p.addUserDebugText('b', p.getAABB(self.objectUid,-1)[0], textColorRGB=[1, 0, 0], textSize=1) 
        
        # p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        # p.setPhysicsEngineParameter(erp=0.05)
        # p.setPhysicsEngineParameter(contactERP=0.02)
        p.setPhysicsEngineParameter(contactSlop=0.001)
        #p.stepSimulation()
        target_positions = np.array([0.0, 0.0])
        forces = [10,10]
        for _ in range(100):
            # p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
            # p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
            p.setJointMotorControl2(self.pandaUid, 9, p.VELOCITY_CONTROL, targetVelocity=-1, force=5)
            p.setJointMotorControl2(self.pandaUid, 10, p.VELOCITY_CONTROL, targetVelocity=-1, force=5)

            p.stepSimulation()
            #time.sleep(1./500)  # Remove for speed
        pos_9 = p.getJointState(self.pandaUid, 9)[0]
        pos_10 = p.getJointState(self.pandaUid, 10)[0]
        # print(f'Finger 9 position before reset: {pos_9}')
        # print(f'Finger 10 position before reset: {pos_10}')
        # print(p.getJointState(self.pandaUid, 10))
        p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=0, force=1)
        p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=0, force=1)
        p.stepSimulation()
        #p.resetJointState(self.pandaUid, 9, target_positions[0])
        #p.resetJointState(self.pandaUid, 10, target_positions[1])
            #time.sleep(1.)  # Remove for speed
        # for _ in range(500):
        #     p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
        #     p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
        #     p.stepSimulation()
        
        # for _ in range(500):
        #     p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
        #     p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
        #     p.stepSimulation()
        #time.sleep(10)
        p.addUserDebugText('l',legstartpos, textColorRGB=[0, 1, 0], textSize=1)
        self.leg = p.loadURDF(leg_path,
                        basePosition =legstartpos,
                        baseOrientation = legorientation,
                        globalScaling = 1.0,
                        useFixedBase = 1)
        p.changeDynamics(self.leg, 0, mass = 0.1, lateralFriction=0.1)
        #child_9in_parent_pos, child_9in_parent_orn = utils.local_coords(self,9)
        #child_10in_parent_pos, child_10in_parent_orn = utils.local_coords(self,10)

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
        p.changeDynamics(self.pandaUid, 9, lateralFriction=5.0, maxJointVelocity=0.2)
        p.changeDynamics(self.pandaUid, 10, lateralFriction=5.0, maxJointVelocity=0.2)
        # Dummy visual shape for goal marker
        
        
        # goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
        #                   basePosition=self.goal_pos, baseOrientation=self.goal_ori)

        #point_a = [0.0035, 0.05, 0.005]
        #point_b= [-0.035,-0.009,-0.07]
        point_c = [ 0.027233,-0.470717,-0.011423]
        point_d = [-0.065358,-0.023195,-0.005576]
        point_a = [0.029166, 0.045439, 0.003369]
        point_b= [-0.065064,-0.012,-0.007159]
        # point_c = [ 0.027233,-0.070717,-0.011423]
        # point_d = [-0.065358,-0.023195,-0.005576]
        # point_a = [0.029166, 0.051439, 0.003369]
        # point_b= [-0.048064,-0.012,-0.007159]
        # point_c = [ 0.027233,-0.040717,-0.011423]
        # point_d = [-0.065358,-0.023195,-0.005576]
        
        #point_a = [0.019166, 0.051439, 0.003369]
        #point_b= [-0.068064,-0.03,-0.007159]
        #point_c = [ 0.027233,-0.040717,-0.011423]
        #point_d = [-0.065358,-0.023195,-0.005576]
        
        [p.enableJointForceTorqueSensor(self.pandaUid, joint, enableSensor=True) for joint in range(p.getNumJoints(self.pandaUid))]
        p.enableJointForceTorqueSensor(self.objectUid, 0, enableSensor=True)
        for i in range(p.getNumJoints(self.pandaUid)):
            joint_info = p.getJointInfo(self.pandaUid, i)
            #print(f"Joint {i}: {joint_info}")
        max_vel = [2.1750,2.1750,2.1750,2.1750,2.6100,2.6100,2.6100,0.2,0.2]
        #[p.changeDynamics(self.pandaUid, joint, maxJointVelocity=max_vel[joint]) for joint in range(6)]
        # for _ in range(10):
        #     p.stepSimulation()
        #point_a = np.array([-0.04,0.12,-0.12])
        #point_a = np.array([-0.18,0.08,-0.02])
        #point_b = np.array([0,0,0])
        if self.softtissue:
            #make_ligament(self,"cloth_Id1", self.objectUid, self.leg, point_c, point_d,orientation=p.getQuaternionFromEuler([0, 90/180*np.pi, 70/180*np.pi]), scale =1)
            make_ligament(self, "cloth_Id2", self.objectUid, self.leg, point_a, point_b,orientation=p.getQuaternionFromEuler([90/180*np.pi, 90/180*np.pi, 180/180*np.pi]), scale =1) #0.75
            #make_ligament_rod(self.objectUid, self.leg, point_c, point_d, rod_radius=0.0025, rod_mass=0.01, stiffness=5e4)([0, 90/180*np.pi, 298/180*np.pi]) [90/180*np.pi,-15/180*np.pi,90/180*np.pi]90/180*np.pi,90/180*np.pi,0]
        #time.sleep(5)
        p.changeDynamics(self.objectUid, -1, mass=0.1, lateralFriction=1)
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
        #print(p.getJointState(self.pandaUid, 9))
        #print(p.getJointState(self.pandaUid, 10))
        p.changeDynamics(self.pandaUid, 9, jointLowerLimit=0.00, jointUpperLimit=0.004)
        p.changeDynamics(self.pandaUid, 10, jointLowerLimit=0.0, jointUpperLimit=0.0042)
        # link_state = p.getLinkState(self.pandaUid, 11)
        # ee_pos = link_state[0]
        # ee_orn = link_state[1]

        # rot_matrix = p.getMatrixFromQuaternion(ee_orn)
        # x_axis = rot_matrix[0:3]
        # y_axis = rot_matrix[3:6]
        # z_axis = rot_matrix[6:9]

        # scale = 1
        # p.addUserDebugLine(ee_pos, ee_pos + scale * np.array(x_axis), [1, 0, 0], 2)
        # p.addUserDebugLine(ee_pos, ee_pos + scale * np.array(y_axis), [0, 1, 0], 2)
        # p.addUserDebugLine(ee_pos, ee_pos + scale * np.array(z_axis), [0, 0, 1], 2)
        self.band = spring_system.ElasticBand(
                            bodyA=self.objectUid, linkA=0, 
                            bodyB=self.leg, linkB=0,
                            young_modulus=1e6,     # Pa
                            area=5e-6,             # m^2
                            rest_length=0.2,       # m
                            damping_ratio=1.0
                        )
        
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
                #newPosition = np.array(p.getLinkState(self.pandaUid, 11)[0]) + np.array([0.0,0.1,0.0])
                p.addUserDebugText('NP', newPosition, textColorRGB=[0, 0, 1], textSize=1)
                jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, maxNumIterations=100, residualThreshold=1e-4)
            else:
                #newPosition = np.array(action[0:3]) 
                #newOrientation = np.array(action[3:6])
                #newOrientation = p.getQuaternionFromEuler(newOrientation)
                p.addUserDebugText('NP', newPosition, textColorRGB=[0, 0, 1], textSize=1)
                jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, targetOrientation=newOrientation, maxNumIterations=100, residualThreshold=1e-4)
                #print(jointPoses)
            if np.any(np.isnan(jointPoses)) or np.any(np.abs(jointPoses) > 10):
                print("IK failure, skipping step")
                # Avoid passing NaNs/invalid targets into PyBullet (can segfault)
                # Fallback: use current joint positions for all 9 joints so
                # `setJointMotorControlArray` receives valid targets of the
                # expected length and dtype. Alternatively one could `continue`
                # or return early here depending on desired behaviour.
                try:
                    jointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
                except Exception:
                    # As a last-resort fallback, build a safe zero vector
                    jointPoses = [0.0] * 9
        max_force = [8.7,8.7,8.7,8.7,1.2,1.2,1.2,20,20]
        #max_force = [1,1,1,1,1,1,1,10,10]
        max_vel = [2.1750,2.1750,2.1750,2.1750,2.6100,2.6100,2.6100,0.2,0.2]
        desired_pos = np.array(jointPoses)
        current_pos = np.array([p.getJointState(self.pandaUid,j)[0] for j in list(range(9))])
        targetVelocities = utils.compute_target_velocity(
                            desired_pos=desired_pos,
                            current_pos=current_pos,
                            current_vel=[p.getJointState(self.pandaUid,j)[1] for j in list(range(9))],
                            dt=0.05,
                            max_speed=max_vel,
                            Kd=0.001
                        )
        #print('Target Velocities: ', targetVelocities)  
        p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL,targetPositions = jointPoses,targetVelocities=targetVelocities, forces=max_force)#, maxVelocities=max_vel)
#         utils.move_panda_smoothly(self,robot_id=self.pandaUid,
#     joint_indices=list(range(9)),
#     target_positions=jointPoses,
#     max_speeds=max_vel,
#     Kd=0.01,
#     max_force=max_force,
#     dt=0.01,
#     tolerance=1e-2
# )
        p.addUserDebugLine(p.getLinkState(self.leg, 0)[0], p.getLinkState(self.objectUid, 0)[0], [0, 1, 0], 2)
        for _ in range(20):
            self.band.step()
            p.stepSimulation()
            #time.sleep(1.)  # Remove for speed
      
        self.force = utils.visualize_contact_forces(self,self.pandaUid, self.objectUid, scale=0.01, lifeTime=5)
        wrench = spring_system.estimate_ee_force(self, self.pandaUid,11,list(range(9)))
        if self.force != None:
            self.output_force+=self.force 
            if self.force > self.max_force:
                self.max_force = self.force
                #print('New max force: ', self.max_force)
        else :
            self.output_force+=0
            self.force = 0 # without this, force in the obs is nan and then that messes everything up 
        #p.addUserDebugText('L', p.getLinkState(self.leg, 0)[0], textColorRGB=[1, 0, 0], textSize=1)
        forces = [p.getJointState(self.objectUid, j)[2] for j in range(p.getNumJoints(self.objectUid))]
        f = utils.compute_end_effector_force(self.pandaUid, 11, list(range(9)))

        print("EE force:", f[:3])
        print("EE torque:", f[3:])
        print('Forces on object joints: ', forces) 
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
                                               jointVelocities,self.force, left_contact, 
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
        #print('force: ', self.force, reward)
        
        return self.state, reward, done, truncated, info

    

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False
