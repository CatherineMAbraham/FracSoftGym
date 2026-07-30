## Position and Orientation with Dictionary Observation

## Modules to Import
from turtle import width

import gymnasium as gym
from gymnasium import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
#from gym_fracture.envs import spring_system, utils #calculate_distances, make_scene, getStarts, getGoal, check_done, get_new_pose, unpack_action,fingertip_distance, visualize_contact_forces, world_to_local
from gym_fracture.versions.v2 import env_utils, utils
from gym_fracture.versions.v2 import dynamics, new_band,new_band2,createligament
from scipy.spatial.transform import Rotation as R
import wandb
#from gym_fracture.envs.spring_damper import SpringDamper
#from gym_fracture.envs.createligament import make_ligament,radius_spring
#from gym_fracture.envs.multispring import create_ligament_chain, apply_axial_springs

class fracturesurgery_env_v2(gym.Env):
    def __init__(
        self,
        render_mode=None,
        reward_type='sparse',
        distance_threshold_pos=0.005,
        distance_threshold_ori=np.deg2rad(5),
        max_steps=100,
        obs_type='dict',
        goal_type='random',
        dt=0.001,
        dr=0.1,
        action_type='euler',
        horizon='variable',
        softtissue='spring',
        vtk_file = None,
        start_pos = 'home',
        maxforce = 3.5,
        number_of_springs = 3,
        contact_type = 0,
        youngs_modulus = 1e6,
        youngs_modulus_type = 'testing', #None, 'eval_mode', 'testing'
        randomise_ligs = False,
        randomise_start = False,
        patient = 110,
        width = 0.005,
        test = False
    ):
        """Gym Environment for training agents to perform fracture reduction surgery with a robotic manipulator.
        Args:
            render_mode (str): The mode to render the environment. Options are 'human' or None.
            reward_type (str): The type of reward to use. Options are 'sparse' or 'dense'.
            distance_threshold_pos (float): The distance threshold for considering the position goal achieved.
            distance_threshold_ori (float): The angle threshold (in degrees) for considering the orientation goal achieved.
            max_steps (int): The maximum number of steps per episode.
            obs_type (str): The type of observation to return. Options are 'dict' or 'flat'.
            goal_type (str or list): The type of goal to use. If 'random', the goal will be randomly generated within a specified range. If a list of 6 values is provided, it will be used as the fixed goal (first 3 values for position and last 3 values for orientation in Euler angles).
            dt (float): The scale of translational action.
            dr (float): The scale of rotational action.
            action_type (str): The type of action to use. Options are 'euler' (position and orientation), 'fouractions' (up/down/left/right), 'ori_only' (orientation only), 'pos_only' (position only).
            horizon (str): The episode horizon. Options are 'variable' (episode ends when goal is achieved) or 'fixed' (episode ends after max_steps).
            softtissue (str): The type of soft tissue modeling to use. Options are 'soft' (deformable soft body) or 'spring' (spring-based soft tissue) or 'none' (no soft tissue).
            start_pos (str): The starting position of the end-effector. Options are 'home' (default) or 'extended'.
            maxforce (float): The maximum force threshold for termination during testing.
            number_of_springs (int): The number of springs to use in the spring-based soft tissue model.
            contact_type (int): The type of contact modeling to use. Options are 0 (no contact), 1 (contact-based reward shaping).
            youngs_modulus (float): The Young's modulus to use for the soft tissue modeling.
            youngs_modulus_type (str): The type of Young's modulus to use. Options are 'testing', 'eval_mode', or None.
            test (bool): Whether the environment is being used for testing (True) or training (False). During testing, episodes will"""
       
        metadata = {"render_modes": ["human", None]}
        ## Initialise variables from input args 
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.dt = dt
        self.dr = dr
        self.max_steps = max_steps
        self.action_type = action_type
        self.horizon = horizon
        self.soft_tissue = softtissue
        self.vtk_file = vtk_file
        self.distance_threshold_pos = distance_threshold_pos
        self.distance_threshold_ori = distance_threshold_ori
        self.start_pos = start_pos # 'home' or 'extended'
        self.max_force = maxforce
        self.contact_type = contact_type
        self.number_of_springs = number_of_springs
        self.young_modulus = youngs_modulus
        self.young_modulus_type = youngs_modulus_type
        self.patient = patient
        self.test= test
        self.width = width
        self.randomise_ligs = randomise_ligs
        self.randomise_start = randomise_start

        ## Initialise variables to 0 
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
        self.n = 0
        self.anycontact = 0
        self.filerted_force = 0
        self.eval_count = 0
        self.not_valid_count = 0
        self.goal_gen_count = 0
        
        ## Rendering setup
         ## need to fix this and add a render function, keep getting a warning about it
        
        self.render()

        
        #p.setTimeStep(1/240)

        ##Obs and Action Space setup
        if self.action_type not in ['euler', 'fouractions','ori_only', 'pos_only']: 
            raise ValueError(f"Invalid action_type: {self.action_type}")
        
        env_utils.set_observation_space(self)

        # Action space
        env_utils.set__action_space(self)
        
           
        #self.ligament = None 
    
    ## Reward Function : Needed here for HER compatibility
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == 'sparse':
            # Handle ori_only case
            if self.action_type == 'ori_only':
                reward = env_utils.compute_reward_sparse_ori(self, achieved_goal, desired_goal, info)

            # Handle pos_only case
            elif self.action_type == 'pos_only':
                reward = env_utils.compute_reward_sparse_pos(self, achieved_goal, desired_goal, info)
            elif self.action_type == 'euler' and self.contact_type ==1:
                reward = env_utils.compute_reward_sparse_euler_contact(self, achieved_goal, desired_goal, info)

            # Handle general case (position + orientation)
            elif self.action_type == 'euler' or self.action_type == 'fouractions':
                reward = env_utils.compute_reward_sparse_euler(self, achieved_goal, desired_goal, info)

        elif self.reward_type != 'sparse':
            reward = env_utils.compute_reward_dense(self, achieved_goal, desired_goal, info)
        return reward
    ##

    ##Reset Function            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        active_seed = options.get("seed", seed) if options else seed
        np.random.seed(active_seed)
        ##Counters 
        # self.n += 1
        self.current_step = 0 ##THESE NEED TO BE RESET HERE 
        #self.force = 0
        self.output_force = 0
        self.anycontact = 0
        #   ##This is in init? Check in test 
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD) ##Needed for FEM
        
        self.band_id = None
        self.force_text_id = None
        
        ##Make Scene
        utils.make_scene(self)
        
        fracturestart, fractureorientationDeg = utils.getStarts(self)
        
            # self.goal_pos = np.array(fracturestart.copy())
            # self.goal_ori = np.array(self.goal_type)
            # self.goal_range_low = fracturestart - [0.0125, 0.01, 0.003]
            # self.goal_range_high = fracturestart + [0.0125, 0.02, 0.003]
            # self.goal_ori_low = np.radians(fractureorientationDeg - [15, 5, 15])
            # self.goal_ori_high = np.radians(fractureorientationDeg + [15, 5, 15])

        #print(f"Fracture start position: {fracturestart}, Fracture orientation (deg): {fractureorientationDeg}")
        ##
        ## check targer for possible collision 
        
        ##Load Objects
        current_dir = os.path.dirname(os.path.abspath(__file__))
        leg_path = os.path.join(current_dir, f"Assets/Patient{self.patient}/proximal.urdf")
        foot_path = os.path.join(current_dir, f"Assets/Patient{self.patient}/distal.urdf")

        #footorientation = p.getQuaternionFromEuler([90/180*np.pi, 0, 0])
       
        leg_orientation = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        
        self.foot = p.loadURDF(foot_path, basePosition=fracturestart, 
         #                          baseOrientation=footorientation, 
                                    useFixedBase=0,
                                     globalScaling=1)
        
        dynamics.change_foot_dynamics(self)
        dynamics.change_robot_dynamics(self)
        
        finger_force_n = 5 if self.soft_tissue=='soft' else 5
        for _ in range(100):
            p.setJointMotorControl2(self.pandaUid, 9, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force_n)
            p.setJointMotorControl2(self.pandaUid, 10, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force_n)
            p.stepSimulation()
            #time.sleep(1./240)  # Remove for speed
        
        ##
        #print(p.getLinkState(self.pandaUid, 11))
        difference = np.array([0.0,0.09,0])
        foot = p.getLinkState(self.foot, 1)[0]
        foot_ori = p.getLinkState(self.foot, 1)[1]
        #print('Foot position:', foot)
        #print('Foot orientation (quaternion):', foot_ori)
        leg_start=foot - difference
        if self.patient == 102:
            leg_start = np.array([0.35736772418022156, -0.11651839315891266, 0.07902605086565018])
            leg_orientation = np.array([0.7066577672958374, 0.0034424655605107546, 0.003453752724453807, 0.7075387239456177])
        elif self.patient == 126:
            leg_start = np.array([0.374905, -0.059665, 0.051185])#np.array([0.33901602029800415, -0.050294697284698486, 0.09938723593950272])
            leg_orientation = np.array([0.062514, 0.683191, -0.725171, 0.058895])#np.array([0.6844638586044312, 0.06375731527805328, -0.0574415884912014, 0.7239784002304077])
            ninety_deg = p.getQuaternionFromEuler([90/180*np.pi,np.pi, 0])
            _,leg_orientation = p.multiplyTransforms(
                            positionA=[0, 0, 0], orientationA=ninety_deg,        # Apply 90 deg first (or on left)
                            positionB=[0, 0, 0], orientationB=leg_orientation   # Existing rotation
                        )
        elif self.patient == 198:
            leg_start = np.array([0.3223761320114136, -0.106806, 0.058319])#([0.3228972852230072, -0.10798908770084381, 0.057827770709991455])
            leg_orientation = np.array([0.00022264904691837728, 3.8162688724696636e-05, 0.004301907029002905, 0.999990701675415])#([-0.00017769925761967897, 2.267319541715551e-05, 0.003869270207360387, 0.9999924898147583]) #0.7087432742118835, 0.004880446940660477, 0.0048079658299684525, 0.7054332494735718])
        elif self.patient == 132:
            leg_start = foot-np.array([-0.000852,-0.001007,0.023137])#np.array([0.3382095694541931, -0.054506316781044006, 0.095221608877182])#0.3379322588443756, -0.039989080280065536, 0.07018909603357315])#0.33799970149993896, -0.03988508880138397, 0.07011495530605316])
            #leg_orientation = np.array([0.002232487080618739, -2.0870984371867962e-06, 0.006996737327426672, 0.9999729990959167])#([0.002232487080618739, -2.0870984371867962e-06, 0.006996737327426672, 0.9999729990959167])#0.7071067690849304, 6.547016262459238e-10, -6.624191195570006e-10, 0.7071067690849304])
            #ninety_deg = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        else:
            leg_start = foot - difference
            leg_orientation = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        ##Load Leg
        self.leg = p.loadURDF(leg_path,
                        basePosition =leg_start,
                        baseOrientation = leg_orientation,
                        globalScaling = 1.0,
                        useFixedBase = 1)
        #time.sleep(100)
        
        dynamics.change_leg_dynamics(self)
        p.changeVisualShape(self.leg, -1, rgbaColor=[0.8, 0.8, 0.8, 1])  
        p.setCollisionFilterGroupMask(self.foot, 1, collisionFilterGroup=0, collisionFilterMask=0)
        p.setCollisionFilterGroupMask(self.leg, -1, collisionFilterGroup=0, collisionFilterMask=0)
        ##Settle
        #print('Settling the simulation...') 
        for _ in range(10):
            p.stepSimulation()
        
        
        p.setGravity(0, 0, -9.81)
        
        #pose_valid = utils.is_goal_configuration_valid(self,self.goal_pos, self.goal_ori)
        if isinstance(self.goal_type, str):
            utils.getGoal(self, fracturestart, fractureorientationDeg) ## do i want to increase the range of goals?
            self.target_position = np.concatenate((self.goal_pos, self.goal_ori))
            #print(self.target_position)
        else:
            self.goal_pos = np.array(self.goal_type[0:3])
            goal_ori = np.array(self.goal_type[3:7])
            self.goal_ori = goal_ori#np.array(p.getQuaternionFromEuler(goal_ori))
            #print(self.goal_ori)
            self.target_position = np.concatenate((self.goal_pos, self.goal_ori))#l pose valid:', pose_valid)
        # Dummy visual shape for goal marker
        
        #utils.is_goal_configuration_valid(self,self.goal_pos, self.goal_ori)
        goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
                            basePosition=self.goal_pos, baseOrientation=self.goal_ori)
 
        
       ## Enable force/torque sensors
        [p.enableJointForceTorqueSensor(self.pandaUid, joint, enableSensor=True) for joint in range(p.getNumJoints(self.pandaUid))]
        p.enableJointForceTorqueSensor(self.foot, 0, enableSensor=True) # Load cell joint 
        
        ##
        
        
        ##Initial Observation
        initial_pos = p.getLinkState(self.pandaUid, 11)[0]
        initial_or = p.getLinkState(self.pandaUid, 11)[1]
        #initialholdObject = len(p.getContactPoints(self.pandaUid, self.foot))
        self.dist = utils.fingertip_distance(self.pandaUid, 9, 10)
        # use helper to get 0/1 contact flags
        left_contact = utils.contact_flag(self, 9)
        right_contact = utils.contact_flag(self, 10)


        initial_isHolding = utils.is_holding(self, left_contact, right_contact, self.dist)
        initial_vel = p.getLinkState(self.pandaUid, 11, 1)[6]
        initial_Joint_Poses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
        initial_Joint_Velocities = [p.getJointState(self.pandaUid, i)[1] for i in range(9)]
        self.pos_distance, self.angle = utils.calculate_distances(self, initial_pos, initial_or, self.goal_pos, self.goal_ori)
        initial_isHolding = int(initial_isHolding)
        initial_force = p.getJointState(self.foot, 0)[2]  # Joint index 0 is the fixed joint
        initial_force = np.linalg.norm(initial_force[0:3])
        #print('Initial Force:', initial_force)
        #get initial force without normalization
        #initial_f = np.linalg.norm(force)#utils.visualize_contact_forces(self,self.pandaUid, self.foot)

        self.contact = int(bool(p.getContactPoints(self.foot, self.leg,1,-1)))
        if int(bool(p.getContactPoints(self.foot, self.leg,1,-1))) == 1:
            self.contact = 1 if (p.getContactPoints(self.foot, self.leg,1,-1))[8]>self.distance_threshold_pos else 0
            print(f"Contact detected with distance: {(p.getContactPoints(self.foot, self.leg,1,-1))[8]:.4f} m")
        #print((p.getContactPoints(self.foot, self.leg,1,-1)))
        env_utils.set_observation(self, 
                                  initial_pos, 
                                  initial_or, 
                                  initial_vel, 
                                  initial_Joint_Poses, 
                                  initial_Joint_Velocities, 
                                  initial_force,
                                  self.contact,
                                  self.pos_distance,
                                  self.angle,
                                  left_contact,
                                  right_contact,
                                  self.dist, 
                                  initial_isHolding)
        #print(f"Youngs Modulus Type is {self.young_modulus_type}, not using soft tissue in this environment.")
        if self.young_modulus_type =='testing' :
            self.eval_count = 0
            self.young_modulus, self.width = utils.get_youngs_modulus_and_width(self)
        elif self.young_modulus_type == 'None':
            self.young_modulus = self.young_modulus
            self.width = 0.005
            #print(f'Youngs Modulus: {self.young_modulus} Pa, Width: {self.width} m')
        # elif self.young_modulus_type == 'eval_mode':
        #     self.eval_count+=1
        #     print(f"Evaluation count: {self.eval_count}")
        #     if self.eval_count ==1:
        #         young_modulus_options = [1e6 ,1e7,5e6, 1e8]
        #         ## Select a youngs modulus for the eval, making sure to use a different one each time 
        #         self.young_modulus = np.random.choice(young_modulus_options)
        #         width_options = np.arange(0.001, 0.01, 0.001)
        #         self.width = np.random.choice(width_options)
        #         print(f"Selected width for evaluation: {self.young_modulus:.2e},{self.width}")
        #     if self.eval_count >1:
        #         self.young_modulus = self.young_modulus
        #         self.width = self.width
            
        #print(f'Youngs Modulus Type: {self.young_modulus_type}')
        #print(f'Youngs Modulus: {self.young_modulus} Pa, Width: {self.width} m')
        p.setPhysicsEngineParameter(numSolverIterations=10, numSubSteps=5)
        if self.soft_tissue=='soft':
            self.point_b,_ = new_band.ElasticBand._get_pose_vel(self,self.leg, -1,local_offset=[0.01,0.0,-0.01])
            self.point_a,_ = new_band.ElasticBand._get_pose_vel(self,self.foot, 1,local_offset=[0.01,-0.0015,0.04]) ##trial and error to place them 
            self.point_c,_ = new_band.ElasticBand._get_pose_vel(self,self.leg, -1,local_offset=[-0.03,0.0,-0.01])
            self.point_d,_ = new_band.ElasticBand._get_pose_vel(self,self.foot, 1,local_offset=[-0.03,-0.0015,0.04])
            #make_ligament(self,"cloth_Id1", self.foot, self.leg, self.point_c, self.point_d,orientation=p.getQuaternionFromEuler([90/180*np.pi,270/180*np.pi,180/180*np.pi]), scale =1,youngs_modulus=self.young_modulus)
            ligament = createligament.Ligament("cloth_Id2", self.foot, self.leg, 
                                                self.point_a, self.point_b,
                                                orientation=p.getQuaternionFromEuler([90/180*np.pi,270/180*np.pi, 180/180*np.pi]), 
                                                scale=1, 
                                                youngs_modulus=self.young_modulus,vtk_file=self.vtk_file)
            ligament.make_ligament(self, "cloth_Id2", self.foot, 
                          self.leg, self.point_a, 
                          self.point_b,orientation=p.getQuaternionFromEuler([90/180*np.pi,270/180*np.pi , 
                                                    180/180*np.pi]), 
                                                    scale =1, 
                                                    youngs_modulus=self.young_modulus) #0.75
        elif self.soft_tissue=='spring':
            self.band = new_band2.ElasticBand(bodyA=self.foot, linkA= 1,
                                         bodyB=self.leg, linkB= -1,
                                         young_modulus=self.young_modulus,
                                         area=5e-6,
                                         width= self.width,
                                         num_springs=self.number_of_springs, randomize_position=self.randomise_ligs,
                                         randomize_num_ligaments=self.randomise_ligs
                                         )
            
            
        else: 
            
            pass  
       
        #print(p.getClosestPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1,distance=0.5 ))
        #utils.drawAABB(self,self.leg,-1)
        p.setCollisionFilterPair(self.foot,self.leg,1,-1,1) ## Allow collision between foot and leg but not between the soft object, very unstable 
        return self.state, {}

    
    ## Step Function
    def step(self, action):
        self.current_step += 1
        
        ## Unpack Action
        dx, dy, dz, qx, qy, qz, qw, x, y, z = utils.unpack_action(self,action)
        mode_map = {
            'euler': 'euler',
            'fouractions': 'fouractions',
            'ori_only': 'ori_only',
            'pos_only': 'pos_only'
        }
        mode = mode_map.get(self.action_type, None)

        
        new_Position, new_Orientation = utils.get_new_pose(self,dx, dy, dz, qx, qy, qz, qw, mode)
        if self.action_type == 'pos_only':
            jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=new_Position, maxNumIterations=10, residualThreshold=1e-4)
        else:
            jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=new_Position, 
                                                      targetOrientation=new_Orientation, maxNumIterations=10, residualThreshold=1e-4)
            #p.addUserDebugText('NP',newPosition, textSize=1.5)
        if np.any(np.isnan(jointPoses)) or np.any(np.abs(jointPoses) > 10):
            print("IK failure, skipping step")
            print(action)
            # Avoid passing NaNs/invalid targets into PyBullet (can segfault)
            # Fallback: use current joint positions for all 9 joints so
            # `setJointMotorControlArray` receives valid targets of the
            # expected length and dtype. alternatively one could `continue`
            # or return early here depending on desired behaviour.
            try:
                jointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
            except Exception:
                # As a last-resort fallback, build a safe zero vector
                jointPoses = [0.0] * 9

        # Set Joint Motors
        max_joint_force = [87,87,87,87,12,12,12,20,20] ##max force for each joint, fingers have lower max force found on urdf 
        
        start_pos = np.array([p.getJointState(self.pandaUid, j)[0] for j in range(9)])
        
        #p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL,targetPositions = jointPoses,forces=max_force)#, maxVelocities=max_vel)
        alpha = 1
        if self.soft_tissue=='spring':
           self.output_force, max_step_force,avg_force,all_mean= utils.smooth_motion(self, jointPoses, start_pos, max_joint_force, numsubsteps=12)
           self.filerted_force = (alpha * avg_force) + ((1 - alpha) * self.filerted_force)
           if self.filerted_force > self.output_force:
                self.output_force = self.filerted_force
        elif self.soft_tissue=='soft':
            self.output_force,max_step_force, avg_force, all_mean = utils.smooth_motion(self, jointPoses, start_pos, max_joint_force, numsubsteps=12)
            self.filerted_force = (alpha * avg_force) + ((1 - alpha) * self.filerted_force)
            if self.filerted_force > self.output_force:
                self.output_force = self.filerted_force
        else: 
            self.output_force,max_step_force, avg_force, all_mean = utils.smooth_motion(self, jointPoses, start_pos, max_joint_force, numsubsteps=12)
            self.filerted_force = (alpha * avg_force) + ((1 - alpha) * self.filerted_force)
            if self.filerted_force > self.output_force:
                self.output_force = self.filerted_force
        # if self.soft_tissue=='soft':
        #     worldA, worldB = createligament.Ligament.radius_spring(self.foot, self.leg,
        #                                                     self.point_a, self.point_b)
        #     stretch = np.linalg.norm(worldA - worldB) 
        
        #print(p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1))
        stretch = np.array(p.getLinkState(self.foot, 1)[0]) - np.array(p.getBasePositionAndOrientation(self.leg)[0])
        stretch = np.linalg.norm(stretch)
        self.contact = int(bool(p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1))) ## check for contact between foot and leg, can adjust distance threshold if needed, currently set to -1mm to avoid false positives from close proximity
        if self.contact:
            if p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1)[0][8] <0: ## check contact distance to avoid false positives from close proximity, currently set to -1mm
                #print('Contact!!')
               # self.contact = contact
                self.anycontact = 1
            
       
        # if self.contact==1:
        #     #p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1)[0][8]}') ## print contact distance for debugging
        #     self.anycontact = 1
            #print('Contact!')
        
        ## Observation Update
        actual_New_Position = p.getLinkState(self.pandaUid, 11)[0]
        actual_New_Orientation = p.getLinkState(self.pandaUid, 11)[1]
        actual_New_Velocity = p.getLinkState(self.pandaUid, 11, 1)[6]
        # use helper to get 0/1 contact flags (keeps behaviour identical but centralised)
        left_contact = utils.contact_flag(self, 9)
        right_contact = utils.contact_flag(self, 10)
        dist = utils.fingertip_distance(self.pandaUid, 9, 10)
        
        self.isHolding = utils.is_holding(self, left_contact, right_contact, dist)
        joint_states = [p.getJointState(self.pandaUid, i) for i in range(9)]
        joint_Poses = np.array([js[0] for js in joint_states])        # positions
        joint_Velocities = np.array([js[1] for js in joint_states])   # velocities
        self.pos_distance, self.angle = utils.calculate_distances(self, actual_New_Position, actual_New_Orientation, self.goal_pos, self.goal_ori)
        #self.capped_force = min(self.filerted_force,200)
        #normalise force instead of cap 
        #self.normalised_force = self.filerted_force / self.maxforce ## for visualization only
        env_utils.set_observation(self, 
                                  actual_New_Position, 
                                  actual_New_Orientation, 
                                  actual_New_Velocity, 
                                  joint_Poses, 
                                  joint_Velocities,
                                  self.filerted_force,
                                  self.contact, 
                                  self.pos_distance,
                                  self.angle,
                                  left_contact,
                                  right_contact, 
                                  dist,  
                                  self.isHolding)
        
        #print('Capped Force: ', self.capped_force,)
        done = env_utils.check_done(self)
        
        if self.test and (self.filerted_force >= 100 or self.isHolding ==0):
            print('Terminating episode due to excessive force during testing.')
            truncated = True
            reward = -100
        else:
            truncated = self.current_step >= self.max_steps and not done
        
        # if done:
        #     print('MaxForce: ', self.output_force, 
        #        'Pos Distance: ', self.pos_distance, 
        #        'Angle: ', self.angle, 
        #        'Holding: ', self.isHolding, 
        #        'Contact: ', self.anycontact)
        
        # if done:
        #     print('yay')
        # elif truncated:
        #     print(f'truncated {self.filerted_force},{self.pos_distance},{self.angle}')
        
        info = {'is_success': done,'truncated': truncated, 'current_step': self.current_step, 
                'pos_distance': self.pos_distance, 
                'angle': self.angle, 'Holding': self.isHolding, 
                'force': self.filerted_force,'contact': self.anycontact,'stretch': stretch,'force_axis_mean': all_mean, 
                'young_modulus': self.young_modulus,
                'width': self.width}#,'force_mag':self.force_magnitude}#,
        #print(stretch,self.output_force)
                #'stretch':stretch,'force_mag':force_mag,'contact': self.anycontact}
        if (not self.test) or (self.filerted_force <= 100):
            reward = self.compute_reward(self.achieved_goal, self.desired_goal, info)
        # else: keep the earlier penalty reward (-100)
        reward = np.float32(reward)
        #print('force: ', self.force, reward)
        #print(self.anycontact)
        return self.state, reward, done, truncated, info

    def render(self) :
        if self.render_mode == 'human':
           # colourpicker = #cbb5b5 # Light pink background
           background_color = (0.203, 0.181, 0.181, 1)  # RGBA values for light pink
           #colour = #e5d1ff rgb(229, 209, 255)
            #rgb(203, 181, 181)
           p.connect(p.GUI, options="--background_color_red=0--background_color_blue=0--background_color_green=0")
        else:
            p.connect(p.DIRECT)
        self.connected = True
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=87, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
        ##
        #p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=1.1, yaw=87, pitch=-20, roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        p.getCameraImage(10, 10,viewMatrix=matrix,projectionMatrix=projection)  # Warm up the renderer to prevent first-step lag

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False
