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
from gym_fracture.versions.v2.Assets.transformation_matrices import get_goal_from_proximal_pose
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
        youngs_modulus_type = 'None', #None, 'eval_mode', 'testing'
        randomise_ligs = False,
        randomise_start = False,
        patient = None,
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

        
        #p.setTimeStep(1/500)

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
        #fracturestart = np.array([0.3618006205558777, -0.102467754304409027, 0.07800002501010895]) #252
        #([0.3518006205558777, -0.10467754304409027, 0.07190002501010895])
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
        foot_path = os.path.join(current_dir, f"Assets/Patient{self.patient}/distal_copy.urdf")

        footorientation = p.getQuaternionFromEuler([90/180*np.pi, 0, 0])
       
        #leg_orientation = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        #footorientation = np.array([0.6992329955101013, 0.3331104815006256, 0.29179978370666504, 0.5612159967422485])
        self.foot = p.loadURDF(foot_path, basePosition=fracturestart, 
                                   baseOrientation=footorientation, 
                                    useFixedBase=0,
                                     globalScaling=1)
        p.setCollisionFilterGroupMask(self.foot, 1, collisionFilterGroup=0, collisionFilterMask=0)
        dynamics.change_foot_dynamics(self)
        dynamics.change_robot_dynamics(self)
        #time.sleep(10)
        finger_force_n = 5 if self.soft_tissue=='soft' else 5
        p.setCollisionFilterPair(self.pandaUid, self.foot, 9,1, 1)
        p.setCollisionFilterPair(self.pandaUid, self.foot, 10,1, 1)
        for _ in range(100):
            p.setJointMotorControl2(self.pandaUid, 9, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force_n)
            p.setJointMotorControl2(self.pandaUid, 10, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force_n)
            p.stepSimulation()
            #time.sleep(1./240)  # Remove for speed
        
        ##
        #print(p.getLinkState(self.pandaUid, 11))
        #difference = np.array([0.03,0.00,0.0]) 132
        #difference = np.array([0.01,0.07,0.01]) #102
        difference = np.array([0,0.09,0])
        # don't overwrite `self.foot` (body id); read link state into local vars
        foot = np.array(p.getLinkState(self.foot, 1,computeForwardKinematics=True)[0])
        print('Foot position:', foot)
        foot_ori = np.array(p.getLinkState(self.foot, 1,computeForwardKinematics=True)[1])
        print('Foot position:', foot)
        print('Foot orientation (quaternion):', foot_ori)
        #leg_start = foot - difference
    #     if self.patient == 102:
    #         #leg_start = np.array([0.35736772418022156, -0.11651839315891266, 0.07902605086565018])
    #         #leg_orientation = np.array([0.7066577672958374, 0.0034424655605107546, 0.003453752724453807, 0.7075387239456177])
    #         leg_start = np.array([0.3572990596294403,-0.11652351915836334 , 0.0789283737540245]) #
    #         leg_start = leg_start - [0,0.008,0]
    #         leg_orientation = np.array([0.7071092128753662, 0.0030444269068539143, 0.003227325389161706, 0.7070903182029724])
    #         #leg_start = ([0.36055922508239746, -0.03996943682432175, -0.0015069395303726196])
    #         #leg_orientation = ([0.7071092128753662, 0.0030444269068539143, 0.003227325389161706, 0.7070903182029724])
    #     elif self.patient == 126:
    #         #leg_start = np.array([0.374905, -0.059665, 0.051185])#np.array([0.33901602029800415, -0.050294697284698486, 0.09938723593950272])
    #         leg_start = np.array([0.33908429741859436, -0.03595512732863426, 0.06553331017494202])
    #         leg_orientation = np.array([-0.02957511693239212, -0.0026056983042508364, -0.08343382924795151, 0.9960709810256958 ])
    #         #leg_orientation = np.array([0.062514, 0.683191, -0.725171, 0.058895])#np.array([0.6844638586044312, 0.06375731527805328, -0.0574415884912014, 0.7239784002304077])
    #         ninety_deg = p.getQuaternionFromEuler([90/180*np.pi,np.pi, 0])
    #         #_,leg_orientation = p.multiplyTransforms(
    #          #               positionA=[0, 0, 0], orientationA=ninety_deg,        # Apply 90 deg first (or on left)
    #           #              positionB=[0, 0, 0], orientationB=leg_orientation   # Existing rotation
    #            #         )
    #     elif self.patient == 198:
    # #         leg_start = np.array([0.3223761320114136, -0.106806, 0.058319])#([0.3228972852230072, -0.10798908770084381, 0.057827770709991455])
    # #         leg_start = [0.3229373097419739, -0.10821224749088287, 0.057968251407146454]#foot - np.array([0.079925,-5.1e-4,0.092568])
    # #         leg_orientation = np.array([0.0006749040330760181, -0.00021293869940564036, 0.006133391056209803, 0.9999809265136719])#([-0.00017769925761967897, 2.267319541715551e-05, 0.003869270207360387, 0.9999924898147583]) #0.7087432742118835, 0.004880446940660477, 0.0048079658299684525, 0.7054332494735718])
    #         T_matrix = np.array([[ 1.        ,  0.        ,  0.        , -0.03472044],
    #    [ 0.        ,  1.        ,  0.        , -0.04793354],
    #    [ 0.        ,  0.        ,  1.        , -0.01723594],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])


    # #         position = T_matrix[0:3, 3].tolist()
            
    # #         # 2. Extract 3x3 Rotation Sub-matrix
    # #         rotation_matrix = T_matrix[0:3, 0:3]
    # #         # pybullet.multiplyTransforms expects orientations as quaternions [x,y,z,w]
    # #         # convert the 3x3 rotation matrix to a quaternion using scipy Rotation
    # #         quat_b = R.from_matrix(rotation_matrix).as_quat().tolist()
    # #         pos_a, ori_a = foot, foot_ori
    # #         new_pos, new_ori = p.multiplyTransforms(pos_a, ori_a, position, quat_b)
    # #         leg_start = np.array(new_pos)
    # #         leg_orientation = np.array(new_ori)
    #         self.target_position, leg_start, leg_orientation = get_goal_and_proximal_transforms(self, self.patient,foot,foot_ori)
    #     elif self.patient == 132:
    #         leg_start = foot-np.array([-0.000852,-0.001007,0.023137])#np.array([0.3382095694541931, -0.054506316781044006, 0.095221608877182])#0.3379322588443756, -0.039989080280065536, 0.07018909603357315])#0.33799970149993896, -0.03988508880138397, 0.07011495530605316])
    #         #leg_orientation = np.array([0.002232487080618739, -2.0870984371867962e-06, 0.006996737327426672, 0.9999729990959167])#([0.002232487080618739, -2.0870984371867962e-06, 0.006996737327426672, 0.9999729990959167])#0.7071067690849304, 6.547016262459238e-10, -6.624191195570006e-10, 0.7071067690849304])
    #         #ninety_deg = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        #if self.patient is not None:
         #   self.target_position, leg_start, leg_orientation = get_goal_from_proximal_pose(self, self.patient, foot, foot_ori)
        #else:
        #leg_start = foot - difference
        leg_orientation = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        leg_start = fracturestart-np.array([0.0,0.09,0])#np.array([0.35706788301467896, -0.1598062852025032, 0.07526329159736633])
        ##rotate foot by 90 deg too
        foot_ori = p.multiplyTransforms([0, 0, 0], leg_orientation, [0, 0, 0], foot_ori)[1]
        #new_foot = p.resetBasePositionAndOrientation(self.foot, foot, foot_ori)
        ##Load Leg
        self.leg = p.loadURDF(leg_path,
                        basePosition =leg_start,
                        baseOrientation = leg_orientation,
                        globalScaling = 1.0,
                        useFixedBase = 1)
        #time.sleep(100)
        leg_orientation = p.getBasePositionAndOrientation(self.leg)[1]
        leg_start = p.getBasePositionAndOrientation(self.leg)[0]
       # print('Leg position:', leg_start)
        #print('Leg orientation (quaternion):', np.rad2deg(p.getEulerFromQuaternion(leg_orientation)))
        dynamics.change_leg_dynamics(self)
        p.changeVisualShape(self.leg, -1, rgbaColor=[0.8, 0.8, 0.8, 1])  
        p.setCollisionFilterGroupMask(self.foot, 1, collisionFilterGroup=0, collisionFilterMask=0)
        p.setCollisionFilterGroupMask(self.leg, -1, collisionFilterGroup=0, collisionFilterMask=0)
        ##Settle
        #print('Settling the simulation...') 
        #time.sleep(100)
        for _ in range(10):
            p.stepSimulation()
        
        
        p.setGravity(0, 0, -9.81)
        initial_or = p.getLinkState(self.pandaUid, 11)[1]
        #print('Initial end-effector orientation (quaternion):', initial_or)
        #pose_valid = utils.is_goal_configuration_valid(self,self.goal_pos, self.goal_ori)
        if isinstance(self.goal_type, str):
            utils.getGoal(self, fracturestart, fractureorientationDeg) ## do i want to increase the range of goals?
            self.target_position = np.concatenate((self.goal_pos, self.goal_ori))
            #print(self.target_position)
        else:
            #self.goal_pos = np.array(self.goal_type[0:3])
            goal_ori = np.array(self.goal_type[3:7])
            #self.goal_ori = goal_ori#np.array(p.getQuaternionFromEuler(goal_ori))
            ori_change = p.getQuaternionFromEuler([9.08/180*np.pi,0, 0])#np.array([0.99999994124027, 0.0003417183131417258, 2.7327058643906894e-05, -1.132662577527209e-06])
           # self.goal_ori = np.array(p.multiplyTransforms([0, 0, 0], ori_change, [0, 0, 0], p.getLinkState(self.pandaUid, 11)[1])[1])
            # self.target_position, pos, orientation = get_goal_from_proximal_pose(self, 
            #                                                                      self.patient,
            #                                                                      leg_start,
            #                                                                      leg_orientation,
            #                                                                      foot,
            #                                                                      foot_ori)
            self.goal_pos = np.array([ 0.32180062,-0.09246775, 0.15800003]) - np.array([0.005,-0.005,0.00])#([0.32180062,-0.09246775, 0.15800003]) - np.array([0.005,0.005,0.005])#np.array([0.32180062,-0.09246775, 0.15800003]) - np.array([0.005,0.005,0.005])
            self.goal_ori = np.array([0.9999999728200057, 0.00023313980271510995, -8.89660707914592e-08, 2.4108688676344187e-06])#([2.81656109e-04, -2.81431908e-04,  7.06825125e-01,  7.07388213e-01])
            self.target_position = np.concatenate((self.goal_pos, self.goal_ori))#np.array([ 0.32180062,-0.09246775, 0.15800003,0.9999999728200057, 0.00023313980271510995, -8.89660707914592e-08, 2.4108688676344187e-06])#2.81656109e-04, -2.81431908e-04,  7.06825125e-01,  7.07388213e-01])
            #self.target_position = np.array([0.3148845586806017, -0.06000234856812478, 0.16351743256405377,0.0008761560662228061, 0.0010616809069047318, 0.6413949793268746, 0.7672096100013853])
            #print(self.goal_ori)
            # 4x4 transformation matrix: wrap rows in an outer list
            # T = np.array([[0.999459, -0.017879, -0.027606, -0.003794],
            #               [0.017331, 0.999650, -0.019972, 0.000030],
            #               [0.027953, 0.019482, 0.999419, 0.002779],
            #               [0.000000, 0.000000, 0.000000, 1.000000]])
            # position = T[0:3, 3].tolist()

            # # 2. Extract 3x3 Rotation Sub-matrix
            # rotation_matrix = T[0:3, 0:3]
            # # pybullet.multiplyTransforms expects orientations as quaternions [x,y,z,w]
            # # convert the 3x3 rotation matrix to a quaternion using scipy Rotation
            # quat_b = R.from_matrix(rotation_matrix).as_quat().tolist()
            # pos_a, ori_a = p.getLinkState(self.pandaUid, 11)[0], p.getLinkState(self.pandaUid, 11)[1]
            # new_pos, new_ori = p.multiplyTransforms(pos_a, ori_a, position, quat_b)
            # self.goal_pos = np.array(new_pos)
            # self.goal_ori = np.array(new_ori)
            #print('robot start pos',p.getLinkState(self.pandaUid, 11)[0])
            #print('Goal Position:', self.goal_pos)
            #self.goal_pos = p.getLinkState(self.pandaUid, 11)[0] - np.array([0,0,0.0062299])
            #self.goal_pos = np.array([0.3528081774711609, -0.10790444910526276, 0.1682744324207306])
            #self.goal_ori = np.array([np.float64(0.999999941240567), np.float64(0.0003417174451147411), np.float64(2.732704526842606e-05), np.float64(-1.1326338845288588e-06)])
            #self.target_position = np.concatenate((self.goal_pos, self.goal_ori))#l pose valid:', pose_valid)
        # Dummy visual shape for goal marker
            valid = utils.is_goal_configuration_valid(self,self.goal_pos, self.goal_ori)
           # time.sleep(10)
        #utils.is_goal_configuration_valid(self,self.goal_pos, self.goal_ori)
        goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
                            basePosition=self.goal_pos, baseOrientation=self.goal_ori)
        #time.sleep(1)
        
       ## Enable force/torque sensors
        [p.enableJointForceTorqueSensor(self.pandaUid, joint, enableSensor=True) for joint in range(p.getNumJoints(self.pandaUid))]
        p.enableJointForceTorqueSensor(self.foot, 0, enableSensor=True) # Load cell joint 
        
        ##
        
        # 1. Print visual frame vs base frame position in PyBullet
        # Draw RGB coordinate axes at the base frame of the proximal bone
        p.addUserDebugParameter("show_axes", 0) # optional GUI trigger

        # Draw axis for Distal
        p.addUserDebugLine([0,0,0], [0.05, 0, 0], [1, 0, 0], parentObjectUniqueId=self.foot, parentLinkIndex=1)
        p.addUserDebugLine([0,0,0], [0, 0.05, 0], [0, 1, 0], parentObjectUniqueId=self.foot, parentLinkIndex=1)
        p.addUserDebugLine([0,0,0], [0, 0, 0.05], [0, 0, 1], parentObjectUniqueId=self.foot, parentLinkIndex=1)

        # Draw axis for Proximal
        p.addUserDebugLine([0,0,0], [0.05, 0, 0], [1, 0, 0], parentObjectUniqueId=self.leg)
        p.addUserDebugLine([0,0,0], [0, 0.05, 0], [0, 1, 0], parentObjectUniqueId=self.leg)
        p.addUserDebugLine([0,0,0], [0, 0, 0.05], [0, 0, 1], parentObjectUniqueId=self.leg)
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
        #print(initial_or)
        # self.contact = int(bool(p.getContactPoints(self.foot, self.leg,1,-1)))
        # if int(bool(p.getContactPoints(self.foot, self.leg,1,-1))) == 1:
        #     self.contact = 1 if (p.getContactPoints(self.foot, self.leg,1,-1))[8]<self.distance_threshold_pos else 0
        # Query PyBullet once and store the tuple of contact points
        contacts = p.getContactPoints(self.foot, self.leg, 1, -1)

        # Check if contacts exist AND if any contact distance is below your threshold
        self.contact = 1 if (contacts and any(pt[8] < 0 for pt in contacts)) else 0
        if self.contact ==1:
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
        # pos_range = 0.2
        # ori_range = np.deg2rad(360)
        # foot_pos = p.getLinkState(self.foot, 0, computeForwardKinematics=True)[0]
        # foot_ori = p.getEulerFromQuaternion(p.getLinkState(self.foot, 0, computeForwardKinematics=True)[1])
        # x_slider = p.addUserDebugParameter("X", foot_pos[0] - pos_range, foot_pos[0] + pos_range, foot_pos[0])
        # y_slider = p.addUserDebugParameter("Y", foot_pos[1] - pos_range, foot_pos[1] + pos_range, foot_pos[1])
        # z_slider = p.addUserDebugParameter("Z", foot_pos[2] - pos_range, foot_pos[2] + pos_range, foot_pos[2])
        # yaw_slider = p.addUserDebugParameter("Yaw", foot_ori[2] - ori_range, foot_ori[2] + ori_range, foot_ori[2])
        # roll_slider = p.addUserDebugParameter("Roll", foot_ori[0] - ori_range, foot_ori[0] + ori_range, foot_ori[0])
        # pitch_slider = p.addUserDebugParameter("Pitch", foot_ori[1] - ori_range, foot_ori[1] + ori_range, foot_ori[1])
        # print("Adjust the sliders in the PyBullet GUI. Press Ctrl+C in terminal to output values.")

        # try:
        #     while True:
        #         # Read current slider values
        #         x = p.readUserDebugParameter(x_slider)
        #         y = p.readUserDebugParameter(y_slider)
        #         z = p.readUserDebugParameter(z_slider)
        #         yaw = p.readUserDebugParameter(yaw_slider)
        #         roll = p.readUserDebugParameter(roll_slider)
        #         pitch = p.readUserDebugParameter(pitch_slider)
        #         # Update object orientation/position
        #         orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        #         p.resetBasePositionAndOrientation(self.foot, [x, y, z], orn)
                
        #         p.stepSimulation()
        #         time.sleep(1/240.)

        # except KeyboardInterrupt:
        #     # Extract final position & quaternion
        #     pos, orn = p.getBasePositionAndOrientation(self.foot)
        #     euler = p.getEulerFromQuaternion(orn)
            
        #     print("\n--- Extracted Pose ---")
        #     print(f"Position (x, y, z): {pos}")
        #     print(f"Quaternion (x, y, z, w): {orn}")
        #     print(f"Euler (roll, pitch, yaw): {euler}")
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
            self.band = new_band2.ElasticBand(bodyA=self.foot, linkA= -1,
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
        #print(f"New Position: {new_Position}, New Orientation: {new_Orientation}")
        #new_Position = np.array([0.32091317, -0.07630774,  0.15682939])
        #new_Orientation = np.array([0.98674402,  0.09496498, -0.13048336,  0.01708737])
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
        # if self.contact:
        #     print('Contact within {0:.4f} mm'.format(p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1)[0][8] * 1000))
        if self.contact:
            contact_points = p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1)
            if contact_points and contact_points[0][8] < -0.0005: ## check contact distance to avoid false positives from close proximity, currently set to -1mm
                #print('Contact within {}'.format(p.getContactPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1)[0][8]))
                #print('Contact!!, goal distance: ', self.pos_distance, 'angle: ', self.angle, 'goal:', self.target_position)
               # self.contact = contact
                total_normal_force = sum(pt[9] for pt in contact_points)  # Sum of normal forces at all contact points

                # 2. Maximum collision force at any single contact point
                max_contact_force = max(pt[9] for pt in contact_points)

                # 3. Complete force vector accounting for normal and friction forces
                total_force_magnitude = 0.0
                for pt in contact_points:
                    fn = pt[9]   # Normal force
                    f_fric1 = pt[10] # Friction force 1
                    f_fric2 = pt[12] # Friction force 2
                    # Resultant 3D force magnitude for this point
                    f_point = np.sqrt(fn**2 + f_fric1**2 + f_fric2**2)
                    total_force_magnitude += f_point
                if max_contact_force > 1:  # Threshold to avoid false positives
                #print(f"Total Normal Force: {total_normal_force:.2f} N")
                #print(f"Max Point Force: {max_contact_force:.2f} N")
                    self.anycontact = 1
                    self.contact = 1
                else:
                    self.anycontact = 0
                    self.contact = 0
            
       
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
        
        if self.test and (self.filerted_force >= self.max_force or self.isHolding ==0):
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
        
        if done:
            print('yay')
        elif truncated:
            print(f'truncated {self.filerted_force},{self.pos_distance},{self.angle}')
        
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
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=87, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
        ##
        #p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=1.1, yaw=87, pitch=-20, roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        p.getCameraImage(10, 10,viewMatrix=matrix,projectionMatrix=projection)  # Warm up the renderer to prevent first-step lag

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False
