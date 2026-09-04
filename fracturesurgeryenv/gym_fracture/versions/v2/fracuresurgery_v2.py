## Position and Orientation with Dictionary Observation

## Modules to Import
from collections import deque
from turtle import width

from gym_fracture.versions.v2.Assets import transformation_matrices
import gymnasium as gym
from gymnasium import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
#from gym_fracture.envs import spring_system, utils #calculate_distances, make_scene, getStarts, getGoal, check_done, get_new_pose, unpack_action,fingertip_distance, visualize_contact_forces, world_to_local
from gym_fracture.versions.v2 import createligament, env_utils, utils
from gym_fracture.versions.v2 import dynamics, new_band,new_band2
from scipy.spatial.transform import Rotation as R
import wandb
from gym_fracture.versions.v2.Assets.transformation_matrices import get_goal_from_proximal_pose, get_patient_goal
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
        randomise_num_springs = False,
        randomise_ligs = False,
        randomise_start = False,
        randomise_foot_dynamics= False,
        randomise_sensor_noise=False,
        safe_mode = False,
        patient = None,
        width = 0.005,
        test = False,
        maximum_contact_force_threshold = 0.2
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
        self.maximum_contact_force_threshold = maximum_contact_force_threshold
        self.start_pos = start_pos # 'home' or 'extended'
        self.max_force = maxforce
        self.contact_type = contact_type
        self.number_of_springs = number_of_springs
        self.young_modulus = youngs_modulus
        self.young_modulus_type = youngs_modulus_type
        self.randomise_num_springs = randomise_num_springs
        self.patient = patient
        self.test= test
        self.width = width
        self.randomise_ligs = randomise_ligs
        self.randomise_start = randomise_start
        self.randomise_foot_dynamics = randomise_foot_dynamics
        self.alpha = 0.4 # Set alpha between 0.05 and 0.1
        self.force_window = deque(maxlen=5)
        ## Initialise variables to 0 
        self.episodes_done = 0
        self.force = np.float32(0)
        self.output_force = np.float32(0)
        self.goal_range_low = np.zeros(3)
        self.goal_range_high = np.zeros(3)
        self.goal_ori_low = np.zeros(3)
        self.goal_ori_high = np.zeros(3)
        self.current_step = 0
        self.filtered_contact_force = 0.0
        self.pos_distance = 0.0
        self.angle = 0.0
        self.n = 0
        self.anycontact = 0
        self.filtered_force = 0
        self.eval_count = 0
        self.not_valid_count = 0
        self.goal_gen_count = 0
        self.f_smooth = 0.0
        self.footjoint = -1
        self.loadcell = 0 
        self.randomise_sensor_noise = randomise_sensor_noise
        self.safe_mode = safe_mode
        ## Rendering setup
         ## need to fix this and add a render function, keep getting a warning about it
        
        self.render()

        # if self.randomise_ligs or self.randomise_num_springs or self.randomise_start:
        #     print("Randomisation enabled for ligaments, number of springs, or start position.")
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
        self.maximum_force = 0
        self.contact_distance =0
        self.anycontact = 0
        self.filtered_contact_force = 0.0
        self.contact_ema = 0.0
        self.interlock_count = 0
        if getattr(self, 'randomise_sensor_noise', False):
    # Sample 3D bias drift once per episode for 5N scale
            self.loadcell_bias_3d = self.np_random.uniform(-0.25, 0.25, size=3)
        else:
            self.loadcell_bias_3d = np.zeros(3)
        if self.randomise_num_springs:
            self.number_of_springs = np.random.randint(1, 3)  # Randomly choose between 1 and 5 springs
            #print(f"Randomised number of springs: {self.number_of_springs}")
        #   ##This is in init? Check in test 
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD) ##Needed for FEM
        
        self.band_id = None
        self.force_text_id = None
        
        ##Make Scene
        utils.make_scene(self)
        
        fracturestart, fractureorientationDeg = utils.getStarts(self)
        #fracturestart = np.array([0.3618006205558777, -0.102467754304409027, 0.07800002501010895]) #252
        #fracturestart = np.array([0.35496586561203003, -0.08662302792072296, 0.07155311107635498])-np.array([0.01,-0.01,-0.005]) #252
        #([0.3390733003616333, -0.15371645987033844, 0.13864654302597046])#np.array([0.3316114842891693, -0.33624500036239624, 0.21571703255176544])
        #fracturestart = np.array([0.35496586561203003, -0.11662302911281586, 0.07155311107635498 ])-np.array([0.01,-0.01,-0.005])
        #fracturestart = np.array([0.3817400336265564, -0.09757738560438156, 0.08099538087844849]) #- np.array([0.0,0.0,-0.005]) #252
        #np.array([0.3395577669143677, -0.24252857267856598, 0.15234005451202393])#0.3432255685329437, -0.1527790129184723, 0.07556955516338348]) #252
        #([0.3518006205558777, -0.10467754304409027, 0.07190002501010895])
            # self.goal_pos = np.array(fracturestart.copy())
            # self.goal_ori = np.array(self.goal_type)
            # self.goal_range_low = fracturestart - [0.0125, 0.01, 0.003]
            # self.goal_range_high = fracturestart + [0.0125, 0.02, 0.003]
            # self.goal_ori_low = np.radians(fractureorientationDeg - [15, 5, 15])
            # self.goal_ori_high = np.radians(fractureorientationDeg + [15, 5, 15])
        #fracturestart = np.array([0.3484595715999603, -0.07589279115200043, 0.03660069406032562]) +np.array([0,0,0.025])
        #print(f"Fracture start position: {fracturestart}, Fracture orientation (deg): {fractureorientationDeg}")
        ##
        ## check targer for possible collision 
       # fracturestart = np.array([0.3379683792591095, -0.0786883607506752, 0.06340644508600235])- np.array([0.,-0.005,-0.01])
        ##Load Objects
        current_dir = os.path.dirname(os.path.abspath(__file__))
        leg_path = os.path.join(current_dir, f"Assets/Patient{self.patient}/proximal.urdf")
        foot_path = os.path.join(current_dir, f"Assets/Patient{self.patient}/distal_copy.urdf")

        #footorientation = np.array([-0.07917334884405136, 0.0, 0.0, 0.9968608617782593])#p.getQuaternionFromEuler([90/180*np.pi, 0, 0])
        #footorientation = np.array([0.7139526009559631, -0.016048969700932503, -0.0035978537052869797, 0.7000008821487427])
        #fracturestart = np.array([0.3484595715999603, -0.1658927947282791, 0.06660069525241852])
        orientation = np.array([89/180*np.pi, 15/180*np.pi, 11/180*np.pi])
        footorientation = p.getQuaternionFromEuler([90/180*np.pi,-0/180*np.pi, 0])#p.getQuaternionFromEuler([orientation[0], orientation[1], orientation[2]])
        #p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        #footorientation = np.array([0.6992329955101013, 0.3331104815006256, 0.29179978370666504, 0.5612159967422485])
        if self.patient == 198 :
                    footorientation = p.getQuaternionFromEuler([0,0, 0])
        self.foot = p.loadURDF(foot_path, basePosition=fracturestart, 
                                  baseOrientation=footorientation, 
                                    useFixedBase=0,
                                     globalScaling=1)
        #p.setCollisionFilterGroupMask(self.foot, 1, collisionFilterGroup=0, collisionFilterMask=0)
        dynamics.change_foot_dynamics(self)
        dynamics.change_robot_dynamics(self)
        #time.sleep(100)
        finger_force_n = 5 if self.soft_tissue=='soft' else 5
        #p.setCollisionFilterPair(self.pandaUid, self.foot, 9,1, 1)
        #p.setCollisionFilterPair(self.pandaUid, self.foot, 10,1, 1)
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
        #foot = np.array(p.getLinkState(self.foot, -1,computeForwardKinematics=True)[0])
        #print('Foot position:', p.getBasePositionAndOrientation(self.foot)[0])
        foot_ori = np.array(p.getLinkState(self.foot, 1,computeForwardKinematics=True)[1])
        #print('Foot position:', foot)
        #print('Foot orientation (quaternion):', foot_ori)
      
        leg_orientation = p.getQuaternionFromEuler([0,0, 0]) if self.patient == 132 or self.patient== 198 else p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        #leg_start = np.array([0.3470195700516103, -0.15000000000594865, 0.07526955827664446])#fracturestart-np.array([0.0,0.09,0])#np.array([0.35706788301467896, -0.1598062852025032, 0.07526329159736633])
        goal, leg = get_patient_goal(self, self.patient)
        
        for _ in range(10):
            p.stepSimulation()
        #time.sleep(0.1)
        foot = p.getBasePositionAndOrientation(self.foot)[0]
        leg_start = foot - np.array([0,0.0,0.015])
        
                ##
        p.setGravity(0, 0, -9.81)
        initial_or = p.getLinkState(self.pandaUid, 11)[1]
        #print('Initial end-effector orientation (quaternion):', initial_or)
        #pose_valid = utils.is_goal_configuration_valid(self,self.goal_pos, self.goal_ori)
        if isinstance(self.goal_type, str):
            utils.getGoal(self, fracturestart, fractureorientationDeg) ## do i want to increase the range of goals?
            self.target_position = np.concatenate((self.goal_pos, self.goal_ori))
            #print(self.target_position)
        else:
            # self.target_position, pos, orientation = get_goal_from_proximal_pose(self, 
            #                                                                      self.patient,
            #                                                                      leg_start,
            #                                                                      leg_orientation,
            #                                                                      foot,
            #                                                                      foot_ori)
            self.goal_pos = goal[0]#np.array([0.31803113376479907, -0.08517002163540139, 0.1481109452402959])#goal[0]
            #goal_ori = goal[1]#p.getEulerFromQuaternion(goal[1])-np.array([3/180*np.pi,1/180*np.pi, 0/180*np.pi])
            #print('Goal position:', self.goal_pos, 'Goal orientation (quaternion):', goal_ori)   
            self.goal_ori = goal[1]
           # self.goal_ori =np.array(p.getQuaternionFromEuler(goal_ori))#np.array([-0.06132328,-0.06193331,0.70415999,0.70467186])#np.array(p.getQuaternionFromEuler(goal_ori))#np.array([0.999857944938553, 0.0034038286589975777, -0.014537799360434847, 0.007820248299749461])#np.array(p.getQuaternionFromEuler(goal_ori))
            self.target_position = np.concatenate((self.goal_pos, self.goal_ori))#
            #print('Goal position:', self.goal_pos, 'Goal orientation (quaternion):', self.goal_ori)
            self.target_position = utils.is_goal_in_range(self)
            #self.target_position = np.concatenate((self.goal_pos, self.goal_ori))#np.array([ 0.32180062,-0.09246775, 0.15800003,0.9999999728200057, 0.00023313980271510995, -8.89660707914592e-08, 2.4108688676344187e-06])#2.81656109e-04, -2.81431908e-04,  7.06825125e-01,  7.07388213e-01])
            self.goal_pos = self.target_position[0:3]
            self.goal_ori = self.target_position[3:7]
        #self.target_position = utils.is_goal_in_range(self)
        #self.goal_pos = self.target_position[0:3]
        #self.goal_ori = self.target_position[3:7]
        #print(f"Goal position: {self.goal_pos}, Goal orientation (quaternion): {self.goal_ori}")
        goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
                            basePosition=self.target_position[0:3], baseOrientation=self.goal_ori)
        #time.sleep(1)
        
       ## Enable force/torque sensors
        [p.enableJointForceTorqueSensor(self.pandaUid, joint, enableSensor=True) for joint in range(p.getNumJoints(self.pandaUid))]
        p.enableJointForceTorqueSensor(self.foot, self.loadcell, enableSensor=True) # Load cell joint 
        if self.patient==110:
            leg_start = fracturestart - np.array([0,0.09,0]) 
        else:
            leg_start,leg_start_ori = transformation_matrices.get_leg_start_working(self)
        
        self.leg = p.loadURDF(leg_path,
                                    basePosition =leg_start,#-[0,1,0],
                                    baseOrientation = leg_orientation,
                                    globalScaling = 1.0,
                                    useFixedBase = 1)
        #leg_start = fracturestart - np.array([0,0.09,0])
       # leg_start = (0.3470195700516103, -0.13000000000594865, 0.07526955827664446)
        ## need to combine leg_orientation and leg_start_ori to get the correct orientation for the leg
        #leg_orientation = p.multiplyTransforms([0, 0, 0], leg_start_ori, [0, 0, 0], leg_orientation)[1]
        #leg_start = [0.35707396,0.13249574,0.00356559 ]
        #print('Leg start position:', leg_start)
        if self.patient == 198:
            leg_orientation = p.getQuaternionFromEuler([0,0, 0])
        #p.getLinkState(self.foot, -1)[0]
                #print(foot)
        #difference = np.array([0, 0.105, 0.02])
        #leg_start=foot - difference
        
        
        dynamics.change_leg_dynamics(self)
        p.changeVisualShape(self.leg, -1, rgbaColor=[0.8, 0.8, 0.8, 1])  
        p.setCollisionFilterGroupMask(self.foot, self.footjoint, collisionFilterGroup=0, collisionFilterMask=0)
        p.setCollisionFilterGroupMask(self.leg, -1, collisionFilterGroup=0, collisionFilterMask=0)
        ##Initial Observation
        initial_pos = p.getLinkState(self.pandaUid, 11)[0]
        initial_or = p.getLinkState(self.pandaUid, 11)[1]
        #initialholdObject = len(p.getContactPoints(self.pandaUid, self.foot))
        self.dist = utils.fingertip_distance(self.pandaUid, 9, 10)
        # use helper to get 0/1 contact flags
        left_contact = utils.contact_flag(self, 9)
        right_contact = utils.contact_flag(self, 10)
        #print(f'initial_pos: {initial_pos}, initial_or: {initial_or}, left_contact: {left_contact}, right_contact: {right_contact}, dist: {self.dist}')

        initial_isHolding = utils.is_holding(self, left_contact, right_contact, self.dist)
        initial_vel = p.getLinkState(self.pandaUid, 11, 1)[6]
        initial_Joint_Poses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
        initial_Joint_Velocities = [p.getJointState(self.pandaUid, i)[1] for i in range(9)]
        #print(self.goal_ori)
        self.pos_distance, self.angle = utils.calculate_distances(self, initial_pos, initial_or, self.goal_pos, self.goal_ori)
        initial_isHolding = int(initial_isHolding)
        initial_force = p.getJointState(self.foot, self.loadcell)[2]  # Joint index 0 is the fixed joint
        initial_force = np.linalg.norm(initial_force[0:3])
       
        contacts = p.getContactPoints(self.foot, self.leg, self.footjoint, -1)

        # Check if contacts exist AND if any contact distance is below your threshold
        self.contact = 1 if (contacts and any(pt[8] < 0 for pt in contacts)) else 0
        if self.contact ==1:
            print(f"Contact detected with distance: {(p.getContactPoints(self.foot, self.leg,self.footjoint,-1))[8]:.4f} m")
            self.contact_force,self.contact_distance = utils.get_contact_force(self, self.foot, self.leg)
        #print((p.getContactPoints(self.foot, self.leg,1,-1)))
        env_utils.set_observation(self, 
                                  initial_pos, 
                                  initial_or, 
                                  initial_vel, 
                                  initial_Joint_Poses, 
                                  initial_Joint_Velocities, 
                                  initial_force,
                                  self.maximum_force,
                                  self.contact,
                                  self.anycontact,
                                  self.contact_distance,
                                  self.contact_ema,
                                  self.pos_distance,
                                  self.angle,
                                  initial_isHolding)
        #print(f"Youngs Modulus Type is {self.young_modulus_type}, not using soft tissue in this environment.")
        if self.young_modulus_type =='testing' :
            self.eval_count = 0
            self.young_modulus, self.width = utils.get_youngs_modulus_and_width(self)
        elif self.young_modulus_type == 'None':
            self.young_modulus = self.young_modulus
            self.width = 0.005
        
        p.setPhysicsEngineParameter(numSolverIterations=100, numSubSteps=5)
        if self.soft_tissue=='soft':
            self.point_b,_ = new_band.ElasticBand._get_pose_vel(self,self.leg, -1,local_offset=[0.01,0.0,-0.01])
            self.point_a,_ = new_band.ElasticBand._get_pose_vel(self,self.foot, self.footjoint,local_offset=[0.01,-0.0015,0.04]) ##trial and error to place them 
            self.point_c,_ = new_band.ElasticBand._get_pose_vel(self,self.leg, -1,local_offset=[-0.03,0.0,-0.01])
            self.point_d,_ = new_band.ElasticBand._get_pose_vel(self,self.foot, self.footjoint,local_offset=[-0.03,-0.0015,0.04])
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
            self.band = new_band2.ElasticBand(bodyA=self.foot, linkA= self.footjoint,
                                         bodyB=self.leg, linkB= -1,
                                         young_modulus=self.young_modulus,
                                         area=5e-6,
                                         width= self.width,
                                         num_springs=self.number_of_springs, randomize_position=self.randomise_ligs,
                                         randomize_num_ligaments=self.randomise_num_springs, patient= self.patient
                                         )
            
            
        else: 
            
            pass  
       
        #print(p.getClosestPoints(bodyA=self.foot, bodyB=self.leg, linkIndexA=1, linkIndexB=-1,distance=0.5 ))
        #utils.drawAABB(self,self.leg,-1)
        p.setCollisionFilterPair(self.foot,self.leg,self.footjoint,-1,1) ## Allow collision between foot and leg but not between the soft object, very unstable 
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
        
        #dx, dy, dz = utils.apply_cbf_emergency_guardrail(self, dx, dy, dz)

        # 2. Derive target end-effector pose using the safe (CBF-filtered) displacements
        new_Position, new_Orientation = utils.get_new_pose(self, dx, dy, dz, qx, qy, qz, qw, mode)
        #print(action)
        # 3. Calculate Inverse Kinematics for the guarded target position
        if self.action_type == 'pos_only':
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, 11, targetPosition=new_Position, maxNumIterations=10, residualThreshold=1e-4
            )
        else:
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, 11, targetPosition=new_Position, targetOrientation=new_Orientation, maxNumIterations=10, residualThreshold=1e-4
            )

        # 4. Check for IK numerical instability (NaNs or unphysical joint values)
        if np.any(np.isnan(jointPoses)) or np.any(np.abs(jointPoses) > 10):
            print("IK solver failure. Reverting to current joint targets.")
            try:
                jointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
            except Exception:
                jointPoses = [0.0] * 9

        # 5. Retrieve current joint states and define actuator limits
        start_pos = np.array([p.getJointState(self.pandaUid, j)[0] for j in range(9)])
        max_joint_force = [87, 87, 87, 87, 12, 12, 12, 20, 20]

        # 6. Adapt step speed & enforce hard halts based on contact force
        # if self.contact_ema >= 0.15 or getattr(self, 'halted', False):
        #     num_steps = 50
        # elif self.contact_ema > 0.10:
        #     num_steps = 25  # Increase interpolation sub-steps to reduce end-effector velocity
        # else:
        #     num_steps = 20  # Standard execution speed
          # Standard execution speed
        # 7. Reset emergency flag and step simulation with sub-step monitoring
        # if self.contact_distance < 0:
        #     num_steps = 50
        # else:
        #     num_steps = 12  # Slow down if in contact to reduce force spikes
        #num_steps = utils.compute_smooth_substeps(self,self.contact_distance)
        #num_steps = utils.compute_cbf_substeps(self.contact_distance, dd_ds=-1.0, nominal_steps=12, max_steps=50)
        if self.safe_mode:
            num_steps = utils.compute_cbf_substeps(self.contact_distance, dd_ds=-1.0, nominal_steps=12, max_steps=50)
            self.output_force, max_step_force, avg_force, all_mean, contact_mean, contact_distance = utils.smooth_motion_safe(
                        self, jointPoses, start_pos, max_joint_force, numsubsteps=num_steps
                    )
        else:
            num_steps =12
            self.output_force, max_step_force, avg_force, all_mean, contact_mean, contact_distance = utils.smooth_motion(
                self, jointPoses, start_pos, max_joint_force, numsubsteps=num_steps
            )
        #print(f'Contact Force: {contact_mean:.4f} N, Contact Distance: {contact_distance:.4f} m')
        self.contact_ema = (self.alpha * contact_mean) + ((1.0 - self.alpha) * self.contact_ema)
        
        # 1. Set distance metric unconditionally (DRY principle)
        self.contact_distance = contact_distance

        # 2. Update contact flags using assignment '=' rather than comparison '=='
        if self.contact_ema >= self.maximum_contact_force_threshold:
            self.contact = 1
            self.anycontact = 1  # Persists as 1 for the episode once contact occurs
            #print('Contact!')
        else:
            self.contact = 0
        #print(f"Contact EMA: {self.contact_ema:.4f}, Contact Distance: {self.contact_distance:.4f}, Any Contact: {self.anycontact}")
        alpha_step = 0.4  # Smoothing factor for force EMA  
        spike_threshold = 15.0  # Define a threshold for spike detection: Pybullet gives random spikes in force,
        # going to ignore any readings above 15N which is likely just a spike and not a real reading 
        
        if avg_force > spike_threshold:
            self.filtered_force = self.filtered_force  # Ignore spike, keep previous filtered value
        else:
            self.filtered_force = (alpha_step * avg_force) + ((1.0 - alpha_step) * self.filtered_force)

       
        # 6. Peak-hold tracking for maximum observed filtered force
        if self.filtered_force > self.maximum_force:
            self.maximum_force = self.filtered_force
        # if self.soft_tissue=='soft':
        #     worldA, worldB = createligament.Ligament.radius_spring(self.foot, self.leg,
        #                                                     self.point_a, self.point_b)
        #     stretch = np.linalg.norm(worldA - worldB) 
        
       
       
        
        
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
        
        env_utils.set_observation(self, 
                                  actual_New_Position, 
                                  actual_New_Orientation, 
                                  actual_New_Velocity, 
                                  joint_Poses, 
                                  joint_Velocities,
                                  self.filtered_force,
                                  self.maximum_force,
                                  self.contact, 
                                  self.anycontact,
                                  self.contact_distance,
                                  self.contact_ema,
                                  self.pos_distance,
                                  self.angle, 
                                  self.isHolding)
        
        #print('Max Force: ', self.maximum_force, 'Filtered Force',self.filtered_force)
        done = env_utils.check_done(self)
        exploded = False
        #print(actual_New_Position, actual_New_Orientation,self.pos_distance,self.angle)
        if self.test and (avg_force >= 100 or self.isHolding ==0):
            print('Terminating episode due to excessive force during testing.')
            truncated = True
            reward = -100
            exploded = True
        else:
            truncated = self.current_step >= self.max_steps and not done
        
        # if done:
        #     print('MaxForce: ', self.output_force, 
        #        'Pos Distance: ', self.pos_distance, 
        #        'Angle: ', self.angle, 
        #        'Holding: ', self.isHolding, 
        #        'Contact: ', self.anycontact)
        #if done or truncated:
        
        # if done:
        #    # time.sleep(100)
        #     print('yay')
        # if truncated:
        #     print(f'truncated Max Force: {self.maximum_force}, Contact Force: {self.contact_ema}, Pos Distance: {self.pos_distance}, Angle: {self.angle}')#,{self.isHolding},{self.contact}')
            #print(f'End-effector pos: {actual_New_Position}, End-effector ori: {actual_New_Orientation}, goal pos: {self.goal_pos}, goal ori: {self.goal_ori}')
        info = {'is_success': done,'truncated': truncated, 'current_step': self.current_step, 
                'pos_distance': self.pos_distance, 
                'angle': self.angle, 'Holding': self.isHolding, 
                'force': self.filtered_force,'maximum_force': self.maximum_force,
                'contact': self.anycontact,'force_axis_mso ean': all_mean, 
                'young_modulus': self.young_modulus,
                'contact_force':self.contact_ema,
                'exploded': exploded,
                'contact_distance':self.contact_distance,
                'width': self.width,
                'interlock_count': self.interlock_count}#,'force_mag':self.force_magnitude}#,
        #print(stretch,self.output_force)
                #'stretch':stretch,'force_mag':force_mag,'contact': self.anycontact}
        if (not self.test) or (avg_force <= 100):
            reward = self.compute_reward(self.achieved_goal, self.desired_goal, info)
        # else: keep the earlier penalty reward (-100)
        reward = np.float32(reward)
        #print('force: ', self.force, reward)
        #print(self.anycontact)
        # if done:
        #     #time.sleep(50)
        #     print(f'foot pos: {p.getBasePositionAndOrientation(self.foot)[0]}, foot ori: {p.getBasePositionAndOrientation(self.foot)[1]}, goal pos: {self.goal_pos}, goal ori: {self.goal_ori}')
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
        p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=1.1, yaw=87, pitch=-20, roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        p.getCameraImage(10, 10,viewMatrix=matrix,projectionMatrix=projection)  # Warm up the renderer to prevent first-step lag

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False