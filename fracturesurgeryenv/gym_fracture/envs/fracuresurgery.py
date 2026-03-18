## Position and Orientation with Dictionary Observation

## Modules to Import
import gymnasium as gym
from gymnasium import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
#from gym_fracture.envs import spring_system, utils #calculate_distances, make_scene, getStarts, getGoal, check_done, get_new_pose, unpack_action,fingertip_distance, visualize_contact_forces, world_to_local
from gym_fracture.envs import env_utils, utils
from gym_fracture.envs import dynamics, new_band,new_band2
from scipy.spatial.transform import Rotation as R
import wandb
#from gym_fracture.envs.spring_damper import SpringDamper
from gym_fracture.envs.createligament import make_ligament, make_ligament_rod,radius_spring
#from gym_fracture.envs.multispring import create_ligament_chain, apply_axial_springs

class fracturesurgery_env(gym.Env):
    def __init__(
        self,
        render_mode=None,
        reward_type='sparse',
        distance_threshold_pos=0.005,
        distance_threshold_ori=3,
        max_steps=100,
        obs_type='dict',
        goal_type='random',
        dt=0.001,
        dr=0.1,
        action_type='euler',
        horizon='variable',
        softtissue='spring',
        start_pos = 'home',
        maxforce = 3.5,
        number_of_springs = 3,
        contact_type = 0,
        youngs_modulus = 1e6,
        test = False
    ):
        metadata = {"render_modes": ["human", None]}
        ## Initialise variables
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.dt = dt
        self.dr = dr
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
        self.start_pos = start_pos # 'home' or 'extended'
        self.maxforce = maxforce
        self.contact_type = contact_type
        self.number_of_springs = number_of_springs
        self.anycontact = 0
        self.young_modulus = youngs_modulus
        self.test= test
        ##
        
        ## Rendering setup
         ## need to fix this and add a render function, keep getting a warning about it
        
        self.render()

        self.connected = True
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
        #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=87, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
        ##
        #p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=1.1, yaw=87, pitch=-20, roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
        p.getCameraImage(10, 10,viewMatrix=matrix,projectionMatrix=projection)  # Warm up the renderer to prevent first-step lag
        p.setTimeStep(1/240)
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
        self.n += 1
        self.current_step = 0
        self.force = 0
        self.output_force = 0
        contact = 0
        self.anycontact = 0
        self.filerted_force = 0
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        
        self.band_id = None
        self.force_text_id = None
        
        ##Make Scene
        utils.make_scene(self)
        
        fracturestart, fractureorientationDeg = utils.getStarts(self)
        if isinstance(self.goal_type, str):
            utils.getGoal(self, fracturestart, fractureorientationDeg) ## do i want to increase the range of goals?
        else:
            self.goal_pos = np.array(fracturestart.copy())
            self.goal_ori = np.array(self.goal_type)
            self.goal_range_low = fracturestart - [0.0125, 0.01, 0.003]
            self.goal_range_high = fracturestart + [0.0125, 0.02, 0.003]
            self.goal_ori_low = np.radians(fractureorientationDeg - [15, 5, 15])
            self.goal_ori_high = np.radians(fractureorientationDeg + [15, 5, 15])

        self.target_position = np.concatenate((self.goal_pos, self.goal_ori))
        ##

        ##Load Objects
        currentDir = os.path.dirname(os.path.abspath(__file__))
        leg_path = os.path.join(currentDir, "Assets/Patient110/proximal.urdf")
        foot_path = os.path.join(currentDir, "Assets/Patient110/distal.urdf")
        
        #footorientation = p.getQuaternionFromEuler([90/180*np.pi, 0, 0])
       
        legorientation = p.getQuaternionFromEuler([90/180*np.pi,0, 0])
        
        self.objectUid = p.loadURDF(foot_path, basePosition=fracturestart, 
                                   # baseOrientation=footorientation, 
                                    useFixedBase=0,
                                     globalScaling=1)
        
        dynamics.change_foot_dynamics(self)
        dynamics.change_robot_dynamics(self)
        #time.sleep(100)
        ##
        #p.setPhysicsEngineParameter(contactERP=0.1) 
        ## Close gripper
        #target_positions = np.array([0.0, 0.0])
        fingerforce = 2 if self.softtissue=='soft' else 10
        for _ in range(100):
            p.setJointMotorControl2(self.pandaUid, 9, p.VELOCITY_CONTROL, targetVelocity=-1, force=fingerforce)
            p.setJointMotorControl2(self.pandaUid, 10, p.VELOCITY_CONTROL, targetVelocity=-1, force=fingerforce)
            p.stepSimulation()
            #time.sleep(1./500)  # Remove for speed
        
        ##
        difference = [0.0,0.09,0]
        difference =np.array(difference)
        foot = p.getLinkState(self.objectUid, 1)[0]
        legstart=foot - difference
    
        ##Load Leg
        self.leg = p.loadURDF(leg_path,
                        basePosition =legstart,
                        baseOrientation = legorientation,
                        globalScaling = 1.0,
                        useFixedBase = 1)
        #time.sleep(100)
        
        dynamics.change_leg_dynamics(self)

        ##Settle
        #print('Settling the simulation...') 
        for _ in range(10):
            p.stepSimulation()
        
        
        p.setGravity(0, 0, -9.81)
        
        
        # Dummy visual shape for goal marker
        
        
        # goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
        #                    basePosition=self.goal_pos, baseOrientation=self.goal_ori)
 
        
       ## Enable force/torque sensors
        [p.enableJointForceTorqueSensor(self.pandaUid, joint, enableSensor=True) for joint in range(p.getNumJoints(self.pandaUid))]
        p.enableJointForceTorqueSensor(self.objectUid, 0, enableSensor=True) # Load cell joint 
        
        ##
        
        
        ##Initial Observation
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
        force = p.getJointState(self.objectUid, 0)[2]  # Joint index 0 is the fixed joint
        initial_force = np.linalg.norm(force)#utils.visualize_contact_forces(self,self.pandaUid, self.objectUid)
        self.contact = int(bool(p.getContactPoints(self.objectUid, self.leg,1,-1)))
        env_utils.set_observation(self, 
                                  initialpos, 
                                  initialor, 
                                  initialvel, 
                                  initialJointPoses, 
                                  initialJointVelocities, 
                                  initial_force,
                                  self.contact,
                                  self.pos_distance,
                                  self.angle,
                                  left_contact,
                                  right_contact,
                                  self.dist, 
                                  initialisHolding)
        
        ##
        #time.sleep(5)
        if self.softtissue=='soft':
            self.point_b,_ = new_band.ElasticBand._get_pose_vel(self,self.leg, -1,local_offset=[0.01,0.0,-0.01])
            self.point_a,_ = new_band.ElasticBand._get_pose_vel(self,self.objectUid, 1,local_offset=[0.01,-0.0015,0.04]) ##trial and error to place them 
            #make_ligament(self,"cloth_Id1", self.objectUid, self.leg, point_c, point_d,orientation=p.getQuaternionFromEuler([0, 90/180*np.pi, 70/180*np.pi]), scale =1)
            make_ligament(self, "cloth_Id2", self.objectUid, 
                          self.leg, self.point_a, 
                          self.point_b,orientation=p.getQuaternionFromEuler([90/180*np.pi,270/180*np.pi , 180/180*np.pi]), scale =1, youngs_modulus=self.young_modulus) #0.75
        elif self.softtissue=='spring':
            self.band = new_band2.ElasticBand(bodyA=self.objectUid, linkA= 1,
                                         bodyB=self.leg, linkB= -1,
                                         young_modulus=self.young_modulus,
                                         area=5e-6,
                                         rest_length=0.1,
                                         num_springs=2
                                         )
            
            
        else: 
            
            pass  
       
        
        p.setPhysicsEngineParameter(numSolverIterations=10, numSubSteps=10)  # Increase solver iterations for better stability with springs
        ##draw aabb boxes round leg and foot 
        #utils.drawAABB(self, self.leg,-1)
        #utils.drawAABB(self, self.objectUid,1)
        #time.sleep(5)
        return self.state, {}

    
    ## Step Function
    def step(self, action):
        self.current_step += 1
        
        ## Unpack Action
        dx, dy, dz, qx, qy, qz, qw, x, y, z = utils.unpack_action(self,action, self.dr)
        mode_map = {
            'euler': 'euler',
            'fouractions': 'fouractions',
            'ori_only': 'ori_only',
            'pos_only': 'pos_only'
        }
        mode = mode_map.get(self.action_type, None)

        
        newPosition, newOrientation = utils.get_new_pose(self,dx, dy, dz, qx, qy, qz, qw, mode)
        if self.action_type == 'pos_only':
            jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, maxNumIterations=100, residualThreshold=1e-4)
        else:
            jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, 
                                                      targetOrientation=newOrientation, maxNumIterations=100, residualThreshold=1e-4)
            #p.addUserDebugText('NP',newPosition, textSize=1.5)
        if np.any(np.isnan(jointPoses)) or np.any(np.abs(jointPoses) > 10):
            print("IK failure, skipping step")
            print(action)
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

        # Set Joint Motors
        max_force = [87,87,87,87,12,12,12,20,20]
        
        start_pos = np.array([p.getJointState(self.pandaUid, j)[0] for j in range(9)])
        
        #p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL,targetPositions = jointPoses,forces=max_force)#, maxVelocities=max_vel)
        
        if self.softtissue=='spring':
           self.output_force, max_step_force,avg_force= utils.smooth_motion(self, jointPoses, start_pos, max_force, numsubsteps=20)
           alpha = 0.2
           self.filerted_force = (alpha * avg_force) + ((1 - alpha) * self.filerted_force)
           if self.filerted_force > self.output_force:
                self.output_force = self.filerted_force
        elif self.softtissue=='soft':
            self.output_force,max_step_force, avg_force = utils.smooth_motion(self, jointPoses, start_pos, max_force, numsubsteps=20)
        else: 
            self.output_force,max_step_force, avg_force = utils.smooth_motion(self, jointPoses, start_pos, max_force, numsubsteps=12)
            alpha = 0.2
            self.filerted_force = (alpha * avg_force) + ((1 - alpha) * self.filerted_force)
            if self.filerted_force > self.output_force:
                self.output_force = self.filerted_force
        if self.softtissue=='soft':
            worldA, worldB = radius_spring(self.objectUid, self.leg,
                                                            self.point_a, self.point_b)
            stretch = np.linalg.norm(worldA - worldB) 
        
        force = p.getJointState(self.objectUid, 0)[2]  # Joint index 0 is the fixed joint
        force_magnitude = np.linalg.norm(force)
        print(f'Force: {self.filerted_force:.2f} N')    
        ##measure the distance between the foot and leg to get an estimate of stretch (not exact but should correlate well and is much cheaper to compute than the world_to_local for each spring every step)
        
        
        #force = p.getJointState(self.objectUid, 0)[2]  # Joint index 0 is the fixed joint
        #force_magnitude = np.linalg.norm(force)
        #print(f'Force: {force_magnitude}')
        
        
        
        self.contact = int(bool(p.getContactPoints(self.objectUid, self.leg,1,-1))) 
        
        #print('Contact: ', self.contact)
        if self.contact==1:
            #print('Contact detected between foot and leg!')
            self.anycontact = 1
            #print('Contact!')
        #print('Contact points between foot and leg: ', contact)
        #print('Contact points between foot and leg: ', contact)
        #print('Load cell force: ', force, force_magnitude)
        ## Observation Update
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
        capped_force = min(self.output_force,200)
        env_utils.set_observation(self, 
                                  actualNewPosition, 
                                  actualNewOrientation, 
                                  actualNewVelocity, 
                                  jointPoses, 
                                  jointVelocities,
                                  capped_force,
                                  self.contact, 
                                  self.pos_distance,
                                  self.angle,
                                  left_contact, 
                                  right_contact, 
                                  dist,  
                                  self.isHolding)
        
        
        done = env_utils.check_done(self)
        if self.test and self.output_force > self.maxforce:
            print('Terminating episode due to excessive force during testing.')
            truncated = True
            reward = -100
        else:
            truncated = self.current_step >= self.max_steps and not done
        
        if done:
            print('yay!')
        elif truncated:
            print('MaxForce: ', self.output_force, 
               'Pos Distance: ', self.pos_distance, 
               'Angle: ', self.angle)
        
        
        
        info = {'is_success': done,'truncated': truncated, 'current_step': self.current_step, 
                'pos_distance': self.pos_distance, 
                'angle': self.angle, 'Holding': self.isHolding, 
                'force': self.output_force,'contact': self.anycontact}#'stretch': stretch_step}#,'force_mag':self.force_magnitude}#,
        #print(stretch,self.output_force)
                #'stretch':stretch,'force_mag':force_mag,'contact': self.anycontact}
        if (not self.test) or (self.output_force <= self.maxforce):
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

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False
