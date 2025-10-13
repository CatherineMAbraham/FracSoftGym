## Position and Orientation with Dictionary Observation

## Modules to Import
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random 
import time
from gym_fracture.envs.utils import calculate_distances, make_scene,getStarts, getGoal, compute_reward, constrain_quat#, get_new_orientation, compute_reward
from scipy.spatial.transform import Rotation as R
import tensorboard

## Fracture Surgery Environment
class fracturesurgery_env(gym.Env):
   
   def __init__(self, render_mode=None, 
                reward_type='sparse', 
                distance_threshold_pos=0.005,
                max_steps=50, 
                obs_type= 'dict',
                asset_path="/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/"):
       self.metadata = {'render_mode': ['human']}
       
       ##setup
       scaling = 1.
       self.render_mode =render_mode
       self.reward_type= reward_type
       self.obs_type = obs_type
       self.asset_path = asset_path

       if self.render_mode == 'human':
           p.connect(p.GUI,options="--background_color_red=0.9686--background_color_blue=0.79216--background_color_green=0.7882")
       else:
           p.connect(p.DIRECT)
       
       p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) 
       p.setAdditionalSearchPath(pybullet_data.getDataPath())
       p.resetDebugVisualizerCamera(cameraDistance = 1.4, cameraYaw= 45, cameraPitch = -30,
                                     cameraTargetPosition = [0,0,0])
       
       #initilise
       if self.obs_type == 'dict':
        self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=-200, high=200, shape=(7,), dtype=np.float32), #pos, ori, vel
                'achieved_goal': spaces.Box(low=-200, high=200, shape=(7,), dtype=np.float32),
                'desired_goal': spaces.Box(low=-200, high=200, shape=(7,), dtype=np.float32)
            })
       else :
              self.observation_space = spaces.Box(low=-200, high=200, shape=(7,), dtype=np.float32)
       
       self.action_space = spaces.Box(low=-1, high=1, shape=(7,)) #x,y,z, qx,qz
       
       
        
       #The range of where the goal position should be. Need to tune this.
       self.goal_range_low = np.zeros(3)
       self.goal_range_high = np.zeros(3)
       self.goal_ori_low= np.zeros(3)
       self.goal_ori_high=np.zeros(3) 
       
       self.current_step = 0
      
       self.max_steps = max_steps
       self.pos_distance = 0.0
       self.angle = 0.0
       
    #    # Initialize state
    #    self.state = {'observation' : np.zeros(7, dtype=np.float32),
    #                  'achieved_goal' : np.zeros(7, dtype=np.float32),
    #                  'desired_goal' : np.zeros(7, dtype=np.float32),}
       
       #Threshold for how close the end-effector needs to be to be considered at goal
       self.distance_threshold_pos = distance_threshold_pos 
       self.distance_threshold_ori = 0.05
       
       self.pitch = 0.0
       #Counter
       self.n=0

   def reset(self, seed=None, options=None):
       self.n+=1
       p.resetSimulation()
       
       self.current_step = 0

       make_scene(self)

        #Get the object start position and orientation
       fracturestart, fractureorientationDeg, legstartpos= getStarts(self)#

        #Set the goal range for the object position and orientation and pick a random goal
       getGoal(self,fracturestart, fractureorientationDeg)
       
       ##Load Leg and Fracture
       leg_path =os.path.join(self.asset_path, "leg.urdf")
       foot_path = os.path.join(self.asset_path, "foot.urdf")
       self.leg = p.loadURDF(leg_path,
                        basePosition =legstartpos,
                        #baseOrientation = legorientation,
                        globalScaling = 1,
                        useFixedBase = 1)

       self.objectUid = p.loadURDF(foot_path,
                                   basePosition = fracturestart,
                                    globalScaling=1)
       
    
       p.changeDynamics(self.objectUid,
                        -1,
                        mass=0.6,
                        lateralFriction=0.5)
       #add_constraints(self)
       # Gradually close the gripper around the object
       target_positions = np.array([0.00, 0.00])
       forces = [50, 50]  # Use smaller forces to avoid squeezing too hard


       for _ in range(50):
            p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
            p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
            p.stepSimulation()
            time.sleep(1./500.)
       
       
       p.setGravity(0,0,-9.8)
       
       
       #give time to load
       for _ in range(10):
           p.stepSimulation()
           time.sleep(0.002)

       self.target_position = np.concatenate((self.goal_pos ,self.goal_ori))
       
       
       object_id = p.createMultiBody(baseMass=0, 
                               baseCollisionShapeIndex=-1,  # No collision shape
                               baseVisualShapeIndex=self.visual_shape, 
                               basePosition=self.goal_pos, baseOrientation=self.goal_ori)
          
       
       #Get state info
       initialpos = p.getLinkState(self.pandaUid, 11)[0]
       
       initialor = p.getLinkState(self.pandaUid, 11)[1]
       
       #initialvel = p.getLinkState(self.pandaUid, 11,1)[6]
       observation = initialpos + initialor #+initialvel
       
       
       if self.obs_type == 'dict':
           achieved_goal = initialpos + initialor
           desired_goal = self.target_position
           observation = {
            "observation": observation,
            "achieved_goal": achieved_goal, # observation and achieved goal the same in this scenario as observation is only looking at position
            "desired_goal": desired_goal.astype(np.float32),
        }
           self.state = observation
       else: 
           observation = observation
           self.state = np.array(observation)
       
       
       
       return self.state,{}
   
   def step(self, action):
       self.current_step += 1
       

       #Organise the selected actions chosen according to policy
       dv = 0.05 #0.005

       dx = action[0]  * dv
       dy = action[1] * dv
       dz = action[2] * dv
      
       qx = action[3] *dv
       qy = action[4]*dv
       qz = action[5]*dv
       qw = action[6]*dv

       #Calculate the new orientation change
       deltaPos =[dx,dy,dz]
       deltaOr =[qx,qy,qz,qw]#,qw]
       #deltaOr = p.getQuaternionFromEuler(deltaorE) 
       
       #Calculate the new positions
       currentPose = p.getLinkState(self.pandaUid, 11,1) #1 is velocity
       currentPosition = currentPose[0]
    #    newPosition = [currentPosition[0] + dx,
    #                    currentPosition[1] + dy,
    #                    currentPosition[2] + dz]
       
       #Clip the new position - speeds up learning
       
       
       #Work out new orientation
       currentOrientation = currentPose[1]
       newPosition, newOrientation = p.multiplyTransforms(currentPosition, currentOrientation, deltaPos, deltaOr)
       newPosition = np.array(newPosition)
       newOrientation = np.array(newOrientation)
       newPosition = np.clip(newPosition,self.goal_range_low,self.goal_range_high)
       #newOrientation1 = -newOrientation
       #euler = p.getEulerFromQuaternion(newOrientation)
       newOrientation = constrain_quat(self,newOrientation)
       #newOrientationE = np.clip(euler,self.goal_ori_low,self.goal_ori_high)
       #newOrientation = p.getQuaternionFromEuler(newOrientationE)
       
       #Calculate the joint positions
       jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, newOrientation, maxNumIterations = 1000, residualThreshold = 0.0000001)
       p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL, list(jointPoses))
       
       #Step to new position in 20 timesteps
       for _ in range(20):
           p.stepSimulation()
           time.sleep(1./500)

       #Get position the robot is actually in  
       actualNewPosition = p.getLinkState(self.pandaUid, 11)[0]
       actualNewOrientation = p.getLinkState(self.pandaUid, 11)[1]
       #actualNewVelocity = p.getLinkState(self.pandaUid, 11,1)[6]
       
       #Print out observation 
       observation_state = actualNewPosition + actualNewOrientation #+actualNewVelocity
       observation_state = np.array(observation_state)
       
       if self.obs_type == 'dict':
           achieved_goal = actualNewPosition + actualNewOrientation
           achieved_goal = np.array(achieved_goal)
           desired_goal = self.target_position
           desired_goal= np.array(desired_goal)

           observation = {
            "observation": observation_state.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32), # observation and achieved goal the same in this scenario as observation is only looking at position
            "desired_goal": desired_goal.astype(np.float32),#self.task.get_goal().astype(np.float32),
        }
           self.state = observation
       else: 
           observation = observation_state
           self.state = observation
       

       #Reward is negatively proportional to distance from target
       #d = self.pos_distance + self.angle
       self.pos_distance, self.angle= calculate_distances(self,actualNewPosition, 
                                                          actualNewOrientation,
                                                          self.goal_pos,
                                                          self.goal_ori)
       
       holdObject = len(p.getContactPoints(self.pandaUid, self.objectUid))
       self.isHolding = 1 if holdObject > 0 else 0
       
       done = bool(self.pos_distance<self.distance_threshold_pos 
                   and self.angle<self.distance_threshold_ori 
                   and self.isHolding==1)# and self.angle <0.05) #done when we're less than 1mm away 
       truncated = bool(self.current_step >= self.max_steps and not done)
       
       info = {'is_success':done}
              
       reward = compute_reward(self,achieved_goal,desired_goal,info) 
       #print(reward.type)
       if done:
           #print('Hit done')
           reward = reward
           #print(f'We won, position, {self.pos_distance}, orientation{self.angle}')
         #  print(f'We are at target: {newPosition}')
       elif truncated:
           #print(f'orientation :{actualNewOrientation}')
           #print(f'We failed, position, {self.pos_distance}, orientation{self.angle}, isHolding: {isHolding}')
           reward = reward
       else:
           reward = reward  # Penalty for not reaching the target within the maximum steps
       return self.state, reward, done, truncated,info
   
   
   def close(self):
        pass
