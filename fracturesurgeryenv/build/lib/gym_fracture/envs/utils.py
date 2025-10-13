import pybullet as p
import numpy as np
import time 
import pybullet_data
import os
from scipy.spatial.transform import Rotation as R
def make_scene(self):
    #Start Positions: Worked out previously
       startposition = np.array([0.03, 0.2, 0, -1.802, -2.89, 2.8, 0.61, 0.04, 0.04])

       #load scene
       #Make Plane, Table, Cube       
       plane_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=np.array([30.0, 30.0, 0.01]))
       plane_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=np.array([30.0, 30.0, 0.01]),rgbaColor=[0.678, 0.847, 0.902, 1])
       plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, 
                             baseVisualShapeIndex=plane_visual_shape,basePosition=[0, 0, -0.33])
       
       table =p.loadURDF("table/table.urdf", basePosition =[0.8,-0.32,-0.33], globalScaling =0.5);#[0.8, 0.4, -0.33]
       
       self.visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01], rgbaColor=[0.835, 0.7216, 1, 1])  # Purple Goal box - no collision properties
       

       #Set up robot with calculated start positions
       
       urdfRootPath=pybullet_data.getDataPath()
      
       self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),
                                  basePosition=[0,-0.06,-0.33],#[-0.5,0,-0.65],
                                  useFixedBase=True, globalScaling = 1)
       
       p.changeDynamics(self.pandaUid,9, lateralFriction= 1,spinningFriction= 0.001)
       p.changeDynamics(self.pandaUid,10, lateralFriction= 1,spinningFriction= 0.001)
       p.resetJointState(self.pandaUid,9, 0.01)
       p.resetJointState(self.pandaUid,10, 0.01) 

       for i in range(8):
           p.resetJointState(self.pandaUid,i, startposition[i])
        
       for _ in range(10):
           p.stepSimulation()
           time.sleep(0.002)

           

       return self.pandaUid

def getGoal(self, fracturestart, fractureorientaionDeg):
    self.goal_range_low = fracturestart- [0.0125,0.01,0.003]
    self.goal_range_high = fracturestart+[0.0125,0.02,0.003]
    self.goal_ori_low= np.radians(fractureorientaionDeg - [15,5,15])
    self.goal_ori_high=np.radians(fractureorientaionDeg + [15,5,15])

    self.goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        
    ori = np.array(self.np_random.uniform(self.goal_ori_low, self.goal_ori_high))
    self.goal_ori = R.from_euler('xyz', ori).as_quat()
    

def getStarts(self):
    fracturestart= np.array(p.getLinkState(self.pandaUid, 11)[0] )
    fractureorientaionRad =p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])
    fractureorientaionDeg = np.degrees(np.array(fractureorientaionRad)) 
    
    #Calculated this difference from the object start position
    difference = [-0.004493, 0.079895+0.001, 0.073322]
    difference =np.array(difference)
    legstart=[]
    for i in range(len(difference)):
        leg = (fracturestart[i])-(difference[i])
        legstart.append(leg)
        
        i+=1
    

    return fracturestart, fractureorientaionDeg, legstart

def add_constraints(self):
    p.removeAllUserDebugItems()
    # Define gripper links (for Franka Panda's fingers)
    left_gripper_link_index = 9   # Left gripper finger link index
    right_gripper_link_index = 10 # Right gripper finger link index


    # Gradually close the gripper around the object
    target_positions = np.array([0.01, 0.01])
    forces = [50, 50]  # Use smaller forces to avoid squeezing too hard


    for _ in range(100):
        p.setJointMotorControl2(self.pandaUid, left_gripper_link_index, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
        p.setJointMotorControl2(self.pandaUid, right_gripper_link_index, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
        p.stepSimulation()
        time.sleep(1./240.)


    # Switch to velocity control with zero velocity and zero force to lock the fingers in place
    p.setJointMotorControl2(self.pandaUid, left_gripper_link_index, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    p.setJointMotorControl2(self.pandaUid, right_gripper_link_index, p.VELOCITY_CONTROL, targetVelocity=0, force=0)


    # Allow time for the gripper to close around the object
    #time.sleep(2)


    # Get the position and orientation of the gripper fingers and the object
    left_finger_state = p.getLinkState(self.pandaUid, left_gripper_link_index)
    right_finger_state = p.getLinkState(self.pandaUid, right_gripper_link_index)
    object_pos, object_orn = p.getBasePositionAndOrientation(self.objectUid)


    # Convert the global positions to the object's local frame
    object_inv_pos, object_inv_orn = p.invertTransform(object_pos, object_orn)


    left_finger_pos_world = left_finger_state[4]  # Position of the left finger in world coordinates
    right_finger_pos_world = right_finger_state[4]  # Position of the right finger in world coordinates


    left_constraint_pos_local, left_constraint_orn_local = p.multiplyTransforms(
        object_inv_pos, object_inv_orn, left_finger_pos_world, left_finger_state[5]
    )
    right_constraint_pos_local, right_constraint_orn_local = p.multiplyTransforms(
        object_inv_pos, object_inv_orn, right_finger_pos_world, right_finger_state[5]
    )


    # Create a fixed constraint between the left gripper finger and the object
    left_constraint = p.createConstraint(
        parentBodyUniqueId=(self.pandaUid),
        parentLinkIndex=left_gripper_link_index,
        childBodyUniqueId=(self.objectUid),
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],  # Local position on the left gripper finger
        childFramePosition= left_constraint_pos_local,
        childFrameOrientation=left_constraint_orn_local
        )


    # Create a fixed constraint between the right gripper finger and the object
    right_constraint = p.createConstraint(
        parentBodyUniqueId=(self.pandaUid),
        parentLinkIndex=right_gripper_link_index,
        childBodyUniqueId=(self.objectUid),
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],  # Local position on the right gripper finger
        childFramePosition=left_constraint_pos_local,#(0.035587161779403687, 0.12, 0.1015474796295166),
        childFrameOrientation=right_constraint_orn_local
    )
    self.constraint_id = right_constraint
    
    # Verify constraint is applied
    #print(f"Constraint ID: {self.constraint_id}")
    assert self.constraint_id is not None, "Constraint was not created successfully."

def compute_reward(self, achieved_goal, desired_goal,info):
       if achieved_goal.ndim == 1:
        # Single goal case
            pos_achieved, angle_achieved = achieved_goal[:3], achieved_goal[3:]
            pos_desired, angle_desired = desired_goal[:3], desired_goal[3:]
            self.pos_distance, self.angle= calculate_distances(self,pos_achieved, angle_achieved,pos_desired, angle_desired)
            reward = -1 if ((self.pos_distance > self.distance_threshold_pos) or (self.angle > self.distance_threshold_ori) or self.isHolding==0) else 0
            #print(bool((self.angle>self.distance_threshold_ori)))# and (self.angle<self.distance_threshold_ori)))
       else:
        # Batched goals case
            pos_achieved, angle_achieved = achieved_goal[:, :3], achieved_goal[:, 3:]
            pos_desired, angle_desired = desired_goal[:, :3], desired_goal[:, 3:]
            self.pos_distance, self.angle= calculate_distances(self,pos_achieved, angle_achieved,pos_desired, angle_desired)
            #print(self.pos_distance,self.angle)
            reward =[]
            for i in range(len(self.pos_distance)):
                if self.pos_distance[i] > self.distance_threshold_pos or self.angle[i] > self.distance_threshold_ori or self.isHolding==0:
                    rewards = -1
                    reward.append(rewards)
                elif self.pos_distance[i] < self.distance_threshold_pos and self.angle[i] < self.distance_threshold_ori and self.isHolding==1:
                    rewards = 0
                    reward.append(rewards)
            
       d = self.pos_distance +self.angle
       rewardDistance = np.exp(-0.1*self.pos_distance) 
       rewardOrientation = np.exp(-0.1*self.angle)
       #closer to 0 the more reward you get 
       e = rewardDistance + rewardOrientation
       if self.reward_type == 'sparse':
        return np.array(reward)
       elif self.reward_type=='dense':
        return -d


def calculate_distances(self,new_pos,new_ori,goal_pos,goal_ori):
    # Calculate positional distance (Euclidean distance)
    self.pos_distance = (np.linalg.norm(np.array(new_pos) - np.array(goal_pos), axis=-1)) #the new distance
    # Calculate the dot product between the quaternions
    dot_product = np.abs(np.sum(new_ori * goal_ori, axis=-1))
    #dot_product = np.dot(new_ori, goal_ori)
    
    # Ensure the dot product is within the valid range for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle (rotational distance) between the quaternions
    self.angle = 2 * np.arccos(dot_product)

    return self.pos_distance, self.angle

def constrain_quat(self, q):
    q_rel = R.from_quat(q) * R.from_quat(self.goal_ori).inv()  # Relative rotation  
    angle = np.linalg.norm(q_rel.as_rotvec())  # Get angle in radians  

    max_angle = np.deg2rad(30)  # 30-degree limit  
    if angle > max_angle:
        scale = max_angle / angle
        q_rel = R.from_rotvec(q_rel.as_rotvec() * scale)  # Scale down rotation  
        q = (q_rel * R.from_quat(self.goal_ori)).as_quat()  # Apply scaled rotation back  

def get_new_orientation(self, currentOrientation, deltaor): ## This uses euler and doesn't seem to work that well
    eulerCurrent= np.array(p.getEulerFromQuaternion(currentOrientation))
    eulerDelta = np.array(p.getEulerFromQuaternion(deltaor))
    newOrientationEuler = eulerCurrent + eulerDelta
    newOrientation = p.getQuaternionFromEuler(newOrientationEuler)
    return np.array(newOrientation)