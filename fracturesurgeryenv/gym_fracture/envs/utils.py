import pybullet as p
import numpy as np
import time 
import pybullet_data
import os
from scipy.spatial.transform import Rotation as R
import wandb
def make_scene(self):
    #Start Positions: Worked out previously
       startposition = np.array([0.03, 0.2, 0, -1.805, -2.89, 2.8, 0.61, 0.04, 0.04]) #-1.802

       #load scene
       #Make Plane, Table, Cube       
       plane_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=np.array([30.0, 30.0, 0.01]))
       plane_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=np.array([30.0, 30.0, 0.01]),rgbaColor=[0.678, 0.847, 0.902, 1])
       plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, 
                             baseVisualShapeIndex=plane_visual_shape,basePosition=[0, 0, -0.33])
       
       self.table =p.loadURDF("table/table.urdf", basePosition =[0.8,-0.32,-0.29], globalScaling =0.5);#[0.8, 0.4, -0.33]

       self.visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.01,0.01,0.01], rgbaColor=[0.835, 0.7216, 1, 1])  # Purple Goal box - no collision properties

       #Set up robot with calculated start positions
       urdfRootPath=pybullet_data.getDataPath()
                  # 🔹 Create the base surgical table (static)
    #    table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.1, 0.002])
    #    table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.1, 0.002], rgbaColor=[0.3, 0.3, 0.3, 1])
    #    table_body = p.createMultiBody(
    #         baseMass=0,
    #         baseCollisionShapeIndex=table_collision,
    #         baseVisualShapeIndex=table_visual,
    #         basePosition=[0.65, 0.05, 0.005],
    #     )
    #    p.changeDynamics(table_body, -1, lateralFriction=0.1, restitution=0.0)

    #     # 🔹 Create a soft pad (a smaller box resting on the table)
    #    pad_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.1, 0.02])
    #    pad_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.1, 0.02], rgbaColor=[0.8, 0.2, 0.2, 1])
    #    pad_body = p.createMultiBody(
    #         baseMass=0,  # static pad
    #         baseCollisionShapeIndex=pad_collision,
    #         baseVisualShapeIndex=pad_visual,
    #         basePosition=[0.8, 0.15, 0.05],  # slightly above table
    #     )
    #    p.changeDynamics(pad_body, -1, lateralFriction=1.5, restitution=0.0)
      
       self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),
                                  basePosition=[0,-0.06,-0.33],#[-0.5,0,-0.65],
                                  useFixedBase=True, globalScaling = 1)
       
       #p.changeDynamics(self.pandaUid,9, lateralFriction= 5,spinningFriction= 0.001,jointLowerLimit=0.00, jointUpperLimit=0.01)
       #p.changeDynamics(self.pandaUid,10, lateralFriction= 5,spinningFriction= 0.001,jointLowerLimit=0.00, jointUpperLimit=0.01)
       p.resetJointState(self.pandaUid,9, 0.005)
       p.resetJointState(self.pandaUid,10, 0.005) 

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
    #print(self.curriculum_phase)
    # if self.curriculum_phase ==1:
    #     self.goal_pos = fracturestart.copy()
    # else:
    goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high,))
    
    if self.action_type == 'fiveactions' or self.action_type== 'fouractions':
        goal_pos[2] = fracturestart[2]
   
    if self.action_type == 'fouractions':
        self.goal_ori_low[1] =np.radians(fractureorientaionDeg[1] - 0)
        self.goal_ori_high[1] =np.radians(fractureorientaionDeg[1]+0)    
    
    self.goal_pos = np.round(goal_pos,3)
    ori = np.array(self.np_random.uniform(self.goal_ori_low, self.goal_ori_high))
    goal_ori = np.array(p.getQuaternionFromEuler(ori))
    #goal_ori = R.from_euler('xyz', ori).as_quat()
    self.goal_ori = np.round(goal_ori,3)


def getStarts(self):
    fracturestart= np.array(p.getLinkState(self.pandaUid, 11)[0] )
    fractureorientaionRad =p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])
    fractureorientaionDeg = np.degrees(np.array(fractureorientaionRad)) 
    pin = [0.004462 ,-0.002332 , 0.046608  ]
    #p.addUserDebugText('P', pin, textColorRGB=[1, 0, 0], textSize=1)
    fracturestart = fracturestart - pin
    #Calculated this difference from the object start position
    #difference = [-0.004493, 0.079895+0.005, 0.073322] difference between leg and foot
    #difference = [0.011489 ,-0.045611 ,-0.006535  ]
    difference = [0,0.002,0]
    difference =np.array(difference)
    #legstart=[]
    # for i in range(len(difference)):
    #     leg = (fracturestart[i])-(difference[i])
    #     legstart.append(leg)
    legstart=fracturestart - difference
    #     i+=1
    

    return fracturestart, fractureorientaionDeg, legstart



def get_new_pose(self, dx, dy, dz, qx, qy, qz, qw=None, mode=None):
        currentPose = p.getLinkState(self.pandaUid, 11, 1)
        currentPosition = np.array(currentPose[0])
        currentOrientation = np.array(currentPose[1])

        if mode == 'rot_vec':
            rotation_vector = np.array([qx, qy, qz])
            angle = np.linalg.norm(rotation_vector)
            if angle < 1e-6:
                deltaOr = [0, 0, 0, 1]
            else:
                max_rotation = np.deg2rad(1)
                clipped_angle = min(angle, max_rotation)
                axis = rotation_vector / angle
                deltaOr = p.getQuaternionFromAxisAngle(axis, clipped_angle)
            deltaPos = [dx, dy, dz]
            newPosition, newOrientation = p.multiplyTransforms(currentPosition, currentOrientation, deltaPos, deltaOr)
            newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            return newPosition, newOrientation

        elif mode in ['euler', 'fouractions', 'fiveactions', 'ori_only']:
            deltaorE = [qx, qy, qz]
            deltaor = p.getQuaternionFromEuler(deltaorE)
            if mode == 'ori_only':
                newPosition = currentPosition
            else:
                newPosition = currentPosition + np.array([dx, dy, dz])
            #newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            newOrientation = np.array(p.multiplyTransforms([0, 0, 0], currentOrientation, [0, 0, 0], deltaor)[1])
            #euler = p.getEulerFromQuaternion(newOrientation)
            #newOrientationE = np.clip(euler, self.goal_ori_low, self.goal_ori_high)
            #newOrientation = p.getQuaternionFromEuler(newOrientationE)
            return newPosition, newOrientation

        elif mode == 'quat':
            deltaOr = np.array([qx, qy, qz, qw])
            deltaOr /= np.linalg.norm(deltaOr)
            axis, angle = p.getAxisAngleFromQuaternion(deltaOr)
            max_rotation = np.deg2rad(3)
            if angle > 0:
                clipped_angle = min(angle, max_rotation)
                deltaOr = p.getQuaternionFromAxisAngle(axis, clipped_angle)
            else:
                deltaOr = [0, 0, 0, 1]
            deltaPos = [dx, dy, dz]
            newPosition, newOrientation = p.multiplyTransforms(currentPosition, currentOrientation, deltaPos, deltaOr)
            #newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            return newPosition, newOrientation

        elif mode == 'pos_only':
            newPosition = currentPosition + np.array([qx, qy, qz])
            newOrientation = currentOrientation
            #newPosition[2] = np.clip(newPosition[2], self.goal_range_low[2], self.goal_range_high[2])
            newPosition = np.clip(newPosition, (self.goal_range_low), (self.goal_range_high))
            return newPosition, newOrientation

        elif mode == 'joint':
            currentJointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
            jointPoses = np.array(currentJointPoses) + np.array([dx, dy, dz, qx, qy, qz, 0, 0, 0])
            return jointPoses, None

        else:
            newPosition = currentPosition + np.array([dx, dy, dz])
            newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            newOrientation = np.array([qx, qy, qz])
            return newPosition, newOrientation

def unpack_action(self, action, dv):
    zeros = [0] * 10
    if self.action_type in ['ori_only', 'pos_only']:
        return [0, 0, 0, action[0] * dv, action[1] * dv, action[2] * dv, 0, 0, 0, 0]
    elif self.action_type == 'quat':
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, action[6] * dv, 0, 0, 0]
    elif self.action_type == 'joint':
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, action[6] * dv, action[6] * dv, action[7] * dv, action[8] * dv]
    elif self.action_type == 'fiveactions':
        return [action[0] * dv, action[1] * dv, 0, action[2] * dv, action[3] * dv, action[4] * dv, 0, 0, 0, 0]
    elif self.action_type == 'fouractions':
        return [action[0] * dv, action[1] * dv, 0, action[2] * dv, 0, action[3] * dv, 0, 0, 0, 0]
    else:
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, 0, 0, 0, 0]


def calculate_distances(self,new_pos,new_ori,goal_pos,goal_ori):
    # Calculate positional distance (Euclidean distance)
    self.pos_distance = (np.linalg.norm(np.array(new_pos) - np.array(goal_pos), axis=-1)) #the new distance
    
    # Calculate the dot product between the quaternions
    dot_product = np.abs(np.sum(new_ori * goal_ori, axis=-1))
    
    
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
        q = (q_rel * R.from_quat(self.goal_ori)).as_quat() 
    
    return q # Apply scaled rotation back  


def visualize_contact_forces(bodyA, bodyB, scale=0.01, lifeTime=0.05, lineWidth=2):
    """
    Draw contact normal, friction vectors and total force for every contact between bodyA and bodyB.
    - bodyA: robot (or the contacting body)
    - bodyB: object (the body receiving force in c[...] interpretation used here)
    - scale: visual scaling factor (1 meter per 1 N would be huge; use ~0.001-0.05)
    """
    contacts = p.getContactPoints(bodyA=bodyA, bodyB=bodyB, linkIndexB=-1)
    for c in contacts:
        # contact point in world frame (use the reported contact position)
        contact_pos = c[6]  # commonly used in examples; point on body B
        # normal on body B (unit vector), points away from B toward A
        normal_dir = np.array(c[7], dtype=float)
        normal_mag = float(c[9])    # normal force magnitude (N)
        tan1_mag = float(c[10])
        tan1_dir = np.array(c[11], dtype=float)
        tan2_mag = float(c[12])
        tan2_dir = np.array(c[13], dtype=float)

        #print(normal_mag, tan1_mag, tan2_mag)
        # compute vector contributions (force on object B)
        f_normal = normal_mag * normal_dir
        f_t1 = tan1_mag * tan1_dir
        f_t2 = tan2_mag * tan2_dir
        f_total = f_normal + f_t1 + f_t2
        print(f_total)
        # p.addUserDebugLine(contact_pos,
        #                    (np.array(contact_pos) + normal_dir * normal_mag * scale).tolist(),
        #                    lineColorRGB=[1, 0, 0], lifeTime=lifeTime, lineWidth=lineWidth)  # red

        # # friction vectors (may be zero)
        # if np.linalg.norm(tan1_dir) > 0 and abs(tan1_mag) > 1e-9:
        #     p.addUserDebugLine(contact_pos,
        #                        (np.array(contact_pos) + tan1_dir * tan1_mag * scale).tolist(),
        #                        lineColorRGB=[0, 0, 1], lifeTime=lifeTime, lineWidth=lineWidth)  # blue

        # if np.linalg.norm(tan2_dir) > 0 and abs(tan2_mag) > 1e-9:
        #     p.addUserDebugLine(contact_pos,
        #                        (np.array(contact_pos) + tan2_dir * tan2_mag * scale).tolist(),
        #                        lineColorRGB=[0, 1, 0], lifeTime=lifeTime, lineWidth=lineWidth)  # green

        # # total contact force (yellow)
        # p.addUserDebugLine(contact_pos,
        #                    (np.array(contact_pos) + f_total * scale).tolist(),
        #                    lineColorRGB=[1, 1, 0], lifeTime=lifeTime, lineWidth=lineWidth+1)
        f_total = np.linalg.norm(f_total)
        f_total = np.float32(f_total)
        return f_total
    
def fingertip_distance(body_id, left_idx, right_idx, physicsClientId=0):
    # getLinkState(...)[0] is world position of link frame
    left_pos = p.getLinkState(body_id, left_idx, physicsClientId=physicsClientId)[0]
    right_pos = p.getLinkState(body_id, right_idx, physicsClientId=physicsClientId)[0]
    left_pos = np.array(left_pos)
    right_pos = np.array(right_pos)
    return np.linalg.norm(left_pos - right_pos)

def contact_flag(self, link_index: int) -> int:
        """Return 1 if there is at least one contact between the given panda
        link (link_index) and the currently loaded object, otherwise 0.
        """
        return int(bool(p.getContactPoints(self.pandaUid, self.objectUid, linkIndexA=link_index)))

def is_holding(self, left_flag: int, right_flag: int, dist: float, threshold: float = 0.02) -> int:
    """Return 1 when both fingers have contact and the fingertip distance
    exceeds threshold; otherwise 0.
    """
    return int(bool(left_flag and right_flag and dist > threshold))


def world_to_local(body_id, link_index, world_pos):
    if link_index == -1:
        body_pos, body_ori = p.getBasePositionAndOrientation(body_id)
    else:
        body_pos, body_ori = p.getLinkState(body_id, link_index)[:2]
    inv_pos, inv_ori = p.invertTransform(body_pos, body_ori)
    local_pos, _ = p.multiplyTransforms(inv_pos, inv_ori, world_pos, [0, 0, 0, 1])
    return local_pos

def local_coords(self,link):
    parent_pos, parent_orn = p.getLinkState(self.pandaUid, link)[0:2]
    child_pos, child_orn = p.getBasePositionAndOrientation(self.objectUid)
    parent_inv_pos, parent_inv_orn = p.invertTransform(parent_pos, parent_orn)
    child_in_parent_pos, child_in_parent_orn = p.multiplyTransforms(
        parent_inv_pos, parent_inv_orn, child_pos, child_orn
    )
    return child_in_parent_pos, child_in_parent_orn