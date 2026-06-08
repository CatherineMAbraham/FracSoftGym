import pybullet as p
import numpy as np
import time 
import pybullet_data
import os
from scipy.spatial.transform import Rotation as R
import wandb

INVALID_GOALS_PATH = os.path.join(os.path.dirname(__file__), 'invalid_goals.npy')

# Compare your poses:
# Pose A (Current): [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
# Pose B (Suggested): [0, -0.5, 0, -2.0, 0, 1.5, 0]
def make_scene(env):
    #Start Positions: Worked out previously
    
       if env.start_pos == 'home':
         startposition = np.array([0,-0.785,0,-2.356,0,1.571,0.785,0.04,0.04])
         #startposition = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79,0.04,0.04])
       elif env.start_pos == 'extended':
         startposition = np.array([0.03, 0.2, 0, -1.6, 0, -3, 0.8, -0.04, 0.04]) #-1.802, -2.89
       
       #load scene
       #Make Plane, Table, Cube       
       plane_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=np.array([30.0, 30.0, 0.01]))
       planecolour = [1, 0.94, 0.94, 1] # RGBA, Light pink color for the plane
       plane_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=np.array([30.0, 30.0, 0.01]),rgbaColor=planecolour)
       plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, 
                             baseVisualShapeIndex=plane_visual_shape,basePosition=[0, 0, -0.33])
       tableori = p.getQuaternionFromEuler([0, 0, 1.57])
       env.table =p.loadURDF("table/table.urdf", basePosition =[0.5,-0.45,-0.36] ,baseOrientation =tableori, globalScaling =0.5);#[0.8, 0.4, -0.33]

       env.visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.005,0.005,0.005], rgbaColor=[0.835, 0.7216, 1, 1])  # Purple Goal box - no collision properties

       #Set up robot with calculated start positions
       urdfRootPath=pybullet_data.getDataPath()
                  # 🔹 Create the base surgical table (static)
   
      
       env.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),
                                  basePosition=[-0,-0.06,-0.33],#[-0.5,0,-0.65],
                                  useFixedBase=True, globalScaling = 1)
       
       #p.changeDynamics(env.pandaUid,9, lateralFriction= 5,spinningFriction= 0.001)#,jointLowerLimit=0.00, jointUpperLimit=0.01)
       #p.changeDynamics(env.pandaUid,10, lateralFriction= 5,spinningFriction= 0.001)#,jointLowerLimit=0.00, jointUpperLimit=0.01)
       p.resetJointState(env.pandaUid,9, 0.04)
       p.resetJointState(env.pandaUid,10, 0.04) 

       for i in range(8):
           p.resetJointState(env.pandaUid,i, startposition[i])
        
    #    for _ in range(10):
    #        p.stepSimulation()
    #        time.sleep(0.002)
        
      # time.sleep(10)
           

       return env.pandaUid

def is_point_in_bone(env,point, bone_id):
    # Check distance between a coordinate and the entire mesh
    # distance > 0 means it is outside
    goal = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0, 0, 1], visualFramePosition=point)
    closest_points = p.getClosestPoints(bodyA=-1, bodyB=bone_id, 
                                        distance=0.01, # Search radius
                                        linkIndexA=-1, 
                                        positionA=point)
    
    if len(closest_points) > 0:
        # If the shortest distance is 0 or negative, it's a collision
        if closest_points[0][8] <= 0: 
            print(f'Point {point} is inside the bone (distance: {closest_points[0][8]:.4f} m)')
            return True
    return False

def is_goal_configuration_valid(env, goal_pos, goal_quat):
    """Checks if the foot/gripper would collide with the leg at the goal pose."""
    # 1. Save current real positions to restore them later
    joint_states = [p.getJointState(env.pandaUid, j) for j in range(9)]
    new_states  = p.calculateInverseKinematics(env.pandaUid, 11, targetPosition=goal_pos, 
                                                      targetOrientation=goal_quat, maxNumIterations=1000, residualThreshold=1e-9)
    #p.setJointMotorControlArray(env.pandaUid, range(9), controlMode=p.POSITION_CONTROL, targetPositions=new_states[:9])
    [p.resetJointState(env.pandaUid, i, new_states[i]) for i in range(9)]
    
    p.performCollisionDetection()
    
    # 4. Check for contact between the moved foot and the static leg
    contacts = p.getContactPoints(bodyA=env.foot, bodyB=env.leg)
    ## check how close it is to the goal to see if pose is physically possible
    position = p.getLinkState(env.pandaUid, 11)[0]
    orientation = p.getLinkState(env.pandaUid, 11)[1]
    pos, ori = calculate_distances(env, goal_pos, goal_quat, position, orientation)
    # print(f'Checking goal pose validity: Pos Dist={pos:.4f} m, Ori Dist={np.degrees(ori):.4f} deg, Contacts={len(contacts)}')
    # #print(f'ori check: {np.rad2deg(p.getEulerFromQuaternion(goal_quat))}')
    # if ori >0:
    #     print(f'Orientation is not valid {goal_quat}')
    # else:
    #     print(f'Orientation is valid {goal_quat}')
    # 5. Restore original position immediately
    #p.resetBasePositionAndOrientation(env.foot, orig_foot_pos, orig_foot_ori)
    for i in range(9):
           p.resetJointState(env.pandaUid,i, joint_states[i][0])
    #       time.sleep(1)
   # time.sleep(5)
    print('Back at home')
    if len(contacts) == 0 and ori <=env.distance_threshold_ori and pos <= env.distance_threshold_pos:
        valid = True
    else:
        valid = False
        #[print(f"Joint {i} attempted: {new_states[i]:.4f} rad") for i in range(9)]
        #print(f'Goal pose is invalid due to contact(s) with the leg.')
        print(f'Goal pose is invalid: Pos Dist={pos:.4f} m, Ori Dist={np.degrees(ori):.4f} deg, Contacts={len(contacts)}')
    # If len(contacts) > 0, the goal pose is physically impossible
    return valid# also check if orientation is within 30 degrees of goal orientation
def getGoal(env, fracturestart, fractureorientaionDeg):
    env.goal_gen_count += 1
    fracturestart = np.array(p.getLinkState(env.pandaUid, 11)[0] )
    #p.addUserDebugText('FS', fracturestart, textColorRGB=[1, 0, 0], textSize=1)
    #print('Fracture Start:', fracturestart)
    limit_low = [0.0125,0.008,0.003]
    limit_high = [0.0125,0.022,0.003]
    #print('Fracture Start:', fracturestart, 'Orientation:', fractureorientaionDeg)
    env.goal_range_low = fracturestart-limit_low #[0.0125,0.01,0.003]
    env.goal_range_high = fracturestart+ limit_high
    env.goal_ori_low= np.radians(fractureorientaionDeg - [15,5,15])
    env.goal_ori_high=np.radians(fractureorientaionDeg + [15,5,15])
    #print('Goal Pos Range Low:', env.goal_range_low, 'High:', env.goal_range_high,'Goal Ori Low:', env.goal_ori_low, 'High:', env.goal_ori_high)
    fracturestart_end = np.array(fracturestart - np.array([-0.01,0.045,0]))
    a = fracturestart - limit_low#[0.0125,0.0,-0.003] 
    b = fracturestart + limit_high#[-0.0125,0.03,0.003]
    c = fracturestart + limit_low#[0.0125,-0.0,0.003]
    d = fracturestart + limit_high#[0.0125,0.03,0.003]
    e = fracturestart - limit_low#[0.0125,0.0,0.103] 
    f = fracturestart + limit_high#[-0.0125,0.03,-0.103]
    g = fracturestart + limit_low#[0.0125,-0.0,-0.103]
    h = fracturestart + limit_high#[0.0125,0.03,-0.103]
    # p.addUserDebugLine(a, b, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(a, c, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(b, d, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(d, c, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(e, f, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(e, g, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(f, h, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(h, g, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(a, e, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(b, f, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(c, g, lineColorRGB=[0, 1, 0], lineWidth=3)
    # p.addUserDebugLine(d, h, lineColorRGB=[0, 1, 0], lineWidth=3)
    #print(env.curriculum_phase)
    # if env.curriculum_phase ==1:
    #     env.goal_pos = fracturestart.copy()
    # else:0
    env.goal_pos = np.array(env.np_random.uniform(env.goal_range_low, env.goal_range_high,))
    ##choose the most extreme goal position for debugging purposes, can change to random within range later
    #env.goal_pos = np.array([env.goal_range_low[0], env.goal_range_low[1], env.goal_range_low[2]])
    #print('Goal Position:', env.goal_pos)
    if env.action_type== 'fouractions':
        env.goal_pos[2] = fracturestart[2]
        env.goal_ori_low[1] =np.radians(fractureorientaionDeg[1] - 0)
        env.goal_ori_high[1] =np.radians(fractureorientaionDeg[1]+0)    
    
    #env.goal_pos = np.round(goal_pos,3)
    ori = np.array(env.np_random.uniform(env.goal_ori_low, env.goal_ori_high))
    env.goal_ori = np.array(p.getQuaternionFromEuler(ori))
    

    #goal_ori = R.from_euler('xyz', ori).as_quat()
    #env.goal_ori = np.round(goal_ori,3)
    valid = is_goal_configuration_valid(env, env.goal_pos, env.goal_ori)#
    
    while not valid:
        env.not_valid_count += 1
        invalid_goal = np.array(
            [(np.asarray(env.goal_pos, dtype=float), np.asarray(env.goal_ori, dtype=float))],
            dtype=[('pos', float, (3,)), ('ori', float, (4,))]
        )

        if os.path.exists(INVALID_GOALS_PATH):
            existing_goals = np.load(INVALID_GOALS_PATH)
            invalid_goals = np.concatenate((existing_goals, invalid_goal))
        else:
            invalid_goals = invalid_goal

        np.save(INVALID_GOALS_PATH, invalid_goals)
        print(f'Invalid Percentage: {env.not_valid_count/env.goal_gen_count:.2%} | Invalid Count: {env.not_valid_count} | Total Generated: {env.goal_gen_count}')
        env.goal_pos = np.array(env.np_random.uniform(env.goal_range_low, env.goal_range_high,))
        ori = np.array(env.np_random.uniform(env.goal_ori_low, env.goal_ori_high))
        goal_ori = np.array(p.getQuaternionFromEuler(ori))
        env.goal_ori = np.round(goal_ori,3)
        valid = is_goal_configuration_valid(env, env.goal_pos, env.goal_ori)
        env.goal_gen_count += 1 
    
def get_youngs_modulus_and_width(env):
    youngs_modulus_range = range(1000000, 100000000, 1000)
    width_range = np.arange(0.001, 0.01, 0.001)

    ## select random values from the ranges
    youngs_modulus = env.np_random.choice(youngs_modulus_range) 
    width = env.np_random.choice(width_range)
    #print(f'Youngs Modulus: {youngs_modulus} Pa, Width: {width} m')
    return youngs_modulus, width

def getStarts(env):
    fracturestart= np.array(p.getLinkState(env.pandaUid, 11)[0] )
    fractureorientaionRad =p.getEulerFromQuaternion(p.getLinkState(env.pandaUid, 11)[1])
    fractureorientaionDeg = np.degrees(np.array(fractureorientaionRad)) 
    #pin = [0.004462 ,-0.002332 , 0.046608  ]
   # pin = [0.004462 ,-0.002332 , 0.049608  ]
    #p.addUserDebugText('P', pin, textColorRGB=[1, 0, 0], textSize=1)
    fracturestart = fracturestart - [-0.05,0,0]
    #Calculated this difference from the object start position
    #difference = [-0.004493, 0.079895+0.005, 0.073322] difference between leg and foot
    #difference = [0.011489 ,-0.045611 ,-0.006535  ]
    difference = [0.0,0.005,0]
    difference =np.array(difference)
    #legstart=[]
    # for i in range(len(difference)):
    #     leg = (fracturestart[i])-(difference[i])
    #     legstart.append(leg)
    #     i+=1
    

    return fracturestart, fractureorientaionDeg#, legstart



def get_new_pose(env, dx, dy, dz, qx, qy, qz, qw=None, mode=None):
        currentPose = p.getLinkState(env.pandaUid, 11, 1)
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
            newPosition = np.clip(newPosition, env.goal_range_low, env.goal_range_high)
            return newPosition, newOrientation

        elif mode in ['euler', 'fouractions', 'ori_only']:
            deltaorE = [qx, qy, qz]
            deltaor = p.getQuaternionFromEuler(deltaorE)
            if mode == 'ori_only':
                newPosition = currentPosition
            else:
                newPosition = currentPosition + np.array([dx, dy, dz])
            #newPosition = np.clip(newPosition, env.goal_range_low, env.goal_range_high)
            newOrientation = np.array(p.multiplyTransforms([0, 0, 0], currentOrientation, [0, 0, 0], deltaor)[1])
            #ensure normalised quaternion
            #check if quat is normalised 
            #norm = np.linalg.norm(newOrientation)
            norm = newOrientation[0]**2 + newOrientation[1]**2 + newOrientation[2]**2 + newOrientation[3]**2
            if norm > 1+1e-6 or norm < 1-1e-6:
                print(f'not normalised! {norm}')
            #newOrientation = newOrientation / np.linalg.norm(newOrientation)
            #euler = p.getEulerFromQuaternion(newOrientation)
            #newOrientationE = np.clip(euler, env.goal_ori_low, env.goal_ori_high)
            #newOrientation = p.getQuaternionFromEuler(newOrientationE)
            return newPosition, newOrientation

        
        elif mode == 'pos_only':
            newPosition = currentPosition + np.array([qx, qy, qz])
            #print(newPosition)
            #p.addUserDebugText('NP', newPosition, textColorRGB=[0, 0, 1], textSize=1, lifeTime=0.5)
            newOrientation = currentOrientation
            #newPosition[2] = np.clip(newPosition[2], env.goal_range_low[2], env.goal_range_high[2])
            #newPosition = np.clip(newPosition, (env.goal_range_low), (env.goal_range_high))
            #p.addUserDebugText('NP', newPosition, textColorRGB=[0, 1, 0], textSize=1, lifeTime=0.5)
            return newPosition, newOrientation

        
        # else:
        #     newPosition = currentPosition + np.array([dx, dy, dz])
        #     newPosition = np.clip(newPosition, env.goal_range_low, env.goal_range_high)
        #     newOrientation = np.array([qx, qy, qz])
        return newPosition, newOrientation

def unpack_action(env, action):
    zeros = [0] * 10
    if env.action_type in ['ori_only', 'pos_only']:
        return [0, 0, 0, action[0] * dv, action[1] * dv, action[2] * dv, 0, 0, 0, 0]
    elif env.action_type == 'quat':
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, action[6] * dv, 0, 0, 0]
    elif env.action_type == 'joint':
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, action[6] * dv, action[6] * dv, action[7] * dv, action[8] * dv]
    elif env.action_type == 'fiveactions':
        return [action[0] * dv, action[1] * dv, 0, action[2] * dv, action[3] * dv, action[4] * dv, 0, 0, 0, 0]
    elif env.action_type == 'fouractions':
        return [action[0] * dv, action[1] * dv, 0, action[2] * dv*10, 0, action[3] * dv*10, 0, 0, 0, 0]
    else:
        return [action[0] * env.dt, action[1] * env.dt, action[2] * env.dt, action[3] * env.dr, action[4] * env.dr, action[5] * env.dr, 0, 0, 0, 0]


def calculate_distances(env, new_pos, new_ori, goal_pos, goal_ori):
    # Calculate positional distance (Euclidean distance)
    env.pos_distance = (np.linalg.norm(np.array(new_pos) - np.array(goal_pos), axis=-1)) #the new distance
    
    # Calculate the dot product between the quaternions
    dot_product = np.abs(np.sum(new_ori * goal_ori, axis=-1))
    
    
    # Ensure the dot product is within the valid range for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle (rotational distance) between the quaternions
    env.angle = 2 * np.arccos(dot_product)
    
    return env.pos_distance, env.angle
    
def fingertip_distance(body_id, left_idx, right_idx, physicsClientId=0):
    # getLinkState(...)[0] is world position of link frame
    left_pos = p.getLinkState(body_id, left_idx, physicsClientId=physicsClientId)[0]
    right_pos = p.getLinkState(body_id, right_idx, physicsClientId=physicsClientId)[0]
    left_pos = np.array(left_pos)
    right_pos = np.array(right_pos)
    return np.linalg.norm(left_pos - right_pos)

def contact_flag(env, link_index: int) -> int:
        """Return 1 if there is at least one contact between the given panda
        link (link_index) and the currently loaded object, otherwise 0.
        """
        return int(bool(p.getContactPoints(env.pandaUid, env.foot, linkIndexA=link_index)))

def is_holding(env, left_flag: int, right_flag: int, dist: float, threshold: float = 0.02) -> int:
    """Return 1 when both fingers have contact and the fingertip distance
    exceeds threshold; otherwise 0.
    """
    return int(bool(left_flag and right_flag and dist > threshold))


def world_to_local(env, link_index, world_pos):
    if link_index == -1:
        body_pos, body_ori = p.getBasePositionAndOrientation(env.foot)
    else:
        body_pos, body_ori = p.getLinkState(env.pandaUid, link_index)[:2]
    inv_pos, inv_ori = p.invertTransform(body_pos, body_ori)
    local_pos, _ = p.multiplyTransforms(inv_pos, inv_ori, world_pos, [0, 0, 0, 1])
    return local_pos

def local_coords(env,link):
    parent_pos, parent_orn = p.getLinkState(env.pandaUid, link)[0:2]
    child_pos, child_orn = p.getBasePositionAndOrientation(env.foot)
    parent_inv_pos, parent_inv_orn = p.invertTransform(parent_pos, parent_orn)
    child_in_parent_pos, child_in_parent_orn = p.multiplyTransforms(
        parent_inv_pos, parent_inv_orn, child_pos, child_orn
    )
    return child_in_parent_pos, child_in_parent_orn


def smooth_motion(env, joint_targets, joint_current, maxforce,numsubsteps):
    max_step_force = 0 
    force_total = 0
    all_forces=[]
    for i in range(numsubsteps):
        alpha = (i + 1) / numsubsteps
        intermediate_targets = joint_current + alpha * (joint_targets - joint_current)
        p.setJointMotorControlArray(
            env.pandaUid,
            jointIndices=range(9),
            controlMode=p.POSITION_CONTROL,
            targetPositions=intermediate_targets.tolist(),
            forces=maxforce
        )
        
        if env.soft_tissue == 'spring':
            #print('Stepping spring')
            env.band.step()
        #print('stepping')
        p.stepSimulation()
        joint_current = np.array([p.getJointState(env.pandaUid, j)[0] for j in range(9)])
        force = p.getJointState(env.foot, 0)[2]  # Joint index 0 is the fixed joint
        all_forces.append(force)
        force_magnitude = np.linalg.norm(force[:3])  # Magnitude of the force vector}])
        force = force_magnitude
        force_total += force
        #forces.append(force)
        #p.addUserDebugText(f'Force: {force:.2f} N', [0.5, 0, 0.5], textColorRGB=[1, 0, 0], textSize=1, lifeTime=0.1)
        if force > max_step_force: ## step max force
            max_step_force = force
            if max_step_force > env.output_force: ##episode max force 
                env.output_force = max_step_force
                #print('New Max Force:', env.output_force)
            

    return env.output_force, max_step_force, force_total/numsubsteps, np.mean(all_forces,axis=0)
        

def drawAABB(env,object,link):
    aabb = p.getAABB(object,link)
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 0, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [0, 1, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [0, 0, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])