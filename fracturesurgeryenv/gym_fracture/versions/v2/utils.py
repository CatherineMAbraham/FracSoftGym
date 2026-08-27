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
       #startposition = np.array([0.03, 0.2, 0, -1.6, 0, 1.571, 0.8, -0.04, 0.04])
        #-1.802, -2.89  ##home position of franka 
       #startposition = np.array([0,-0.785,0,-2.356,0,-3,0.785,0.04,0.04])
    #([0.03, 0.2, 0, -1.805, 0, 2, 0.61, -0.04, 0.04])
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
                  # Create the base surgical table (static)
    #    table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.1, 0.002])
    #    table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.1, 0.002], rgbaColor=[0.3, 0.3, 0.3, 1])
    #    table_body = p.createMultiBody(
    #         baseMass=0,
    #         baseCollisionShapeIndex=table_collision,
    #         baseVisualShapeIndex=table_visual,
    #         basePosition=[0.65, 0.05, 0.005],
    #     )
    #    p.changeDynamics(table_body, -1, lateralFriction=0.1, restitution=0.0)

    #     # Create a soft pad (a smaller box resting on the table)
    #    pad_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.1, 0.02])
    #    pad_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.1, 0.02], rgbaColor=[0.8, 0.2, 0.2, 1])
    #    pad_body = p.createMultiBody(
    #         baseMass=0,  # static pad
    #         baseCollisionShapeIndex=pad_collision,
    #         baseVisualShapeIndex=pad_visual,
    #         basePosition=[0.8, 0.15, 0.05],  # slightly above table
    #     )
    #    p.changeDynamics(pad_body, -1, lateralFriction=1.5, restitution=0.0)
      
       env.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),
                                  basePosition=[-0,-0.06,-0.33],#[-0.5,0,-0.65],
                                  useFixedBase=True, globalScaling = 1)
       
       #p.changeDynamics(env.pandaUid,9, lateralFriction= 5,spinningFriction= 0.001)#,jointLowerLimit=0.00, jointUpperLimit=0.01)
       #p.changeDynamics(env.pandaUid,10, lateralFriction= 5,spinningFriction= 0.001)#,jointLowerLimit=0.00, jointUpperLimit=0.01)
       p.resetJointState(env.pandaUid,9, 0.04)
       p.resetJointState(env.pandaUid,10, 0.04) 
       #test = p.calculateInverseKinematics(env.pandaUid, 11, targetPosition=[0.35701957, -0.06,0.15526956], targetOrientation=p.getQuaternionFromEuler([0, 2.80671241e-10, 4.50000000e+01]), maxNumIterations=1000, residualThreshold=1e-9)
      # test = np.concatenate([test[:9], [0.04, 0.04]])
       for i in range(8):
           p.resetJointState(env.pandaUid,i, startposition[i])

       
       if env.randomise_start==True:
           random = np.random.uniform(-0.005, 0.005, size=2)
           end_effector_pos = p.getLinkState(env.pandaUid, 11)[0] + np.array([random[0], random[1], 0])  # small random offset
           random_joint_positions = p.calculateInverseKinematics(env.pandaUid, 11, targetPosition=end_effector_pos, maxNumIterations=1000, residualThreshold=1e-9)
           for i in range(9):
               p.resetJointState(env.pandaUid, i, random_joint_positions[i])
   
           

       return env.pandaUid



def is_goal_configuration_valid(env, goal_pos, goal_quat):
    """Checks if the foot/gripper would collide with the leg at the goal pose."""
    # 1. Save current real positions to restore them later
    joint_states = [p.getJointState(env.pandaUid, j) for j in range(9)]
    new_states  = p.calculateInverseKinematics(env.pandaUid, 11, targetPosition=goal_pos, 
                                                      targetOrientation=goal_quat, maxNumIterations=1000, residualThreshold=1e-9)
    #p.setJointMotorControlArray(env.pandaUid, range(9), controlMode=p.POSITION_CONTROL, targetPositions=new_states[:9])
    [p.resetJointState(env.pandaUid, i, new_states[i]) for i in range(9)]
    # p.setJointMotorControlArray(
    #             env.pandaUid,
    #             jointIndices=range(9),
    #             controlMode=p.POSITION_CONTROL,
    #             targetPositions=new_states[:9],
    #             #forces=maxforce
    #         )
    # for _ in range(100):
    #     p.stepSimulation()
    #     time.sleep(0.1)
    #time.sleep(5)  # Allow physics to update after moving the foot
    p.performCollisionDetection()
    
    # 4. Check for contact between the moved foot and the static leg
    contacts = p.getContactPoints(bodyA=env.foot, bodyB=env.leg, linkIndexA=1, linkIndexB=-1)
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
          # time.sleep(1)
    #time.sleep(5)
    #print('Back at home')
    if len(contacts) == 0 and ori <=env.distance_threshold_ori and pos <= env.distance_threshold_pos:
        valid = True
    else:
        valid = False
        #[print(f"Joint {i} attempted: {new_states[i]:.4f} rad") for i in range(9)]
        #print(f'Goal pose is invalid due to contact(s) with the leg.')
        print(f'Goal pose is invalid: Pos Dist={pos} m, Ori Dist={np.degrees(ori)} deg, Contacts={len(contacts)}')
    # If len(contacts) > 0, the goal pose is physically impossible
    return valid# also check if orientation is within 30 degrees of goal orientation
# def is_goal_in_range(env):
#     goal_high = np.array([ 0.31951126, -0.03799936,  0.15826347])
#     goal_low = [ 0.29451126, -0.06799936,  0.15226347]
#     goal_ori_high = [3.40339204,0.08726646,0.26219755]
#     goal_ori_low = [ 2.87979327, -0.08726646, -0.26140122]
#     target_pos = env.target_position[0:3]
#     target_ori = (p.getEulerFromQuaternion(env.target_position[3:7]))
#     for i in range(3):
#         if target_pos[i] < goal_low[i] or target_pos[i] > goal_high[i]:
#             print(f'Axis {i} out of range: {target_pos[i]} not in [{goal_low[i]}, {goal_high[i]}]')
#             env.target_position[i] = np.clip(env.target_position[i], goal_low[i], goal_high[i])
#             env.target_position[i]-=0.001
#     for i in range(3):
#         if target_ori[i] < goal_ori_low[i] or target_ori[i] > goal_ori_high[i]:
#             print(f'Orientation axis {i} out of range: {target_ori[i]} not in [{goal_ori_low[i]}, {goal_ori_high[i]}]')
#             target_ori[i] = np.clip(target_ori[i], goal_ori_low[i], goal_ori_high[i])
#             env.target_position[3:7] = p.getQuaternionFromEuler(target_ori)
#     return env.target_position
def is_goal_in_range(env, pos_buffer=0.001, ori_buffer=0.005):
    goal_high = np.array([0.31951126, -0.03799936, 0.15826347])
    goal_low = np.array([0.29451126, -0.06799936, 0.15226347])

    goal_ori_high = np.array([3.40339204, 0.08726646, 0.26219755])  # Radians
    goal_ori_low = np.array([2.87979327, -0.08726646, -0.26140122])  # Radians

    # Apply buffer inwards from boundaries
    buffered_pos_low = goal_low + pos_buffer
    buffered_pos_high = goal_high - pos_buffer

    buffered_ori_low = goal_ori_low + ori_buffer
    buffered_ori_high = goal_ori_high - ori_buffer

    target_pos = np.array(env.target_position[0:3])
    target_ori = np.array(p.getEulerFromQuaternion(env.target_position[3:7]))

    # --- 1. Position Check & Inward Clip ---
    for i in range(3):
        if target_pos[i] < goal_low[i] or target_pos[i] > goal_high[i]:
            print(
                f"Axis {i} out of range: {target_pos[i]:.6f} not in [{goal_low[i]}, {goal_high[i]}]"
            )

    clipped_pos = np.clip(target_pos, buffered_pos_low, buffered_pos_high)
    env.target_position[0:3] = clipped_pos

    # --- 2. Orientation Check & Inward Clip ---
    for i in range(3):
        if target_ori[i] < goal_ori_low[i] or target_ori[i] > goal_ori_high[i]:
            print(
                f"Orientation axis {i} out of range: {target_ori[i]:.6f} not in [{goal_ori_low[i]}, {goal_ori_high[i]}]"
            )

    clipped_ori = np.clip(target_ori, buffered_ori_low, buffered_ori_high)

    # Convert directly back to Quaternion (no np.radians)
    env.target_position[3:7] = p.getQuaternionFromEuler(clipped_ori)

    return env.target_position

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
   # print(f'Fracture Start: {fracturestart}, Orientation: {fractureorientaionDeg}')
    #print(f'Goal Pos Range Low: {env.goal_range_low}, High: {env.goal_range_high}')
   # print(f'Goal Ori Low: {(env.goal_ori_low)}, High: {(env.goal_ori_high)}')
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
    #valid = is_goal_configuration_valid(env, env.goal_pos, env.goal_ori)#
    #print('Generated Goal Position:', env.goal_pos, 'Orientation (Euler):', np.degrees(ori), 'Valid:', valid)
    # while not valid:
    #     env.not_valid_count += 1
    #     invalid_goal = np.array(
    #         [(np.asarray(env.goal_pos, dtype=float), np.asarray(env.goal_ori, dtype=float))],
    #         dtype=[('pos', float, (3,)), ('ori', float, (4,))]
    #     )

    #     if os.path.exists(INVALID_GOALS_PATH):
    #         existing_goals = np.load(INVALID_GOALS_PATH)
    #         invalid_goals = np.concatenate((existing_goals, invalid_goal))
    #     else:
    #         invalid_goals = invalid_goal

    #     np.save(INVALID_GOALS_PATH, invalid_goals)
    #     print(f'Invalid Percentage: {env.not_valid_count/env.goal_gen_count:.2%} | Invalid Count: {env.not_valid_count} | Total Generated: {env.goal_gen_count}')
    #     env.goal_pos = np.array(env.np_random.uniform(env.goal_range_low, env.goal_range_high,))
    #     ori = np.array(env.np_random.uniform(env.goal_ori_low, env.goal_ori_high))
    #     env.goal_ori = np.array(p.getQuaternionFromEuler(ori))
    #     valid = is_goal_configuration_valid(env, env.goal_pos, env.goal_ori)
    #     env.goal_gen_count += 1 
    
def get_youngs_modulus_and_width(env):
    youngs_modulus_range =np.arange(1e5, 5e6, 1e3) #1MPa to 20MPa in 1kPa increments
    width_range = np.round(np.arange(0.001, 0.01, 0.001), 3)

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
    fracturestart = fracturestart - [-0.04,-0.03,0.08]#[-0.05,0,0]#
    #Calculated this difference from the object start position
    #difference = [-0.004493, 0.079895+0.005, 0.073322] difference between leg and foot
    #difference = [0.011489 ,-0.045611 ,-0.006535  ]
    difference = [0.05,0,-0.08]
    difference =np.array(difference)
    #legstart=[]
    # for i in range(len(difference)):
    #     leg = (fracturestart[i])-(difference[i])
    #     legstart.append(leg)
    #     i+=1
    #print(f'Fracture Start: {fracturestart}, Orientation: {fractureorientaionDeg}, {fractureorientaionRad}')
    #fracturestart = np.array([0.37791427969932556, -0.14127257466316223, 0.06339067965745926])
    #fracturestart = np.array([0.35706911463540575, -0.06982252591466533, 0.07526190835600088])
    ##foot start :foot start: 
    # if isinstance(env.goal_type, str):
    #     pass
    # else:
    if isinstance(env.goal_type, np.ndarray):
        fracturestart=np.array([ 0.34701957, -0.03,0.07526956])#-np.array([0,0.03,0])#(0.3487603762848476, -0.08310808157863893, 0.07322828156663022)#([0.35706911463540575, -0.06982252591466533, 0.07526190835600088])#[ 0.3468140278354482, -0.029897614059223178, 0.07524706439920568]
    #print(fracturestart)
    # [ 0.30702814 -0.06013873  0.15526305], foot ori: [ 4.33410682e-04 -1.40125743e-04  7.08949001e-01  7.05259602e-01], goal pos: [ 0.30729738 -0.03948561  0.15312103], goal ori: [ 0.99281821 -0.01780929  0.0186322  -0.11682327]
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
        return [0, 0, 0, action[0] * env.dv, action[1] * env.dv, action[2] * env.dv, 0, 0, 0, 0]
    elif env.action_type == 'quat':
        return [action[0] * env.dv, action[1] * env.dv, action[2] * env.dv, action[3] * env.dv, action[4] * env.dv, action[5] * env.dv, action[6] * env.dv, 0, 0, 0]
    elif env.action_type == 'joint':
        return [action[0] * env.dv, action[1] * env.dv, action[2] * env.dv, action[3] * env.dv, action[4] * env.dv, action[5] * env.dv, action[6] * env.dv, action[6] * env.dv, action[7] * env.dv, action[8] * env.dv]
    elif env.action_type == 'fiveactions':
        return [action[0] * env.dv, action[1] * env.dv, 0, action[2] * env.dv, action[3] * env.dv, action[4] * env.dv, 0, 0, 0, 0]
    elif env.action_type == 'fouractions':
        return [action[0] * env.dv, action[1] * env.dv, 0, action[2] * env.dv*10, 0, action[3] * env.dv*10, 0, 0, 0, 0]
    else:
        return [action[0] * env.dt, action[1] *	env.dt, action[2] *	env.dt,	action[3] *	env.dr,	action[4] *	env.dr,	action[5] *	env.dr,	0,	0,	0,	0]


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

def constrain_quat(env, q):
    q_rel = R.from_quat(q) * R.from_quat(env.goal_ori).inv()  # Relative rotation  
    
    angle = np.linalg.norm(q_rel.as_rotvec())  # Get angle in radians  

    max_angle = np.deg2rad(30)  # 30-degree limit  
    if angle > max_angle:
        scale = max_angle / angle
        q_rel = R.from_rotvec(q_rel.as_rotvec() * scale)  # Scale down rotation  
        q = (q_rel * R.from_quat(env.goal_ori)).as_quat() 
    
    return q # Apply scaled rotation back  


def visualize_contact_forces(env,bodyA, bodyB, scale=0.01, lifeTime=0.05, lineWidth=2):
    """
    Draw contact normal, friction vectors and total force for every contact between bodyA and bodyB.
    - bodyA: robot (or the contacting body)
    - bodyB: object (the body receiving force in c[...] interpretation used here)
    - scale: visual scaling factor (1 meter per 1 N would be huge; use ~0.001-0.05)
    """
    contacts = p.getContactPoints(bodyA=bodyA, bodyB=bodyB, linkIndexB=-1)
    #print(contacts)
    for c in contacts:
        # contact point in world frame (use the reported contact position)
        #print('Contact Info:', c)
        linkIndex = c[3]  # link index on body A (robot)
        #print('Contact Link Index:', linkIndex)
        contact_pos = c[6]  # commonly used in examples; point on body B
        # normal on body B (unit vector), points away from B toward A
        normal_dir = np.array(c[7], dtype=float)
        normal_mag = float(c[9])    # normal force magnitude (N)
        #print('Normal Mag:', normal_mag)
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
        #print(f_normal, f_t1, f_t2, f_total)
#        p.addUserDebugLine(normal_dir,)
        forces = [p.getJointState(env.pandaUid, i)[3] for i in range(9)]
        q_list = [p.getJointState(env.pandaUid, j)[0] for j in range(9)]
        v_list = [p.getJointState(env.pandaUid, j)[1] for j in range(9)]
        tau = np.array(forces)
        # linkIndex = index of end-effector link
        link_state = p.getLinkState(env.pandaUid, linkIndex, computeForwardKinematics=True)
        link_world_pos = np.array(link_state[0])
        link_world_orn = np.array(link_state[1])  # quaternion (x,y,z,w)

        # convert contact world position to link local coordinates
        # build rotation matrix from quaternion
        rot_mat = np.array(p.getMatrixFromQuaternion(link_world_orn)).reshape(3,3)
        local_pos = rot_mat.T.dot(contact_pos - link_world_pos)  # local coords in link frame

        # calculate Jacobian at that local position
        lin_jac, ang_jac = p.calculateJacobian(env.pandaUid, linkIndex,
                                            localPosition=list(local_pos),
                                            objPositions=q_list,
                                            objVelocities=v_list,
                                            objAccelerations=[0.0]*len(q_list))
        J_lin = np.array(lin_jac)   # shape (3, n)
        J_ang = np.array(ang_jac)   # shape (3, n)

        # predicted joint torques from force only (tau = J^T * F)
        tau_pred_from_force = J_lin.T.dot(f_total)   # (n,) vector, Nm

        #print("contact f_total world (N):", f_total, "||mag||:", np.linalg.norm(f_total))
        #print("predicted tau from linear force (Nm):", tau_pred_from_force)

        # optionally also include any contact moment (if you have it) via angular jacobian
        # compare to measured joint torques:
        measured_taus = np.array([p.getJointState(env.pandaUid, j)[3] for j in range(9)])
        #print("measured joint torques (Nm):", measured_taus)
        #print("difference (meas - predicted):", measured_taus - tau_pred_from_force)
        f_total = np.linalg.norm(f_total)
        f_total = np.float32(f_total)
        #print('Total Contact Force:', f_total)
        return f_total
    
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


def compute_target_velocity(desired_pos, current_pos, current_vel, dt,
                            max_speed, Kd=0.01, desired_vel=None):
    if desired_vel is None:
        desired_vel = np.zeros_like(current_vel)
    
    # Step 1: base proportional velocity
    prop_vel = (desired_pos - current_pos) / dt
    prop_vel_np = np.array(prop_vel, dtype=float)
    #print('Unclamped Vel:', prop_vel_np)
    # Step 2: clamp to max speed
    prop_vel_clamped_np = np.zeros(len(prop_vel_np), dtype=float)
    for i in range(len(prop_vel_np)):
        #print('Unclamped Vel:', vel)
        v = max_speed[i]
        prop_vel_clamped_np[i] = np.clip(prop_vel_np[i], -v, v)
        #vels.append(vel)
    #print('Clamped Vel:', prop_vel_clamped_np)
    # Step 3: damping correction
    #print(Kd * (desired_vel - current_vel))
    damping = np.array(Kd * (desired_vel - current_vel), dtype=float)
    prop_vel_clamped_np += damping

    return prop_vel_clamped_np


def move_panda_smoothly(env,robot_id, joint_indices, target_positions,
                        max_speeds, Kd=0.01, max_force=20,
                        dt=0.01, tolerance=1e-3, sleep_time=None):
    """
    Smoothly move Panda to target_positions using per-joint velocity control.
    """
    target_positions = np.array(target_positions, dtype=float)
    max_speeds = np.array(max_speeds, dtype=float)

    # Initialize current joint positions and velocities
    q_current = np.array([p.getJointState(robot_id, j)[0] for j in joint_indices])
    v_current = np.array([p.getJointState(robot_id, j)[1] for j in joint_indices])

    while np.linalg.norm(target_positions - q_current) > tolerance:
        # Compute target velocities using per-joint limits
        target_velocities = compute_target_velocity(
            desired_pos=target_positions,
            current_pos=q_current,
            current_vel=v_current,
            dt=dt,
            max_speed=max_speeds,
            Kd=Kd
        )

        # Apply velocity control
        p.setJointMotorControlArray(
            robot_id,
            jointIndices=joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_velocities.tolist(),
            forces=max_force
        )

        # Step simulations
        env.band.step()
        p.stepSimulation()
        #print('Target Velocities:', target_velocities)
        if sleep_time:
            time.sleep(sleep_time)

        # Update current positions and velocities
        q_current = np.array([p.getJointState(robot_id, j)[0] for j in joint_indices])
        v_current = np.array([p.getJointState(robot_id, j)[1] for j in joint_indices])

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
        #time.sleep(0.01)
        joint_current = np.array([p.getJointState(env.pandaUid, j)[0] for j in range(9)])
        force = p.getJointState(env.foot, 1)[2]  # Joint index 1 is the fixed joint
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
        
def compute_ee_forward_dynamics(
    robot_id,
    ee_link_index,
    joint_indices,
    joint_torques,
    eps=1e-4
):
    # -------------------------
    # 1. State
    # -------------------------
    q = np.array([p.getJointState(robot_id, j)[0] for j in joint_indices])
    qdot = np.array([p.getJointState(robot_id, j)[1] for j in joint_indices])

    # -------------------------
    # 2. Mass matrix (M)
    # -------------------------
    M = np.array(p.calculateMassMatrix(robot_id, q))

    # -------------------------
    # 3. Nonlinear terms C(q,qdot) + g(q)
    # -------------------------
    Cg = np.array(p.calculateInverseDynamics(robot_id, q, qdot, [0]*len(joint_indices)))

    # -------------------------
    # 4. Joint accelerations: qdd = M⁻¹(τ - Cg)
    # -------------------------
    tau = np.array(joint_torques)
    qdd = np.linalg.solve(M, tau - Cg)

    # -------------------------
    # 5. Jacobian at current state
    # -------------------------
    zero_local = [0, 0, 0]
    Jv, Jw = p.calculateJacobian(
        robot_id,
        ee_link_index,
        zero_local,
        q.tolist(),
        qdot.tolist(),
        [0]*len(joint_indices)
    )
    Jv = np.array(Jv)
    Jw = np.array(Jw)

    # -------------------------
    # 6. Jacobian at q + eps * qdot  (finite-difference Jdot)
    # -------------------------
    q_eps = (q + eps * qdot).tolist()
    Jv2, Jw2 = p.calculateJacobian(
        robot_id,
        ee_link_index,
        zero_local,
        q_eps,
        qdot.tolist(),
        [0]*len(joint_indices)
    )
    Jv2 = np.array(Jv2)
    Jw2 = np.array(Jw2)

    Jv_dot = (Jv2 - Jv) / eps
    Jw_dot = (Jw2 - Jw) / eps

    # -------------------------
    # 7. End-effector accelerations
    # -------------------------
    xdd = Jv @ qdd + Jv_dot @ qdot     # linear acceleration
    wdd = Jw @ qdd + Jw_dot @ qdot     # angular acceleration

    return qdd, xdd, wdd

def compute_end_effector_force(robot_id, ee_link_index, joint_indices):
    """
    Compute the estimated end-effector force using:
        F = (J^T)^+ * τ
    where τ are the joint torques returned by PyBullet.
    """
    # -------------------------------------------
    # 1. Get joint torques (reaction torques)
    # -------------------------------------------
    tau = []
    for j in joint_indices:
        js = p.getJointState(robot_id, j)
        tau.append(js[3])   # joint reaction torque from physics
    tau = np.array(tau)

    # -------------------------------------------
    # 2. Get joint positions/velocities for Jacobian
    # -------------------------------------------
    q = []
    qdot = []
    for j in joint_indices:
        js = p.getJointState(robot_id, j)
        q.append(js[0])
        qdot.append(js[1])

    # -------------------------------------------
    # 3. Compute Jacobian at the end effector
    # -------------------------------------------
    link_state = p.getLinkState(robot_id, ee_link_index)
    link_pos = link_state[0]

    Jv, Jw = p.calculateJacobian(
        robot_id,
        ee_link_index,
        localPosition=[0, 0, 0],
        objPositions=q,
        objVelocities=qdot,
        objAccelerations=[0]*len(q),
    )

    Jv = np.array(Jv)
    Jw = np.array(Jw)

    # Full 6xN geometric Jacobian
    J = np.vstack([Jv, Jw])
    A = J.T                         # (N, 6)
    w, residuals, rank, s = np.linalg.lstsq(A, tau, rcond=None)
    # -------------------------------------------
    # 4. Compute EE wrench F = (J^T)^+ τ
    # -------------------------------------------
    # JT = J.T
    # JT_pinv = np.linalg.pinv(JT)

    # F = JT_pinv @ tau

    # F contains:
    #   F[0:3] = force (x,y,z)
    #   F[3:6] = torque (roll,pitch,yaw)
    return w

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
    


def apply_cbf_safety_filter_diagnostic(
    env,
    dx,
    dy,
    dz,
    gamma=1.0,
    stiffness=100.0,
    max_correction_step=0.001,
    alpha=0.6,
    workspace_min=None,
    workspace_max=None,
    debug_viz=True,
    step_idx=0,
):
    ee_pos = np.array(env.get_ee_pos()) if hasattr(env, "get_ee_pos") else None
    contacts = p.getContactPoints(bodyA=env.foot, bodyB=env.leg)

    valid_contacts = [pt for pt in contacts if pt[9] > 0.0] if contacts else []
    v_nom = np.array([dx, dy, dz])

    if not valid_contacts:
        env.f_smooth = (1.0 - alpha) * env.f_smooth
        v_safe = v_nom.copy()
        if ee_pos is not None and workspace_min is not None and workspace_max is not None:
            v_safe = np.clip(ee_pos + v_safe, workspace_min, workspace_max) - ee_pos
        return v_safe[0], v_safe[1], v_safe[2]

    # 1. Force Aggregation
    raw_force_sum = sum(pt[9] for pt in valid_contacts)
    raw_force_max = max(pt[9] for pt in valid_contacts)
    env.f_smooth = alpha * raw_force_max + (1.0 - alpha) * env.f_smooth

    # 2. Corrected Push Normal Direction (Points INTO the bone surface)
    normals = [np.array(pt[7]) for pt in valid_contacts]  # REMOVED NEGATION
    n_push = np.mean(normals, axis=0)
    norm_len = np.linalg.norm(n_push)
    if norm_len < 1e-8:
        return dx, dy, dz
    n_push /= norm_len

    # 3. Barrier Margin with Active Back-off when Over Threshold
    f_thresh = env.maximum_contact_force_threshold
    h_force = f_thresh - env.f_smooth

    # Allows negative max_allowed_push_step to actively push robot OUT of penetration
    max_allowed_push_step = (gamma * h_force) / stiffness

    v_push = np.dot(v_nom, n_push)

    # 4. Correction Calculation
    violation = v_push - max_allowed_push_step
    if violation > 0.0:
        clamped_correction = min(violation, max_correction_step)
        v_safe = v_nom - clamped_correction * n_push
    else:
        v_safe = v_nom.copy()

    # 5. Enforce Workspace Limits
    if ee_pos is not None and workspace_min is not None and workspace_max is not None:
        target_pos = ee_pos + v_safe
        clamped_pos = np.clip(target_pos, workspace_min, workspace_max)
        v_safe = clamped_pos - ee_pos

    return v_safe[0], v_safe[1], v_safe[2]