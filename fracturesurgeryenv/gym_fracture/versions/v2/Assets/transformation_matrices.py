import numpy as np 
import pybullet as p
from scipy.spatial.transform import Rotation as R

def get_proximal_to_distal_matrix(patient_id):
    """
    Returns the transformation matrix defining the distal goal 
    pose directly relative to the proximal bone frame.
    """
    prox_to_dist_matrices = {
        198: np.array([[ 1.        ,  0.        ,  0.        ,  0.03472044],
       [ 0.        ,  0.98746312,  0.15784962,  0.05242262],
       [ 0.        , -0.15784962,  0.98746312,  0.00572582],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),

    #     array([[1.        , 0.        , 0.        , 0.03472045],
    #    [0.        , 1.        , 0.        , 0.04793356],
    #    [0.        , 0.        , 1.        , 0.01723593],
    #    [0.        , 0.        , 0.        , 1.        ]]),
#         array([[ 1.        ,  0.        ,  0.        ,  0.03472044],
#        [ 0.        ,  0.98746312,  0.15784962,  0.05242262],
#        [ 0.        , -0.15784962,  0.98746312,  0.00572582],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])
# ,
        
        102: np.array([[ 0.99945903, -0.01787941, -0.02760598, 0.00794629],
       [ 0.01733119,  0.9996503 , -0.01997166,  -0.00371645 ],
       [ 0.02795341,  0.01948241,  0.99941933, -0.06337698],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
, 
    #    np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     -3.66778672e-03],
    #    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
    #      7.67704993e-02],
    #    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
    #     -2.50449404e-04],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]]),

        # np.array([[ 0.999459, -0.017879, -0.027606, -0.003794],
        #                [ 0.017331,  0.999650, -0.019972,  0.00030 ],
        #                [ 0.027953,  0.019482,  0.999419, 0.002779 ],
        #                [ 0.000000,  0.000000,  0.000000,  1.000000]]),

        # 132: np.array([[ 0.946846, -0.196342,  0.254819, -0.002213],
        #                [ 0.187996,  0.980517,  0.056956,  0.003661],
        #                [-0.261037, -0.006023,  0.965310, -0.008602],
        #                [ 0.000000,  0.000000,  0.000000,  1.000000]]),
        132: np.array([[ 0.94684589, -0.19634171,  0.25481883,  0.0209488 ],
       [ 0.18799581,  0.98051709,  0.05695545, -0.00868836],
       [-0.26103693, -0.00602316,  0.96531004,-0.01186311 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
 
        # np.array([[ 0.94684589, -0.19634171,  0.25481883,  0.0209488 ],
        #                 [ 0.18799581,  0.98051709,  0.05695545, -0.01186311],
        #                 [-0.26103693, -0.00602316,  0.96531004, -0.00868836],
        #                 [ 0.        ,  0.        ,  0.        ,  1.        ]]),

       252: np.array([[1.        , 0.        , 0.        , -0.00526727],
                        [0.        , 1.        , 0.        , -0.0336327],
                        [0.        , 0.        , 1.        , -0.05512874],
                        [0.        , 0.        , 0.        , 1.        ]]),

    #    array([[ 0.60778052,  0.13831837,  0.78196603,  0.00261723],
    #                     [ 0.03417828,  0.97924471, -0.19977906, -0.01316311],
    #                     [-0.79336917,  0.14814804,  0.59043843, -0.06440921],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # #    array([[ 0.60778058,  0.03417824, -0.79336911, -0.01803264],
    #                     [ 0.13831836,  0.97924465,  0.14814806, -0.01473225 ],
    #                     [ 0.78196603, -0.19977902,  0.59043831,0.06160719 ],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    126: np.array([[ 9.99999881e-01,  1.51339918e-09, -9.50240064e-09,
                        -8.72234305e-05],
                    [ 0.00000000e+00,  9.99999940e-01, -1.16415322e-09,
                        -8.78721569e-03],
                    [-2.07946869e-08, -4.65661287e-10,  1.00000000e+00,
                        1.58927888e-02],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        1.00000000e+00]])


    }

    offsets = {
        198: np.array([0.0, 0.0, 0.0]),
        102: np.array([0.0, -0.0, 0.0]),
        132: np.array([0.0, 0.0, 0.0])
    }
    
    return prox_to_dist_matrices.get(patient_id, np.eye(4)), offsets.get(patient_id, np.array([0.0, 0.0, 0.0]))


def matrix_to_pos_quat(matrix):
    """Converts a 4x4 homogenous matrix to pos [x,y,z] and quat [x,y,z,w]"""
    pos = matrix[:3, 3].tolist()
    rot_matrix = matrix[:3, :3]
    quat = R.from_matrix(rot_matrix).as_quat().tolist()  # [x, y, z, w]
    return pos, quat


def compute_robot_goal_relative(env, distal_start_pos, distal_start_ori, bone_goal_pos, bone_goal_ori):
    """
    Computes Link 11 goal pose accounting for the initial rigid grip attachment.
    """
    # 1. Get current End-Effector (Link 11) Pose in World space
    hand_state = p.getLinkState(env.pandaUid, 11)
    hand_pos, hand_ori = hand_state[0], hand_state[1]
    distal_state = p.getLinkState(env.foot, 1) 
    distal_start_pos = distal_state[0]
    distal_start_ori = distal_state[1]
    # 2. Compute Grip Offset: Hand pose RELATIVE to Distal Bone's starting pose
    inv_distal_pos, inv_distal_ori = p.invertTransform(distal_start_pos, distal_start_ori)
    grip_pos, grip_ori = p.multiplyTransforms(inv_distal_pos, inv_distal_ori, hand_pos, hand_ori)

    # 3. Apply Grip Offset to target Distal Bone Pose
    robot_goal_pos, robot_goal_ori = p.multiplyTransforms(
        bone_goal_pos, bone_goal_ori, 
        grip_pos, grip_ori
    )
    #print(grip_pos, grip_ori)
    # Normalize output quaternion
    robot_goal_ori = (np.array(robot_goal_ori) / np.linalg.norm(robot_goal_ori)).tolist()

    return robot_goal_pos, robot_goal_ori


def get_goal_from_proximal_pose(env, patient_id, leg_start_pos, leg_start_ori, distal_start_pos, distal_start_ori):
    """
    Calculates Distal Bone Goal directly from Proximal Leg Start Pose 
    using the Proximal-to-Distal transformation matrix.
    """
    # 1. Fetch Proximal-to-Distal matrix
    M_prox_to_dist, offsets = get_proximal_to_distal_matrix(patient_id)

    # 2. Extract position and quaternion [x, y, z, w]
    rel_pos, rel_quat = matrix_to_pos_quat(M_prox_to_dist)

    # 3. Apply Proximal-to-Distal transform directly onto leg_start_pos
    distal_goal_pos, distal_goal_ori = p.multiplyTransforms(
        leg_start_pos, 
        leg_start_ori,
        rel_pos, 
        rel_quat
    )
    
    # Apply scene offset if needed
    distal_goal_pos = (np.array(distal_goal_pos) - offsets).tolist()

    # 4. Compute Robot Hand Goal Pose (Link 11)
    robot_goal_pos, robot_goal_ori = compute_robot_goal_relative(
        env, 
        distal_start_pos, 
        distal_start_ori, 
        distal_goal_pos, 
        distal_goal_ori
    )

    # Store goals
    env.goal_pos = np.array(robot_goal_pos) #- [0.00, -0.01, -0.01]  # Slightly lower to avoid collision
    env.goal_ori = np.array(robot_goal_ori)
    env.target_position = np.concatenate((env.goal_pos, env.goal_ori))
    print(leg_start_pos, leg_start_ori)
    print('Bone Goal Position:', distal_goal_pos)
    print('Bone Goal Orientation:', np.rad2deg(np.array(p.getEulerFromQuaternion(distal_goal_ori))))
    print('Bone Goal Orientation:', distal_goal_ori)
    print("Robot Goal Position:", env.goal_pos)
    print("Robot Goal Orientation:", np.rad2deg(np.array(p.getEulerFromQuaternion(env.goal_ori))))
    print("Robot Goal Orientation:", env.goal_ori)
    print('Current Robot Hand Position:', p.getLinkState(env.pandaUid, 11)[0])
    print('Current Robot Hand Orientation:', np.rad2deg(np.array(p.getEulerFromQuaternion(p.getLinkState(env.pandaUid, 11)[1]))))
    return env.target_position, distal_goal_pos, distal_goal_ori


def get_patient_goal(env, patient_id):
    """
    Returns the distal bone goal position and orientation based on the patient ID.
    """
    #252 leg and foot ori 90 deg on x
    foot = {
        252: np.array([p.getQuaternionFromEuler([0,0, 0])]),
        102: np.array([p.getQuaternionFromEuler([0,0, 0])]),
        198: np.array([p.getQuaternionFromEuler([0,0, 0])]),
        132: np.array([p.getQuaternionFromEuler([0,0, 0])]),
        126: np.array(p.getQuaternionFromEuler([90/180*np.pi,0, 0]))
    }
    leg = {
        252: (np.array([0.34851957,-0.135,0.07026956]),
              np.array([0.7044160264027587, -0.06162841671621936, 0.06162841671621935, 0.7044160264027588])), #[p.getQuaternionFromEuler([90/180*np.pi,0, 0])]
        102: (np.array([0.3470195700516103, -0.15000000000594865, 0.07526955827664446]-np.array([-0.00,0.005,0.01])),
              np.array([p.getQuaternionFromEuler([90/180*np.pi,0, 0])])),
        198: (np.array([0.3470195700516103, -0.13000000000594865, 0.07526955827664446]),
              np.array([p.getQuaternionFromEuler([0/180*np.pi,0/180*np.pi, 0])])),
        132: (np.array([0.3050195700516103, -0.06000000000594865, 0.07526955827664446])- np.array([-0.005,-0.005,-0.005]),
              np.array([p.getQuaternionFromEuler([0,0, 0])])),
        126: (np.array([ 0.34701957,-0.06,0.04526956])-np.array([-0.00,-0.005,0.01]),
              np.array([0.7044160264027587, -0.06162841671621936, 0.06162841671621935, 0.7044160264027588])),
        
        

    }
    goals = {
        252: (np.array([ 0.32180062,-0.09246775, 0.15800003]) -np.array([0.005,-0.019,0.00]), #np.array([0.01,-0.023,0.008]),
              np.array([0.9999999728200057, 0.00023313980271510995, -8.89660707914592e-08, 2.4108688676344187e-06])),
        102: (np.array([ 0.30383321, -0.0970693,   0.15603161])-np.array([0.005,-0.0265,0.003]),
              np.array([0.9999999728200057, 0.00023313980271510995, -8.89660707914592e-08, 2.4108688676344187e-06])),
        198: (np.array([0.34174003, -0.08506263,  0.16001045]) - np.array([0.0,-0.02,-0.005]),#np.array([0.34174003, -0.08506263,  0.16001045]) - np.array([0.01,0.01,0.005])#np.array([ 0.32180062,-0.09246775, 0.15800003]) - np.array([0.005,-0.005,0.00])#np.array([ 0.32180062,-0.09246775, 0.15800003]) - np.array([0.005,-0.005,0.00])
              p.getQuaternionFromEuler(p.getEulerFromQuaternion(p.getLinkState(env.pandaUid, 11)[1])-np.array([9.08/180*np.pi,0, 0]))),
        132: (np.array([0.3121838, -0.08800575, 0.15520251]) - np.array([0.01,-0.03,-0.01]),
              p.getQuaternionFromEuler(p.getEulerFromQuaternion(p.getLinkState(env.pandaUid, 11)[1])-np.array([1/180*np.pi,-11/180*np.pi, -15/180*np.pi]))),
        126: (np.array([ 0.29517541, -0.07789279,  0.13343939])-np.array([-0.010,-0.01,-0.006]),
              p.getEulerFromQuaternion(p.getLinkState(env.pandaUid, 11)[1])-np.array([0,10/180*np.pi, 0])),
    }
    return goals.get(patient_id, (np.zeros(3), np.array([0, 0, 0, 1]))), leg.get(patient_id, (np.zeros(3), [0, 0, 0, 1]))