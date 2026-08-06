import numpy as np 
import pybullet as p
from scipy.spatial.transform import Rotation as R

def get_patient_transformation_matrix(patient_id):
    """
    Returns the transformation matrix for a given patient ID.
    """
    P = P = np.array([
    [1,  0,  0, 0],
    [0,  0,  1, 0],   # New Y = Old Z
    [0, -1,  0, 0],   # New Z = -Old Y (keeps det = +1)
    [0,  0,  0, 1]
])
   
    M = np.array([[ 1.00000000e+00,  0.00000000e+00, -2.49151960e-22,
        -3.02579231e-03],
       [ 0.00000000e+00,  1.00000000e+00, -2.11758237e-22,
         6.34170547e-02],
       [-2.49151986e-22,  9.80987032e-24,  1.00000000e+00,
        -2.13182444e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
    M_rot = P @ M @ P.T
    print("M_rot:", M_rot)
    goal_transformation_matrices = {
        198: np.array([[0.999459, -0.017879, -0.027606, -0.003794],
                       [0.017331, 0.999650, -0.019972, 0.000030],
                       [0.027953, 0.019482, 0.999419, 0.002779],
                       [0.000000, 0.000000, 0.000000, 1.000000]]),
        102: np.array([[0.999459, 0.017331, 0.027953, 0.003714],
                       [-0.017879, 0.999650, 0.019482, -0.000152],
                       [-0.027606, -0.019972, 0.999419, -0.002881],
                       [0.000000, 0.000000, 0.000000, 1.000000]]),
        # 102: np.array([[ 0.99945891,  0.01733119,  0.02795341,  0.0067397 ],
        #             [-0.0178794 ,  0.99965024,  0.01948241, -0.00266791 ],
        #             [-0.02760598, -0.01997166,  0.99941933,-0.0635693 ],
        #             [ 0.        ,  0.        ,  0.        ,  1.        ]])

        132: np.array([[0.946846, -0.196342, 0.254819, -0.002213],
                       [0.187996, 0.980517, 0.056956, 0.003661],
                       [-0.261037, -0.006023, 0.965310, -0.008602],
                       [0.000000, 0.000000, 0.000000, 1.000000]]),
        

    }

    proximal_transformation_matrices = {
        198: np.array([[ 1.        ,  0.        ,  0.        , -0.03472044],
                       [ 0.        ,  1.        ,  0.        , -0.04793354],
                       [ 0.        ,  0.        ,  1.        , -0.01723594],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
        102: np.array([[ 1.00000000e+00,  0.00000000e+00, -2.49151960e-22,
        -3.02579231e-03],
       [ 0.00000000e+00,  1.00000000e+00, -2.11758237e-22,
         2.13182444e-04],
       [-2.49151986e-22,  9.80987032e-24,  1.00000000e+00,
        6.34170547e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]),

    #     132: np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.90349612e-02],
    #    [ 0.00000000e+00,  1.00000000e+00, -1.19209290e-07,
    #     -1.97690930e-02],
    #    [ 0.00000000e+00,  1.19209290e-07,  1.00000000e+00,
    #      4.93443012e-03],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]]),
        # 132: np.array([[0.946846, -0.196342, 0.254819, -0.002213],
        #                [0.187996, 0.980517, 0.056956, 0.003661],
        #                [-0.261037, -0.006023, 0.965310, -0.008602],
        #                [0.000000, 0.000000, 0.000000, 1.000000]]),
        132: np.array([[ 0.94684607,  0.18799581, -0.26103699, -0.01987306],
                        [-0.19634172,  0.98051715, -0.00602316,0.01569277  ],
                        [ 0.25481883,  0.05695545,  0.9653101 ,0.00372448  ],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
#array([[ 0.94684589, -0.19634171,  0.25481883,  0.0209488 ],
                       # [ 0.18799581,  0.98051709,  0.05695545, -0.01186311],
                        #[-0.26103693, -0.00602316,  0.96531004, -0.00868836],
                        #[ 0.        ,  0.        ,  0.        ,  1.        ]]),

    }
    offsets = {
        198: np.array([0,0,0]),
        102: np.array([-0.01, 0.015, -0.00]),
        132: np.array([0.0, 0.015, -0.00])
    }
    
    return goal_transformation_matrices.get(patient_id, np.eye(4)), proximal_transformation_matrices.get(patient_id, np.eye(4)), offsets.get(patient_id, np.array([0,0,0]))


def compute_robot_goal_from_fracture_goal(env, bone_goal_pos, bone_goal_ori, foot_pos, foot_ori):
    """
    Computes where the robot hand (Link 11) needs to go so that 
    the bone it is holding ends up at (bone_goal_pos, bone_goal_ori).
    """
    # 1. Get current world pose of Robot Hand (Link 11)
    hand_pos, hand_ori = p.getLinkState(env.pandaUid, 11)[0], p.getLinkState(env.pandaUid, 11)[1]

    # 2. Calculate the GRIP offset (Robot Hand relative to Bone's initial pose)
    inv_bone_pos, inv_bone_ori = p.invertTransform(foot_pos, foot_ori)
    grip_pos, grip_ori = p.multiplyTransforms(inv_bone_pos, inv_bone_ori, hand_pos, hand_ori)

    # 3. Calculate Final Robot Goal by applying the Grip offset to the Bone Goal Target
    robot_goal_pos, robot_goal_ori = p.multiplyTransforms(
        bone_goal_pos, bone_goal_ori, 
        grip_pos, grip_ori
    )

    # Normalize output quaternion for solver stability
    norm = np.linalg.norm(robot_goal_ori)
    robot_goal_ori = (np.array(robot_goal_ori) / norm).tolist()

    return robot_goal_pos, robot_goal_ori


def get_goal_and_proximal_transforms(env, patient_id, foot, foot_ori):
    """
    Returns the target position for the ROBOT HAND (Link 11) and the starting pose of the proximal leg.
    """
    goal_transform_matrix, proximal_transform_matrix, offsets = get_patient_transformation_matrix(patient_id)
    rot_x_90_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
    # -------------------------------------------------------------
    # 1. Calculate Leg Start Pose (Proximal relative to Foot)
    # -------------------------------------------------------------
    prox_pos = proximal_transform_matrix[0:3, 3].tolist()
    prox_rot_matrix = proximal_transform_matrix[0:3, 0:3]
    prox_quat = R.from_matrix(prox_rot_matrix).as_quat().tolist()
    
    leg_start_pos, leg_start_ori = p.multiplyTransforms(foot, foot_ori, prox_pos, prox_quat)
    #leg_start_pos,leg_start_ori = p.multiplyTransforms(leg_start_pos1, leg_start_ori1, [0, 0, 0], rot_x_90_quat)
    leg_start = np.array(leg_start_pos) - np.array(offsets)  # Slightly lower to avoid collision
    leg_orientation = np.array(leg_start_ori)

    # -------------------------------------------------------------
    # 2. Calculate Bone World Goal Pose (Distal Goal relative to Foot)
    # -------------------------------------------------------------
    bone_goal_offset = goal_transform_matrix[0:3, 3].tolist()
    bone_rot_matrix = goal_transform_matrix[0:3, 0:3]
    bone_goal_quat = R.from_matrix(bone_rot_matrix).as_quat().tolist()
    
    # Bone's target location in the PyBullet world frame
    bone_goal_pos, bone_goal_ori = p.multiplyTransforms(foot, foot_ori, bone_goal_offset, bone_goal_quat)
    bone_goal_pos = np.array(bone_goal_pos) - np.array([0,-0.015,0])  # Slightly lower to avoid collision
    # -------------------------------------------------------------
    # 3. Compute Robot Hand Goal (Accounts for Grip Offset)
    # -------------------------------------------------------------
    robot_goal_pos, robot_goal_ori = compute_robot_goal_from_fracture_goal(
        env, 
        bone_goal_pos, 
        bone_goal_ori, 
        foot, 
        foot_ori
    )

    # Store the ROBOT's goal pose in the environment for IK / Motion Planning
    env.goal_pos = np.array(robot_goal_pos) #- np.array([0.01, -0.015, -0.00])
    env.goal_ori = np.array(robot_goal_ori)
    print('Current Robot Hand Position:', p.getLinkState(env.pandaUid, 11)[0])
    print('Current Robot Hand Orientation:', p.getLinkState(env.pandaUid, 11)[1])
    print("Bone Goal Position:", bone_goal_pos)
    print("Robot Goal Position:", env.goal_pos)
    print("Robot Goal Orientation:", env.goal_ori)
    env.target_position = np.concatenate((env.goal_pos, env.goal_ori))
    return env.target_position, leg_start, leg_orientation