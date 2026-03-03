import math

import numpy as np
import pybullet as p
import os
import time
def world_from_local(body, local_point, link=-1):
    pos, orn = p.getBasePositionAndOrientation(body) if link==-1 else p.getLinkState(body, link)[:2]
    world, _ = p.multiplyTransforms(pos, orn, local_point, [0,0,0,1])
    return np.array(world)

def local_to_local(body_from, body_to, local_point, link_from=-1, link_to=-1):
    # 1. Get current world poses
    posA, ornA = p.getBasePositionAndOrientation(body_from)
    posB, ornB = p.getBasePositionAndOrientation(body_to)

    # 2. Invert Body B's transform
    invPosB, invOrnB = p.invertTransform(posB, ornB)

    # 3. Calculate the transform from B to A
    # This represents Body A's pose as seen from Body B's perspective
    posA_in_B, ornA_in_B = p.multiplyTransforms(invPosB, invOrnB, posA, ornA)

    # 4. Transform the mesh vertices
    mesh_data = p.getMeshData(body_from)
    local_vertices_in_B = []

    for v in mesh_data[1]: # mesh_data[1] is the list of vertices
        # Transform each vertex by the relative pose
        v_relative, _ = p.multiplyTransforms(posA_in_B, ornA_in_B, v, [0, 0, 0, 1])
        local_vertices_in_B.append(v_relative)
    return np.array(local_vertices_in_B)
def radius_spring(foot,leg,a,b):
    radius = 0.01      # distance from center (creates bending resistance)
    num_springs = 1
    angles = np.linspace(0, 2*np.pi, num_springs, endpoint=False)
    local_offsets = []

    for angle in angles:
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        local_offsets.append(np.array([x, y, 0]))
        
    ornA = p.getLinkState(leg, 0)[1] if 0 != -1 else p.getBasePositionAndOrientation(leg)[1]
    ornB = p.getLinkState(foot, 0)[1] if 0 != -1 else p.getBasePositionAndOrientation(foot)[1]

    RA = np.array(p.getMatrixFromQuaternion(ornA)).reshape(3,3)
    RB = np.array(p.getMatrixFromQuaternion(ornB)).reshape(3,3)

    for localA in local_offsets:

        localB = localA.copy()
        worldA = a + RA @ localA
        worldB = b + RB @ localB

    return worldA, worldB
def make_ligament(self,name,foot,leg,a,b, orientation,scale):
    a = a
    b=b
    pC = a#world_from_local(foot, a, 0)
    pD = b#world_from_local(leg, b, 0)
    #print(pC,pD)
    #p.addUserDebugText( f"pC",pC, [1,0,0],1.0)
    #p.addUserDebugText( f"pD",pD, [0,1,0],1.0)
    worldA,worldB = radius_spring(foot, leg, a, b)
    #print(np.linalg.norm(worldA-worldB))
    orientation = orientation
    scale = scale
    name = name
    mid = 0.5 * (worldA + worldB)
    currentDir = os.path.dirname(os.path.abspath(__file__))
    #lig_path = os.path.join(currentDir, "Assets/ligacc.obj")
    
    E = 1e6
    nu = 0.45
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    name = p.loadSoftBody(#"/home/catherine/FractureSoftGym/fracturesurgeryenv/gym_fracture/envs/Assets/ligacc.obj",
       "/home/catherine/Policies/Test/rect3.vtk",
        mass=0.01,
        basePosition=mid-[-0.01,0.015,0],
        baseOrientation=p.getQuaternionFromEuler([90/180*np.pi, 0, 90/180*np.pi]),
        scale=1,
        useNeoHookean=1,
        useMassSpring=0,
        NeoHookeanMu=mu,
        NeoHookeanLambda=lam,
        useBendingSprings=1,
        frictionCoeff=0.5,
        NeoHookeanDamping=0.01,
        repulsionStiffness=0.5 * mu,
        collisionMargin=0.005
    )

    
    colour = [250/255,11/255,58/255,1]
    #print(colour)
    p.changeVisualShape(name, -1, rgbaColor=colour)
    p.changeDynamics(name, -1, mass=0.01, linearDamping=0.05)
    p.setPhysicsEngineParameter(contactERP=0.5)#, 
    p.setPhysicsEngineParameter(numSolverIterations=100, 
                                numSubSteps=50,useSplitImpulse=1,
                                splitImpulsePenetrationThreshold=0.0001) ##This is really important for stability and force control
    p.setPhysicsEngineParameter(contactSlop=0) # Removes the 'allowance' for overlap
    p.setCollisionFilterGroupMask(name, -1, collisionFilterGroup=0, collisionFilterMask=0) # Disable collisions for soft body to prevent explosion during tuning
    #p.setCollisionFilterGroupMask(leg, -1, collisionFilterGroup=0, collisionFilterMask=0)
    p.setCollisionFilterPair(name, foot, -1, -1, enableCollision=0)
    p.setCollisionFilterPair(name, leg, -1, -1, enableCollision=0)
    
    #time.sleep(50)
    
    auto_anchor_ligament(name, bodyA=foot, bodyB=leg, worldA=worldA, worldB=worldB, axis=0, num_anchors=8)
    p.stepSimulation()
    return worldA, worldB
import numpy as np

def make_ligament_rod(foot, leg, a, b, rod_radius=0.01, rod_mass=0.05, stiffness=1e5):
    # Compute endpoints
    pC = world_from_local(foot, a, 0)
    pD = world_from_local(leg, b, 0)
    mid = 0.5 * (pC + pD)
    
    # Rod vector and length
    vec = pD - pC
    L = np.linalg.norm(vec)
    if L < 1e-6:
        L = 0.01  # prevent zero-length
    axis = vec / L
    
    # Compute orientation quaternion for capsule
    z_axis = np.array([0,0,1])
    v = np.cross(z_axis, axis)
    c = np.dot(z_axis, axis)
    if np.linalg.norm(v) < 1e-6:
        orn = [0,0,0,1]  # aligned
    else:
        s = np.sqrt((1+c)*2)
        q = np.array([v[0]/s, v[1]/s, v[2]/s, 0.5*s])
        orn = q.tolist()
    
    # Create capsule rigid body
    rod_collision = p.createCollisionShape(p.GEOM_CAPSULE, radius=rod_radius, height=L)
    rod_visual = p.createVisualShape(p.GEOM_CAPSULE, radius=rod_radius, length=L, rgbaColor=[0,1,0,1])
    rod = p.createMultiBody(baseMass=rod_mass,
                            baseCollisionShapeIndex=rod_collision,
                            baseVisualShapeIndex=rod_visual,
                            basePosition=mid.tolist(),
                            baseOrientation=orn)
    
    # --------------------------
    # Create linear spring along rod axis
    # --------------------------
    # cid = p.createConstraint(
    #     parentBodyUniqueId=foot,
    #     parentLinkIndex=-1,
    #     childBodyUniqueId=rod,
    #     childLinkIndex=-1,
    #     jointType=p.JOINT_PRISMATIC,
    #     jointAxis=axis.tolist(),
    #     parentFramePosition=[0,0,0],
    #     childFramePosition=(-vec/2).tolist()  # align capsule
    # )
    
    # # Set spring stiffness (Hooke's law)
    # p.changeConstraint(cid, maxForce=stiffness)
    
    # # Anchor other end to leg with another spring (optional)
    # cid2 = p.createConstraint(
    #     parentBodyUniqueId=rod,
    #     parentLinkIndex=-1,
    #     childBodyUniqueId=leg,
    #     childLinkIndex=-1,
    #     jointType=p.JOINT_PRISMATIC,
    #     jointAxis=axis.tolist(),
    #     parentFramePosition=(vec/2).tolist(),
    #     childFramePosition=[0,0,0]
    # )
    # p.changeConstraint(cid2, maxForce=stiffness)
    cidA = p.createConstraint(
        parentBodyUniqueId=foot,
        parentLinkIndex=-1,
        childBodyUniqueId=rod,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=axis.tolist(),
        parentFramePosition=a,
        childFramePosition=(-vec/2).tolist()  # capsule local offset
    )
    
    # Anchor rod to leg (end)
    cidB = p.createConstraint(
        parentBodyUniqueId=leg,
        parentLinkIndex=-1,
        childBodyUniqueId=rod,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=axis.tolist(),
        parentFramePosition=b,
        childFramePosition=(vec/2).tolist()
    )
    
    #return rod, cid, cid2


def findClosestVertex(contactPos, vertices):
    vertices_np = np.array(vertices)
    contact_np = np.array(contactPos)
    distances = np.linalg.norm(vertices_np - contact_np, axis=1)
    return np.argmin(distances)



def auto_anchor_ligament(clothId, bodyA, bodyB, worldA, worldB, axis=0, num_anchors=2):
    """
    Automatically anchors a ligament-like soft body to two rigid bodies.
    
    Args:
        clothId   : soft body ID from p.loadSoftBody
        bodyA     : rigid body ID for one end
        bodyB     : rigid body ID for the other end
        axis      : principal axis of ligament (0=x, 1=y, 2=z)
        num_anchors : how many vertices to anchor per side (default: 2)
    """

    # get current simulation mesh
    numVerts, verts = p.getMeshData(clothId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    verts = np.array(verts)
    #print(verts)
    # project vertices onto chosen axis
    #axis_vals = verts[:, axis]
    #data = p.getMeshData(clothId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    # text_uid = []
    # for i in range(data[0]):
    #   pos = data[1][i]
    #   #uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
      #text_uid.append(uid)
    # find min/max along that axis = ends of ligament
    #min_val, max_val = np.min(axis_vals), np.max(axis_vals)

    # # indices sorted by distance from each end
    #endA_ids = np.argsort(np.abs(axis_vals - min_val))[:num_anchors]
    #endB_ids = np.argsort(np.abs(axis_vals - max_val))[:num_anchors]
    # Find closest vertices to bodyA and bodyB
    #distA = np.linalg.norm(verts - worldA, axis=1)
    #distB = np.linalg.norm(verts - worldB, axis=1)
    #print(f"distA:{distA}, distB:{distB}")
    #anchorA_vertices = np.where(distA < 0.05)[0]
    #anchorB_vertices = np.where(distB < 0.05)[0]
    #print(f'verts:{verts}, worldA:{worldA}, worldB:{worldB}, distA:{distA}, distB:{distB}, anchorA_vertices:{anchorA_vertices}, anchorB_vertices:{anchorB_vertices}')
    #print(p.getContactPoints(clothId, bodyA))
    # create anchors at those vertices
    #anchorA_vertices = anchorA_vertices[:num_anchors]
    #anchorB_vertices = anchorB_vertices[:num_anchors]
    #print(f"Anchoring vertices {endA_ids} to bodyA (foot) and {endB_ids} to bodyB (leg)")
    #p.addUserDebugText('WorldA', worldA, [1,1,0], 2.0)
    #p.addUserDebugText('WorldB', worldB, [1,0,1], 2.0)
    anchorB_vertices = [0,3,4,7]
    anchorA_vertices = [1,2,5,6]
    b_verts =local_to_local(clothId, bodyB, -1)
    a_verts =local_to_local(clothId, bodyA, -1)
    a_coords = a_verts[anchorA_vertices]
    b_coords = b_verts[anchorB_vertices]
    #print(a_coords, b_coords)
    
    #print(f"Distance between anchors: {diff:.4f} m")
    # print(f"Anchor A vertices (local to bodyB): {a_coords}")
    # print(f"Anchor B vertices (local to bodyB): {b_coords}")
    # print(f"Anchor A vertices (local to bodyA): {a_coords[1]}")#
    #diff = np.linalg.norm(verts[3]-verts[5])
    #print(f"Distance between anchors: {diff:.4f} m")
    for i, vid in enumerate(anchorA_vertices):
        #print(f'vid:{int(vid)}')
        #print(a_coords[i])
        p.createSoftBodyAnchor(clothId, int(vid), bodyA, -1,a_coords[i].tolist())
        #p.addUserDebugText( f"anchorA_{vid}",vid,[1,0,0], 5.0)
        #p.addUserDebugText(f"A{vid}", verts[int(vid)], [0,0,1], 2)
    for i, vid in enumerate(anchorB_vertices):
        p.createSoftBodyAnchor(clothId, int(vid), bodyB, -1,b_coords[i].tolist())
        #p.addUserDebugText( f"anchorB_{vid}",vid,[0,1,0], 1.0)
        #p.addUserDebugText(f"B{vid}", verts[int(vid)], [0,1,0], 2)
    
    #print(f"Anchored {len(anchorA_vertices)} vertices to bodyA and {len(anchorB_vertices)} to bodyB")