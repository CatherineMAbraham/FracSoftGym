import numpy as np
import pybullet as p
import os
import time
def world_from_local(body, local_point, link=-1):
    pos, orn = p.getBasePositionAndOrientation(body) if link==-1 else p.getLinkState(body, link)[:2]
    world, _ = p.multiplyTransforms(pos, orn, local_point, [0,0,0,1])
    return np.array(world)

def make_ligament(self,name,foot,leg,a,b, orientation,scale):
    a = a
    b=b
    pC = a#world_from_local(foot, a, 0)
    pD = b#world_from_local(leg, b, 0)
    #print(pC,pD)
    p.addUserDebugText( f"pC",pC, [1,0,0],1.0)
    p.addUserDebugText( f"pD",pD, [0,1,0],1.0)
    orientation = orientation
    scale = scale
    name = name
    mid = 0.5 * (pC + pD)
    currentDir = os.path.dirname(os.path.abspath(__file__))
    lig_path = os.path.join(currentDir, "Assets/ligacc.obj")
    # name = p.loadSoftBody(lig_path,
    #     basePosition=mid,
    #     baseOrientation=orientation,
    #     scale=scale,
    #     mass=0.1,
    #     useNeoHookean=0,
    #     useMassSpring=1,
    #     useBendingSprings=0,
    #     springElasticStiffness=40,      # stiffer -> springier/shape-preserving
    #     springDampingStiffness=1,    # moderate damping -> oscillation allowed
    #     #springDampingAllDirections=1,
    #    #springBendingStiffness=2,       # preserve rod shape
    #     useSelfCollision=0,             # disable initially for tuning
    #     collisionMargin=0.01,
    #     frictionCoeff=0.6,
    #     useFaceContact=0
    # )
    E = 1e6
    nu = 0.45
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    name = p.loadSoftBody(#"/home/catherine/FractureSoftGym/fracturesurgeryenv/gym_fracture/envs/Assets/ligacc.obj",
       "/home/catherine/Policies/Test/rect3.vtk",
        mass=0.1,
        basePosition=mid-[0,0.01,0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 90/180*np.pi]),
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

    # name = p.loadSoftBody(
    # lig_path,
    # basePosition=mid,
    # baseOrientation=orientation,
    # scale=scale,
    # mass=0.02,
    # useNeoHookean=1,
    # NeoHookeanMu=961500,
    # NeoHookeanLambda= 1442300,
    # NeoHookeanDamping=0.01,
    # useMassSpring=0,
    # useBendingSprings=0,
    # useFaceContact=0,
    # collisionMargin=0.001,
    # frictionCoeff=0.5,
    # useSelfCollision=1
    # )
    colour = [250/255,11/255,58/255,1]
    #print(colour)
    p.changeVisualShape(name, -1, rgbaColor=colour)
### These parameters work
#  clothId = p.loadSoftBody("/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/ligament2.obj",
#         basePosition=mid,
#         baseOrientation=p.getQuaternionFromEuler([0, 0, 90/180*np.pi]),
#         scale=1,
#         mass=0.1,
#         useNeoHookean=0,
#         useMassSpring=1,
#         useBendingSprings=1,
#         springElasticStiffness=50,      # stiffer -> springier/shape-preserving
#         springDampingStiffness=0.5,    # moderate damping -> oscillation allowed
#         #springDampingAllDirections=1,
#         springBendingStiffness=1,       # preserve rod shape
#         #useSelfCollision=0,             # disable initially for tuning
#         collisionMargin=0.005,
#         frictionCoeff=0.6,
#         useFaceContact=0
#     )

####
    # clothId = p.loadSoftBody("/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/ligamentrectangle1.obj",
    #     basePosition=mid,
    #     baseOrientation=p.getQuaternionFromEuler([0, 0, 90/180*np.pi]),
    #     scale=1,
    #     mass=0.05,
    #     useNeoHookean=1,
    #     NeoHookeanMu=20,
    #     NeoHookeanLambda=50,
    #     NeoHookeanDamping=0.02,
    #     collisionMargin=0.005,
    #     useSelfCollision=1,
    #     frictionCoeff=0.5,
    #     useFaceContact=1)
    # print(p.getAABB(clothId))
    #p.setTimeStep(1.0/100.0)
    #p.setPhysicsEngineParameter(numSolverIterations=200)#, 
    #p.setPhysicsEngineParameter(erp=0.15)#, 
    #p.setPhysicsEngineParameter(contactERP=0.1)#, 
    #p.setPhysicsEngineParameter(numSubSteps=3) ##This is really important for stability and force control 
    #p.setPhysicsEngineParameter(fixedTimeStep=1/120.0)
    p.setPhysicsEngineParameter(numSolverIterations=1000, numSubSteps=50)
    p.stepSimulation()
    #time.sleep(5)
    auto_anchor_ligament(name, bodyA=foot, bodyB=leg, worldA=pC, worldB=pD, axis=0, num_anchors=5)

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

    # project vertices onto chosen axis
    # axis_vals = verts[:, axis]

    # # find min/max along that axis = ends of ligament
    # min_val, max_val = np.min(axis_vals), np.max(axis_vals)

    # # indices sorted by distance from each end
    # endA_ids = np.argsort(np.abs(axis_vals - min_val))[:num_anchors]
    # endB_ids = np.argsort(np.abs(axis_vals - max_val))[:num_anchors]
    # Find closest vertices to bodyA and bodyB
    distA = np.linalg.norm(verts - worldA, axis=1)
    distB = np.linalg.norm(verts - worldB, axis=1)
    anchorA_vertices = np.where(distA < 0.05)[0]
    anchorB_vertices = np.where(distB < 0.05)[0]
    #print(f'verts:{verts}, worldA:{worldA}, worldB:{worldB}, distA:{distA}, distB:{distB}, anchorA_vertices:{anchorA_vertices}, anchorB_vertices:{anchorB_vertices}')
    #print(p.getContactPoints(clothId, bodyA))
    # create anchors at those vertices
    anchorA_vertices = anchorA_vertices[:num_anchors]
    anchorB_vertices = anchorB_vertices[:num_anchors]
    
    for vid in anchorA_vertices:
        #print(f'vid:{int(vid)}')
        p.createSoftBodyAnchor(clothId, int(vid), bodyA, -1)
        #p.addUserDebugText( f"anchorA_{vid}",vid,[1,0,0], 5.0)
    for vid in anchorB_vertices:
        p.createSoftBodyAnchor(clothId, int(vid), bodyB, -1)
        #p.addUserDebugText( f"anchorB_{vid}",vid,[0,1,0], 1.0)

    #print(f"Anchored {len(anchorA_vertices)} vertices to bodyA and {len(anchorB_vertices)} to bodyB")