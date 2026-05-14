"""
spring_between_boxes.py

Runnable demo: two boxes connected by a spring-damper between local anchor points.
Requires: pybullet

Run: python spring_between_boxes.py
"""
import time
import numpy as np
import pybullet as p
import pybullet_data
import os 

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=30, cameraPitch=-30,
#                              cameraTargetPosition=[0.1, 0.0, 0.15])

# dt = 1.0 / 240.0
# p.setTimeStep(dt)
# p.setGravity(0, 0, -9.81)
# p.setPhysicsEngineParameter(numSolverIterations=80)  # helpful for stability

# # ground
# p.loadURDF("plane.urdf")

# # Create two boxes (you can replace with your URDFs)
# box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
# box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
# leg_path = ('/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/leg.obj')
# foot_path = ('/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/foot.obj')
# leg_col = p.createCollisionShape(p.GEOM_MESH, fileName=leg_path, meshScale=[1,1,1])
# foot_col = p.createCollisionShape(p.GEOM_MESH, fileName=foot_path, meshScale=[1,1,1])
# leg_vis = p.createVisualShape(p.GEOM_MESH, fileName=leg_path, meshScale=[1,1,1])
# foot_vis = p.createVisualShape(p.GEOM_MESH, fileName=foot_path, meshScale=[1,1,1])

# massA = 1.0
# massB = 1.0
# startA = [0.0, -0.1, 0.2]
# startB = [0.3, 0.0, 0.2]

# #bodyA = p.createMultiBody(baseMass=massA, baseCollisionShapeIndex=box_col,
#  #                         baseVisualShapeIndex=box_vis, basePosition=startA)
# #bodyB = p.createMultiBody(baseMass=massB, baseCollisionShapeIndex=box_col,
#  #                         baseVisualShapeIndex=box_vis, basePosition=startB)
# # bodyA = p.loadURDF(foot_path, basePosition=startA, globalScaling=1)
# # p.changeDynamics(bodyA, -1, mass=0.6, lateralFriction=0.5)
# # bodyB = p.loadURDF(leg_path,
# #                 basePosition =startB,
# #                 #baseOrientation = legorientation,
# #                 globalScaling = 1,
# #                 useFixedBase = 1)
# bodyA = p.createMultiBody(baseMass=massA, baseCollisionShapeIndex=foot_col,
#                            baseVisualShapeIndex=foot_vis, basePosition=startA)
# bodyB = p.createMultiBody(baseMass=massB, baseCollisionShapeIndex=leg_col,
#                            baseVisualShapeIndex=leg_vis, basePosition=startB)
# # slightly different colors
# p.changeVisualShape(bodyA, -1, rgbaColor=[0.2, 0.6, 0.9, 1])
# p.changeVisualShape(bodyB, -1, rgbaColor=[0.9, 0.5, 0.2, 1])

# # anchor points in local frames (relative to base frame of each box)
# # choose points on the surface facing each other
# anchorA_local = [0.06, 0.0, 0.0]   # a little offset on +x of A
# anchorB_local = [-0.06, 0.0, 0.0]  # a little offset on -x of B



# optional: set an explicit rest length, or None to use initial distance
# def world_from_local(body, local_pos, link_index=-1):
#     if link_index == -1:
#         base_pos, base_orn = p.getBasePositionAndOrientation(body)
#         world_pos, _ = p.multiplyTransforms(base_pos, base_orn, local_pos, [0,0,0,1])
#         return np.array(world_pos)
#     else:
#         ls = p.getLinkState(body, link_index, computeForwardKinematics=1)
#         link_world_pos = ls[0]
#         link_world_orn = ls[1]
#         world_pos, _ = p.multiplyTransforms(link_world_pos, link_world_orn, local_pos, [0,0,0,1])
#         return np.array(world_pos)

# def point_velocity(body, world_point, link_index=-1):
#     if link_index == -1:
#         lin_vel, ang_vel = p.getBaseVelocity(body)
#         lin_vel = np.array(lin_vel)
#         ang_vel = np.array(ang_vel)
#         base_pos, _ = p.getBasePositionAndOrientation(body)
#         r = np.array(world_point) - np.array(base_pos)
#     else:
#         ls = p.getLinkState(body, link_index, computeLinkVelocity=1, computeForwardKinematics=1)
#         lin_vel = np.array(ls[6])
#         ang_vel = np.array(ls[7])
#         link_world_pos = np.array(ls[0])
#         r = np.array(world_point) - link_world_pos
#     return lin_vel + np.cross(ang_vel, r)




# def add_spring_damper(bodyA, bodyB):
#     # spring parameters (tune)
#     k = 80.0        # stiffness (N/m)
#     # estimate critical damping for two masses m1,m2 -> mu = m1*m2/(m1+m2)
#     massA = 1.0
#     massB = 1.0
#     mu = (massA * massB) / (massA + massB)
#     c_crit = 2.0 * np.sqrt(k * mu)
#     # pick damping near critical for stable behavior
#     c = 0.9 * c_crit
#     anchorA_local = [0,0,0]#p.getLinkState(bodyA, -1)
#     anchorB_local = [0,0,0]#p.getLinkState(bodyB, -1)
#     print(f"Estimated effective mass mu = {mu:.3f} kg, critical damping c_crit = {c_crit:.3f} N*s/m")
#     print(f"Using k={k:.1f}, c={c:.3f}")
#     # compute rest length based on initial world anchor distance
#     pA_init = world_from_local(bodyA, anchorA_local, -1)
#     pB_init = world_from_local(bodyB, anchorB_local, -1)
#     rest_len = np.linalg.norm(pB_init - pA_init)
#     print(f"Initial rest length = {rest_len:.4f} m")

#     # small helper to draw anchor spheres & spring line
#     anchor_vis_radius = 0.005
#     anchorA_vis = p.createVisualShape(p.GEOM_SPHERE, radius=anchor_vis_radius)
#     anchorB_vis = p.createVisualShape(p.GEOM_SPHERE, radius=anchor_vis_radius)
#     anchorA_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=anchorA_vis,
#                                     basePosition=pA_init.tolist())
#     anchorB_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=anchorB_vis,
#                                     basePosition=pB_init.tolist())
#     p.changeVisualShape(anchorA_marker, -1, rgbaColor=[1,0,0,1])
#     p.changeVisualShape(anchorB_marker, -1, rgbaColor=[1,0,0,1])

#     # simulation loop
#     line_id = None
#     force_text_id = None
#     #for i in range(20000):
#         # compute world anchor positions
#     pA_world = world_from_local(bodyA, anchorA_local, -1)
#     pB_world = world_from_local(bodyB, anchorB_local, -1)

#     # update small anchor markers
#     p.resetBasePositionAndOrientation(anchorA_marker, pA_world.tolist(), [0,0,0,1])
#     p.resetBasePositionAndOrientation(anchorB_marker, pB_world.tolist(), [0,0,0,1])

#     # displacement and direction
#     d = pB_world - pA_world
#     dist = np.linalg.norm(d)
#     if dist < 1e-8:
#         dirn = np.array([0.0, 0.0, 0.0])
#     else:
#         dirn = d / dist

#     # point velocities (include angular)
#     vA = point_velocity(bodyA, pA_world, -1)
#     vB = point_velocity(bodyB, pB_world, -1)
#     v_rel = vB - vA
#     v_along = np.dot(v_rel, dirn) if dist >= 1e-8 else 0.0

#     # spring/damper forces (scalar)
#     fs = -k * (dist - rest_len)
#     fd = -c * v_along
#     f_total = (fs + fd) * dirn  # vector

#     # apply to bodies: minus to A, plus to B (applied at the world anchor locations)
#     # if body is static (mass=0) you may skip applying to it
#     p.applyExternalForce(bodyA, -1, (-f_total).tolist(), pA_world.tolist(), flags=p.WORLD_FRAME)
#     p.applyExternalForce(bodyB, -1, f_total.tolist(), pB_world.tolist(), flags=p.WORLD_FRAME)

#     # draw the spring (debug line)
#     if line_id is not None:
#         p.removeUserDebugItem(line_id)
#     line_id = p.addUserDebugLine(pA_world.tolist(), pB_world.tolist(), lineColorRGB=[0.2,1,0.2],
#                                 lineWidth=2.5, lifeTime=0.05)

#     # display instantaneous force text near midpoint
#     mid = 0.5 * (pA_world + pB_world)
#     if force_text_id is not None:
#         p.removeUserDebugItem(force_text_id)
#     force_text_id = p.addUserDebugText(f"{np.linalg.norm(f_total):.2f} N", mid.tolist(),
#                                     textColorRGB=[1,1,1], textSize=1.2, lifeTime=0.05)

class SpringDamper:
    """ Connect two bodies with a spring-damper between local anchor points.
    Call step() each sim step to apply forces.
    Args:
        bodyA, bodyB: body unique IDs from p.loadURDF or p.createMultiBody
        anchorA_local, anchorB_local: 3-list of local anchor positions in each body's frame
        k: spring stiffness (N/m)
        c: damping (N*s/m), if None will use ~0.9*critical damping based on unit masses

    """
    def __init__(self, bodyA, bodyB, anchorA_local=[0,0,0], anchorB_local=[0,0,0],
                 anchorC_local=[0,0,0], anchorD_local=[0,0,0],
                 k=10.0, c=None):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.anchorA_local = np.array(anchorA_local)
        self.anchorB_local = np.array(anchorB_local)
        self.anchorC_local = np.array(anchorC_local)
        self.anchorD_local = np.array(anchorD_local)
        self.k = k

        
        # effective damping
        mA = 1.0; mB = 1.0
        mu = (mA*mB)/(mA+mB)
        c_crit = 2*np.sqrt(k*mu)
        self.c = c if c is not None else 0.9*c_crit

        # rest length from initial anchor positions
        pA = world_from_local(bodyA, self.anchorA_local, -1)
        pB = world_from_local(bodyB, self.anchorB_local, -1)
        pC = world_from_local(bodyA, self.anchorC_local, -1)
        pD = world_from_local(bodyB, self.anchorD_local, -1)
        self.rest_len = np.linalg.norm(pB - pA)
        self.rest_len_C = np.linalg.norm(pD - pC)
        mid = 0.5 * (pC + pD)
        p.addUserDebugText('b',pB,[1,0,0],1)
        p.addUserDebugText('a',pA,[1,0,0],1)
        p.addUserDebugText('c',pC,[0,1,0],1)
        p.addUserDebugText('d',pD,[0,1,0],1)

        #self.line_id = None
        self.line_id = p.addUserDebugLine(pA.tolist(), pB.tolist(),
                                          lineColorRGB=[0,1,0], lineWidth=2.0, lifeTime=0.05)
        self.line_id = p.addUserDebugLine(pC.tolist(), pD.tolist(),
                                          lineColorRGB=[0,1,0], lineWidth=2.0, lifeTime=0.05)

        make_ligament(self,bodyA,bodyB,mid,pC,pD)
        #time.sleep(500)
    def step(self):
        pA = world_from_local(self.bodyA, self.anchorA_local, -1)
        pB = world_from_local(self.bodyB, self.anchorB_local, -1)
        pC = world_from_local(self.bodyA, self.anchorC_local, -1)
        pD = world_from_local(self.bodyB, self.anchorD_local, -1)

        d = pB - pA
        d2 = pD - pC
        dist = np.linalg.norm(d)
        dist2 = np.linalg.norm(d2)
        if dist < 1e-8 and dist2 < 1e-8:
            return
        
        dirn = d/dist
        dirn2 = d2/dist2

        vA = point_velocity(self.bodyA, pA, -1)
        vB = point_velocity(self.bodyB, pB, -1)
        vC = point_velocity(self.bodyA, pC, -1)
        vD = point_velocity(self.bodyB, pD, -1)
        
        v_along = np.dot((vB-vA), dirn)
        v_along2 = np.dot((vD-vC), dirn2)

        fs = -self.k * (dist - self.rest_len)
        fd = -self.c * v_along
        f_total = (fs+fd) * dirn

        # apply to both
        p.applyExternalForce(self.bodyA, -1, (-f_total).tolist(), pA.tolist(), p.WORLD_FRAME)
        p.applyExternalForce(self.bodyB, -1, f_total.tolist(),  pB.tolist(), p.WORLD_FRAME)
        p.applyExternalForce(self.bodyA, -1, (-f_total).tolist(), pC.tolist(), p.WORLD_FRAME)
        p.applyExternalForce(self.bodyB, -1, f_total.tolist(),  pD.tolist(), p.WORLD_FRAME)
        # debug line
        if self.line_id is not None:
            p.removeUserDebugItem(self.line_id)
        self.line_id = p.addUserDebugLine(pA.tolist(), pB.tolist(),
                                          lineColorRGB=[0,1,0], lineWidth=2.0, lifeTime=0.05)
        self.line_id = p.addUserDebugLine(pC.tolist(), pD.tolist(),
                                          lineColorRGB=[0,1,0], lineWidth=2.0, lifeTime=0.05)


def world_from_local(body, local_point, link=-1):
    pos, orn = p.getBasePositionAndOrientation(body) if link==-1 else p.getLinkState(body, link)[:2]
    world, _ = p.multiplyTransforms(pos, orn, local_point, [0,0,0,1])
    return np.array(world)

def point_velocity(body, world_point, link=-1):
    if link == -1:
        lin, ang = p.getBaseVelocity(body)
        pos, orn = p.getBasePositionAndOrientation(body)
    else:
        link_state = p.getLinkState(body, link, computeLinkVelocity=1)
        pos, orn, _, _, _, _, lin, ang = link_state
    r = np.array(world_point) - np.array(pos)
    return np.array(lin) + np.cross(np.array(ang), r)

def make_ligament(self,foot,leg,pos,a,b):
    a = a
    b=b
    # clothId = p.loadSoftBody("/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/ligament1.obj", 
    #                          basePosition = pos, 
    #                          baseOrientation = p.getQuaternionFromEuler([0,0,90/180*np.pi]),
    #                          scale = 1, mass = 0.1, 
    #                          useNeoHookean = 0, 
    #                          useBendingSprings=1,
    #                          useMassSpring=1, 
    #                          springElasticStiffness=80, 
    #                          springDampingStiffness=0.5, 
    #                          springDampingAllDirections = 1,
    #                            useSelfCollision = 0, 
    #                            frictionCoeff = .5,
    #                              useFaceContact=1)
    # clothId = p.loadSoftBody("/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/ligamentthin.obj", 
    #                          basePosition = pos, 
    #                          baseOrientation = p.getQuaternionFromEuler([0,0,90/180*np.pi]),
    #                          mass=0.1,
    #                         useNeoHookean=0,
    #                         useBendingSprings=1,
    #                         useMassSpring=1,
    #                         springElasticStiffness=40,
    #                         springDampingStiffness=0.3,
    #                         springBendingStiffness=5,
    #                         useSelfCollision=1,
    #                         collisionMargin=0.002,
    #                         frictionCoeff=0.5,
    #                         useFaceContact=1,
    #                         scale=2
    #                     )
                            
    # # p.createSoftBodyAnchor(clothId  ,24,-1,-1)
    # # p.createSoftBodyAnchor(clothId ,20,-1,-1)
    # # p.createSoftBodyAnchor(clothId ,15,foot,-1, a)
    # # p.createSoftBodyAnchor(clothId ,19,leg,-1, b)
    # p.changeVisualShape(clothId, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
    #  #p.addUserDebugText('b',pB,[1,0,0],1)
    # #p.addUserDebugText('ligament',[1,0,0],[1,0,0],1)
    # p.setPhysicsEngineParameter(fixedTimeStep=1./1000., numSubSteps=10)
    clothId = p.loadSoftBody("/home/catherine/FractureGym/fracturesurgeryenv/gym_fracture/envs/Assets/241206/ligamentthin.obj",
        basePosition=pos,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 90/180*np.pi]),
        scale=1,
        mass=0.05,
        useNeoHookean=0,
        useMassSpring=1,
        useBendingSprings=1,
        springElasticStiffness=8,      # stiffer -> springier/shape-preserving
        springDampingStiffness=1,    # moderate damping -> oscillation allowed
        springDampingAllDirections=1,
        springBendingStiffness=1,       # preserve rod shape
        useSelfCollision=0,             # disable initially for tuning
        collisionMargin=0.001,
        frictionCoeff=0.5,
        useFaceContact=1
    )
    print(p.getAABB(clothId))
    p.setTimeStep(1.0/500.0)
    p.setPhysicsEngineParameter(numSolverIterations=80, erp=0.1, contactERP=0.1)

    p.stepSimulation()
    # data = p.getMeshData(clothId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
   
    
    # contacts = p.getContactPoints(bodyA=leg, bodyB=clothId)
    
    
    # numVertices = data[0]
    # vertices = data[1]

    
    # for c in contacts:
    #     contact_pos = c[5]  # positionOnA
    #     vertexIndex = findClosestVertex(contact_pos, vertices)
    #     p.createSoftBodyAnchor(clothId, vertexIndex, leg,-1, contact_pos)
    #     #print(f'anchoring vertex {vertexIndex} to leg')
    
    # contacts = p.getContactPoints(bodyA=foot, bodyB=clothId)
    
    
    
    # for c in contacts:
    #     contact_pos = c[5]  # positionOnA
    #     vertexIndex = findClosestVertex(contact_pos, vertices)
    #     p.createSoftBodyAnchor(clothId, vertexIndex, foot,-1, contact_pos)
    # #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
    auto_anchor_ligament(clothId, bodyA=foot, bodyB=leg, axis=0, num_anchors=10)

def findClosestVertex(contactPos, vertices):
    vertices_np = np.array(vertices)
    contact_np = np.array(contactPos)
    distances = np.linalg.norm(vertices_np - contact_np, axis=1)
    return np.argmin(distances)



def auto_anchor_ligament(clothId, bodyA, bodyB, axis=0, num_anchors=2):
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
    axis_vals = verts[:, axis]

    # find min/max along that axis = ends of ligament
    min_val, max_val = np.min(axis_vals), np.max(axis_vals)

    # indices sorted by distance from each end
    endA_ids = np.argsort(np.abs(axis_vals - min_val))[:num_anchors]
    endB_ids = np.argsort(np.abs(axis_vals - max_val))[:num_anchors]

    # create anchors at those vertices
    for vid in endA_ids:
        p.createSoftBodyAnchor(clothId, int(vid), bodyA, -1)
    for vid in endB_ids:
        p.createSoftBodyAnchor(clothId, int(vid), bodyB, -1)

    print(f"Anchored {len(endA_ids)} vertices to bodyA and {len(endB_ids)} to bodyB")
