import math
from symtable import Class
from gym_fracture.envs.dynamics import change_ligament_dynamics
import numpy as np
import pybullet as p
import os
import time

class Ligament:
    def __init__(self, name, foot, leg, a, b, orientation, scale, youngs_modulus):
        self.name = name
        self.foot = foot
        self.leg = leg
        self.point_a = a
        self.point_b = b
        self.orientation = orientation
        self.scale = scale
        self.youngs_modulus = youngs_modulus
        self.ligament_id = None
        self.force = 0.0

    def world_from_local(self,body, local_point, link=-1):
        pos, orn = p.getBasePositionAndOrientation(body) if link==-1 else p.getLinkState(body, link)[:2]
        world, _ = p.multiplyTransforms(pos, orn, local_point, [0,0,0,1])
        return np.array(world)

    def get_anchor_local_offsets(self,body, link, anchor_world_positions):
        if link == -1:
            pos, orn = p.getBasePositionAndOrientation(body)
        else:
            # Use index 4 and 5 for the world link frame position/orientation
            state = p.getLinkState(body, link)
            pos, orn = state[2], state[3] 
            
        inv_pos, inv_orn = p.invertTransform(pos, orn)
        local_offsets = []
        for wp in anchor_world_positions:
            local, _ = p.multiplyTransforms(inv_pos, inv_orn, wp, [0,0,0,1])
            local_offsets.append(np.array(local))
        return local_offsets

    def local_to_local(self,body_from, body_to, local_point, link_from=-1, link_to=-1):
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
    def radius_spring(self,foot,leg,a,b):
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

    def measure_ligament_force(self,body, dt):

        vel_before,_ = p.getBaseVelocity(body)

        p.stepSimulation()

        vel_after,_ = p.getBaseVelocity(body)

        mass = p.getDynamicsInfo(body,-1)[0]

        force = mass * (np.array(vel_after) - np.array(vel_before)) / dt

        return np.linalg.norm(force)
    def make_ligament(self,env,name,foot,leg,a,b, orientation,scale, youngs_modulus):
        a = a
        b=b
        pC = a#world_from_local(foot, a, 0)
        pD = b#world_from_local(leg, b, 0)
        #print(pC,pD)
        #p.addUserDebugText( f"pC",pC, [1,0,0],1.0)
        #p.addUserDebugText( f"pD",pD, [0,1,0],1.0)
        worldA,worldB = self.radius_spring(foot, leg, a, b)
        #print(np.linalg.norm(worldA-worldB))
        orientation = orientation
        scale = scale
        name = name
        mid = 0.5 * (worldA + worldB)
        currentDir = os.path.dirname(os.path.abspath(__file__))
        lig_path = os.path.join(currentDir, "Assets/rect00125.vtk")
        
        E = youngs_modulus
        nu = 0.45
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        name = p.loadSoftBody(#"/home/catherine/FractureSoftGym/fracturesurgeryenv/gym_fracture/envs/Assets/ligacc.obj",
        lig_path,
            mass=0.1,
            basePosition=mid-[0.005,0.0,0],
            baseOrientation=p.getQuaternionFromEuler([90/180*np.pi, 0, 90/180*np.pi]),
            scale=1,
            useNeoHookean=1,
            useMassSpring=0,
            NeoHookeanMu=mu,
            NeoHookeanLambda=lam,
            useBendingSprings=1,
            frictionCoeff=0.5,
            NeoHookeanDamping=0.02,
            #repulsionStiffness=0.5 * mu,
            collisionMargin=0.005
        )
        colour = [250/255,11/255,58/255,1]
        p.changeVisualShape(name, -1, rgbaColor=colour)
        change_ligament_dynamics(name)    
        anchorA_vertices, anchorB_vertices = self.auto_anchor_ligament(name, bodyA=foot, bodyB=leg, worldA=worldA, worldB=worldB, axis=0, num_anchors=5)
        
        force = self.measure_ligament_force(name, dt=1/240)
        
        
        return worldA, worldB



    def auto_anchor_ligament(self, clothId, bodyA, bodyB, worldA, worldB, axis=0, num_anchors=2):
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
        
        distA = np.linalg.norm(verts - worldA, axis=1)
        distB = np.linalg.norm(verts - worldB, axis=1)
        
        anchorA_vertices = np.where(distA < 0.005)[0]
        anchorB_vertices = np.where(distB < 0.005)[0]
        anchorA_vertices = anchorA_vertices[:num_anchors]
        anchorB_vertices = anchorB_vertices[:num_anchors]
        ligament_dir = worldB - worldA
        unit_dir = ligament_dir / np.linalg.norm(ligament_dir)
        safety_offset = 0.005
        b_verts =self.local_to_local(clothId, bodyB, -1)
        a_verts =self.local_to_local(clothId, bodyA, -1)
        a_coords = a_verts[anchorA_vertices]
        b_coords = b_verts[anchorB_vertices]
        local_offsets_A = self.get_anchor_local_offsets(bodyA,1, verts[anchorA_vertices])
        local_offsets_B = self.get_anchor_local_offsets(bodyB,-1,  verts[anchorB_vertices])
        #p.addUserDebugText(f"A", verts[anchorA_vertices[0]], [1,0,0], 2.0)
        #p.addUserDebugText(f"B", verts[anchorB_vertices[0]], [0,1,0], 2.0)
        for i, vid in enumerate(anchorA_vertices):
            p.createSoftBodyAnchor(clothId, int(vid), bodyA, 1,a_coords[i].tolist())
        for i, vid in enumerate(anchorB_vertices):
            offset_pos = b_verts[vid] + (unit_dir * safety_offset)
            p.createSoftBodyAnchor(clothId, int(vid), bodyB, -1,b_coords[i].tolist())
        return anchorA_vertices, anchorB_vertices
    
