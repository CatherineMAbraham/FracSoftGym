import numpy as np
import pybullet as p
import math

def vec(x): return np.array(x, dtype=float)

class ElasticBand:
    def __init__(self, bodyA, linkA, bodyB, linkB,
                 young_modulus, area, rest_length,
                 damping_ratio=0.5, exponent=1.5):
        self.bodyA = bodyA
        self.linkA = linkA
        self.bodyB = bodyB
        self.linkB = linkB
        self.band_id = None
        self.force_text_id = None
        # Physical Properties
        self.E = young_modulus
        self.A = area
        self.L0 = rest_length
        self.k = (self.E * self.A) / self.L0
        
        # Realism Parameters
        self.exponent = exponent  # 1.0 = linear, 1.5-2.0 = realistic tissue/rubber
        self.damping_ratio = damping_ratio
        self.last_force_vector = np.zeros(3)

    def _get_pose_vel(self, body, link,local_offset=[0,-0.01,0]):
        if link == -1:
            pos, orn= p.getBasePositionAndOrientation(body)
            vel, _ = p.getBaseVelocity(body)
        else:
            st = p.getLinkState(body, link, computeLinkVelocity=True)
            pos,orn = st[0], st[1]
            vel = st[6]

        world_pos, _ = p.multiplyTransforms(pos, orn, local_offset, [0,0,0,1])
        return vec(world_pos), vec(vel)

    def step(self):
        posA, velA = self._get_pose_vel(self.bodyA, self.linkA,local_offset=[0,0.0,-0.01])
        posB, velB = self._get_pose_vel(self.bodyB, self.linkB,local_offset=[0,-0.0015,0.04])

        delta = posB - posA
        L = np.linalg.norm(delta)
        #print(f"Band length: {L:.4f} m, Stretch: {L - self.L0:.4f} m")
        
        # 1. Slack Check
        if L <= self.L0:
            self.last_force_vector = np.zeros(3)
            return

        # Direction vector
        d = delta / L
        # Displacement (Stretch)
        x = L - self.L0

        # 2. Non-Linear Spring Component (The 'Toe Region')
        # F = k * x^n (n > 1 creates the J-curve characteristic of ligaments)
        spring_force = self.k * (x ** self.exponent)

        # 3. Non-Linear Damping
        # Scaling damping by 'x' ensures the band doesn't 'hit' the robot 
        # with high damping forces the moment it becomes taut.
        rel_vel = np.dot((velB - velA), d)
        damping_force = self.damping_ratio * x * rel_vel 

        # 4. Total Force & Safety
        F = spring_force + damping_force
        F = np.clip(F, 0, 50) # Prevents simulation explosion
        
        F_vec = F * d
        self.last_force_vector = F_vec.copy()

        # Apply Force
        p.applyExternalForce(self.bodyA, self.linkA, (-F_vec).tolist(), posA.tolist(), p.WORLD_FRAME)
        p.applyExternalForce(self.bodyB, self.linkB, (F_vec).tolist(), posB.tolist(), p.WORLD_FRAME)

        color = [1, 0, 0] if x > 0 else [0, 1, 0] 
        width = max(1, int(1)) 
        if self.band_id is None: 
            self.band_id = p.addUserDebugLine(posA, posB, color, 5) 
        else:
            p.addUserDebugLine( posA, posB, color, 5, replaceItemUniqueId=self.band_id ) 
        # ------------------------------- 
        # Force readout 
        # ------------------------------- 
        # text = f"Band force: {self.get_force():.1f} N" 
        # if self.force_text_id is None: 
        #     self.force_text_id = p.addUserDebugText( text, posA + [0, 0.15, 0.15], textSize=1.2 ) 
        # else: 
        #     p.addUserDebugText( text, posA + [0, 0.15, 0.15], textSize=1.2, replaceItemUniqueId=self.force_text_id)

    def get_force(self):
        """Returns the magnitude of the current tension."""
        return np.linalg.norm(self.last_force_vector)
    
    
