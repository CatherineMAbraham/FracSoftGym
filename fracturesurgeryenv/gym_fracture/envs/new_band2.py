import numpy as np
import pybullet as p
import math
import time
def vec(x): return np.array(x, dtype=float)

class ElasticBand:
    def __init__(self, bodyA, linkA, bodyB, linkB,
                 young_modulus, area, rest_length,
                 damping_ratio=0.5, exponent=1.5,num_springs=3):
        self.bodyA = bodyA
        self.linkA = linkA
        self.bodyB = bodyB
        self.linkB = linkB
        self.band_id = None
        self.force_text_id = None
        # Physical Properties
        self.E = young_modulus
        self.A = area
        #self.L0 = rest_length
        #self.k = (self.E * self.A) / self.L0
        
        # Realism Parameters
        self.exponent = exponent  # 1.0 = linear, 1.5-2.0 = realistic tissue/rubber
        self.damping_ratio = damping_ratio
        self.num_springs = num_springs
        radius = 0.01      # distance from center (creates bending resistance)
        posA, velA = self._get_pose_vel(self.bodyA, self.linkA,local_offset=[0,0.0,-0.01])
        posB, velB = self._get_pose_vel(self.bodyB, self.linkB,local_offset=[0,-0.0015,0.04])
        ornA = p.getLinkState(self.bodyA, self.linkA)[1] if self.linkA != -1 else p.getBasePositionAndOrientation(self.bodyA)[1]
        ornB = p.getLinkState(self.bodyB, self.linkB)[1] if self.linkB != -1 else p.getBasePositionAndOrientation(self.bodyB)[1]
        #self.L0 = np.linalg.norm(np.array(posB) - np.array(posA))
        self.last_force_vector = np.zeros(3)
        # critical damping

        # Attachment offsets in circular pattern
        #angles = np.linspace(0, 2*np.pi, num_springs, endpoint=False)
        width = 0.005 #5mm wide rectangle
        #create 3 attachment points in a line across the width of the band
                
        # self.local_offsets_A = [np.array([-0.07402802, -0.02543187, -0.34601694]), np.array([-0.07401566, -0.02544761, -0.35315976])]
        # self.local_offsets_B = [np.array([-0.26201391, -0.48721281, -0.01182608]), np.array([-0.26201391, -0.48721281, -0.01182608])]
        # self.world_offset = [np.array([-0.07402802, -0.02543187, -0.34601694]), np.array([-0.07401566, -0.02544761, -0.35315976])]
        # self.world_offset_B = [np.array([-0.26201391, -0.48721281, -0.01182608]), np.array([-0.26201391, -0.48721281, -0.01182608])]
        #self.local_offsets_A= [np.array([-0.02428431, -0.06002578, -0.08079728]), np.array([-0.01714148, -0.0600186 , -0.08078326])]
        #self.local_offsets_B = [np.array([ 0.01999438, -0.00067805, -0.0100396 ]), np.array([ 0.01999438,  0.00432195, -0.0100396 ])]
        #self.local_offsets_A= [np.array([ 0.3769989,  -0.08445022,  0.07449624]),np.array([ 0.3769989,  -0.07730737,  0.07449624])]
        #self.local_offsets_B=[np.array([ 0.3769989,  -0.12016451,  0.07449624]),np.array([ 0.3769989,  -0.12016451,  0.07949624])]
        self.local_offsets_A=[np.array([ 0.02000636, -0.00083421,  0.04432721]), np.array([ 0.02000484, -0.00081395,  0.03718438])]
        self.local_offsets_B= [np.array([ 0.01999438, -0.00067805, -0.0100396 ]), np.array([ 0.01999438,  0.00432195, -0.0100396 ])]
        posA = np.array(posA)
        posB = np.array(posB)
        velA = np.array(velA)
        velB = np.array(velB)

        self.RA = np.array(p.getMatrixFromQuaternion(ornA)).reshape(3,3)
        self.RB = np.array(p.getMatrixFromQuaternion(ornB)).reshape(3,3)
        base_L0 = np.linalg.norm((posB + self.RB @ self.local_offsets_B[0]) - (posA + self.RA @ self.local_offsets_A[0]))
        self.L0_list = [np.linalg.norm(self.local_offsets_B[i] - self.local_offsets_A[i]) for i in range(len(self.local_offsets_A))]
        recruitment_spread = 0.0 # 2% variation
        if num_springs > 1:
            for i in range(num_springs):
                # Linearly vary L0 for each fiber
                factor = 1.0 + (i * recruitment_spread / (num_springs - 1))
                self.L0_list.append(base_L0 * factor)
        else:
            self.L0_list.append(base_L0)
        
        self.k = (self.E * self.A) / base_L0
        
        mA = p.getDynamicsInfo(self.bodyA, -1)[0]
        mB = p.getDynamicsInfo(self.bodyB, -1)[0]

        if mB == 0:
            m_eff = mA
        else:
            m_eff = (mA * mB) / (mA + mB)
        #m_eff = (mA * mB) / (mA + mB)

        self.c = 2 * math.sqrt(self.k * m_eff)  
        #self.c = 2
    def _get_pose_vel(self, body, link,local_offset=[0,-0.01,0]):
        if link == -1:
            pos, orn= p.getBasePositionAndOrientation(body)
            vel, _ = p.getBaseVelocity(body)
        else:
            st = p.getLinkState(body, link, computeLinkVelocity=True)
            pos,orn = st[0], st[1]
            vel = st[6]

        #world_pos, _ = p.multiplyTransforms(pos, orn, local_offset, [0,0,0,1])
        return vec(pos), vec(vel)

    def step(self):
        total_force_mag = 0.0
        total_stretch = 0.0
        active_springs = 0

        posA, velA = self._get_pose_vel(self.bodyA, self.linkA,local_offset=[0,0,0])#, local_offset=[0.01, -0.0015, 0.04])
        posB, velB = self._get_pose_vel(self.bodyB, self.linkB,local_offset=[0,0,0])#, local_offset=[0.01, 0.0, -0.01])

        ornA = p.getLinkState(self.bodyA, self.linkA)[1] if self.linkA != -1 else p.getBasePositionAndOrientation(self.bodyA)[1]
        ornB = p.getLinkState(self.bodyB, self.linkB)[1] if self.linkB != -1 else p.getBasePositionAndOrientation(self.bodyB)[1]

        self.RA = np.array(p.getMatrixFromQuaternion(ornA)).reshape(3, 3)
        self.RB = np.array(p.getMatrixFromQuaternion(ornB)).reshape(3, 3)

        nu = 0.45
        mu = self.E / (2 * (1 + nu))
        lam = self.E * nu / ((1 + nu) * (1 - 2 * nu))
        spring_area = self.A / len(self.local_offsets_A)

        for i, (localA, localB) in enumerate(zip(self.local_offsets_A, self.local_offsets_B)):
            worldA = posA + self.RA @ localA #np.array([ 0.3769989,  -0.08445022,  0.07449624])#posA + self.RA @ localA
            worldB = posB + self.RB @ localB #np.array([ 0.3769989,  -0.12016451,  0.07449624])#posB + self.RB @ localB

            #print(f'A erro {np.linalg.norm(worldA -[ 0.3769989,  -0.08445022,  0.07449624] )}')
            #print(f'B erro {np.linalg.norm(worldB -[ 0.3769989,  -0.12016451,  0.07449624] )}')
            delta = worldB - worldA
            dist = np.linalg.norm(delta)

            if dist < 1e-6:
                continue

            current_L0 = self.L0_list[i]
            lambda_i = dist / current_L0

            if lambda_i <= 1.0:
                continue  # tension only

            direction = delta / dist
            stretch = dist - current_L0

            mu = self.E / (2 * (1 + 0.45)) 

            # The force magnitude for a single strand
            force_mag = spring_area * mu * (lambda_i - (1 / (lambda_i**2)))

            # Add damping separately
            rel_vel = np.dot((velB - velA), direction)
            force_mag += (self.c / len(self.local_offsets_A)) * rel_vel
            force_mag = min(force_mag, 10)  # Cap the force to prevent instability
            total_force_mag += force_mag
            total_stretch += stretch
            active_springs += 1

            F_vec = force_mag * direction
            self.last_force_vector = F_vec.copy()
            #print(force_mag)
            p.applyExternalForce(self.bodyA, self.linkA, (-F_vec).tolist(),
                                worldA.tolist(), p.WORLD_FRAME)
            p.applyExternalForce(self.bodyB, self.linkB, (F_vec).tolist(),
                                worldB.tolist(), p.WORLD_FRAME)

            colour = [1, 0, 0] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1]
            p.addUserDebugLine(worldA, worldB, colour, 2)
            #p.addUserDebugText(f'A{i}', worldA, colour, 1.5)
            #p.addUserDebugText(f'B{i}', worldB, colour, 1.5)
#            print(stretch, force_mag)

        if active_springs > 0:
            return total_stretch / active_springs, total_force_mag

        return np.linalg.norm(worldA - worldB), 0.0

            # Return default values if no stretch detected
            

    def get_force(self):
        """Returns the magnitude of the current tension."""
        return np.linalg.norm(self.last_force_vector)
    
    