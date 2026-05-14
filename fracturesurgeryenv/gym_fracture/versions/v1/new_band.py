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
        # posA, velA = self._get_pose_vel(self.bodyA, self.linkA,local_offset=[0,0.0,-0.01])
        # posB, velB = self._get_pose_vel(self.bodyB, self.linkB,local_offset=[0,-0.0015,0.04])

        # delta = posB - posA
        # L = np.linalg.norm(delta)
        # #print(f"Band length: {L:.4f} m, Stretch: {L - self.L0:.4f} m")
        
        # # 1. Slack Check
        # if L <= self.L0:
        #     #if less than rest length don't add any force 
        #     self.last_force_vector = np.zeros(3)
        #     return

        # # Direction vector
        # d = delta / L
        # # Displacement (Stretch)
        # x = L - self.L0
        # strain = x / self.L0

        # # 2. Toe region
        # # F = k * x^n 
        # if strain < 0.1:
        #     spring_force = self.k * (x ** self.exponent)
        # else: 
        #     spring_force = self.k * x  

        # # 3. Non-Linear damping
        
        # rel_vel = np.dot((velB - velA), d)
        # damping_force = self.damping_ratio * x * rel_vel 

        # # 4. Total force & safety
        # F = spring_force + damping_force
        # F = np.clip(F, 0, 50) # Prevents simulation explosion, need to tune this 
        
        # F_vec = F * d
        # self.last_force_vector = F_vec.copy()
        num_springs = 1
        radius = 0.01      # distance from center (creates bending resistance)
        posA, velA = self._get_pose_vel(self.bodyA, self.linkA,local_offset=[0,0.0,-0.01])
        posB, velB = self._get_pose_vel(self.bodyB, self.linkB,local_offset=[0,-0.0015,0.04])
        ornA = p.getLinkState(self.bodyA, self.linkA)[1] if self.linkA != -1 else p.getBasePositionAndOrientation(self.bodyA)[1]
        ornB = p.getLinkState(self.bodyB, self.linkB)[1] if self.linkB != -1 else p.getBasePositionAndOrientation(self.bodyB)[1]
        L0 = np.linalg.norm(np.array(posB) - np.array(posA))
        
        #L0 = 0.3           # rest length
        k = (1e6 * 5e-6) / L0
    # k = 800            # axial stiffness

        # Effective mass for damping
        mA = p.getDynamicsInfo(self.bodyA, -1)[0]
        mB = p.getDynamicsInfo(self.bodyB, -1)[0]

        if mB == 0:
            m_eff = mA
        else:
            m_eff = (mA * mB) / (mA + mB)
        #m_eff = (mA * mB) / (mA + mB)

        c = 2 * math.sqrt(k * m_eff)  # critical damping

        # Attachment offsets in circular pattern
        angles = np.linspace(0, 2*np.pi, num_springs, endpoint=False)
        local_offsets = []

        for angle in angles:
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            local_offsets.append(np.array([x, y, 0]))
        
        
        posA = np.array(posA)
        posB = np.array(posB)
        velA = np.array(velA)
        velB = np.array(velB)

        RA = np.array(p.getMatrixFromQuaternion(ornA)).reshape(3,3)
        RB = np.array(p.getMatrixFromQuaternion(ornB)).reshape(3,3)

        for localA in local_offsets:

            localB = localA.copy()

            worldA = posA + RA @ localA
            worldB = posB + RB @ localB

            delta = worldB - worldA
            dist = np.linalg.norm(delta)

            if dist < 1e-6:
                continue

            direction = delta / dist
            stretch = dist - L0

            if stretch <= 0:
                continue  # tension only

            # Linear spring
            Fs = k * (stretch** self.exponent)

            # Relative velocity along spring direction
            rel_vel = np.dot((velB - velA), direction)
            Fd = c * rel_vel

            force_mag = Fs + Fd
            force_mag = max(0.0, force_mag)

            forc_vec = force_mag * direction
            F_vec = forc_vec.copy()
            self.last_force_vector = F_vec.copy()


        # Apply Force
            # p.applyExternalForce(self.bodyA, self.linkA, (-F_vec).tolist(), posA.tolist(), p.WORLD_FRAME)
            # p.applyExternalForce(self.bodyB, self.linkB, (F_vec).tolist(), posB.tolist(), p.WORLD_FRAME)
            p.applyExternalForce(self.bodyA, -1, F_vec.tolist(),
                                     worldA.tolist(), p.WORLD_FRAME)

            p.applyExternalForce(self.bodyB, -1, (-F_vec).tolist(),
                                     worldB.tolist(), p.WORLD_FRAME)

            color = [1, 0, 0] if x > 0 else [0, 1, 0] 
            width = max(1, int(1)) 
            p.addUserDebugLine(worldA, worldB, color, 2) 
            # if self.band_id is None: 
            #     self.band_id = p.addUserDebugLine(posA, posB, color, 1) 
            # else:
            #     p.addUserDebugLine( worldA, worldB, color, 5, replaceItemUniqueId=self.band_id ) 
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
    
    