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
        angles = np.linspace(-width/2, width/2, num_springs)
        self.local_offsets_A = []
        self.local_offsets_B = []

        # for angle in angles:
        #     x = radius * math.cos(angle)
        #     y = radius * math.sin(angle)
        #     self.local_offsets.append(np.array([x, y, 0]))
        
        for x_offset in np.linspace(-width/2, width/2, num_springs):
              self.local_offsets_A.append(np.array([radius, 0, x_offset])) # Flat along X
              self.local_offsets_B.append(np.array([radius, 0, x_offset])) 
        self.local_offset_A = [np.array([-0.07402802, -0.02543187, -0.34601694]), np.array([-0.07401566, -0.02544761, -0.35315976])]
        self.local_offset_B = [np.array([-0.26201391, -0.48721281, -0.01182608]), np.array([-0.26201391, -0.48721281, -0.01182608])]
        posA = np.array(posA)
        posB = np.array(posB)
        velA = np.array(velA)
        velB = np.array(velB)

        self.RA = np.array(p.getMatrixFromQuaternion(ornA)).reshape(3,3)
        self.RB = np.array(p.getMatrixFromQuaternion(ornB)).reshape(3,3)
        base_L0 = np.linalg.norm((posB + self.RB @ self.local_offsets_B[0]) - (posA + self.RA @ self.local_offsets_A[0]))
        self.L0_list = []
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
        # Initialize default return values
        stretch = 0.0
        force_mag = 0.0
        total_force_mag = 0.0
        total_stretch = 0.0
        active_springs = 0
        

        # Effective mass for damping
        posA, velA = self._get_pose_vel(self.bodyA, self.linkA,local_offset=[0.0,0.0,-0.01])
        posB, velB = self._get_pose_vel(self.bodyB, self.linkB,local_offset=[0.0,-0.0015,0.04])
        #ornA = p.getLinkState(self.bodyA, self.linkA)[1] if self.linkA != -1 else p.getBasePositionAndOrientation(self.bodyA)[1]
        #ornB = p.getLinkState(self.bodyB, self.linkB)[1] if self.linkB != -1 else p.getBasePositionAndOrientation(self.bodyB)[1]
        ornA = p.getLinkState(self.bodyA, self.linkA)[1] if self.linkA != -1 else p.getBasePositionAndOrientation(self.bodyA)[1]
        self.RA = np.array(p.getMatrixFromQuaternion(ornA)).reshape(3,3)
        ornB = p.getLinkState(self.bodyB, self.linkB)[1] if self.linkB != -1 else p.getBasePositionAndOrientation(self.bodyB)[1]
        self.RB = np.array(p.getMatrixFromQuaternion(ornB)).reshape(3,3)
        for i, localA in enumerate(self.local_offsets_A):
            #print(f"Local attachment point {i}: {localA}")
            
            worldA = posA + self.RA @ localA
            worldB = posB + self.RB @ self.local_offsets_B[i]

            delta = worldB - worldA
            dist = np.linalg.norm(delta)
            current_L0 = self.L0_list[i]
            lambda_stretch = dist/current_L0
            #print(f"Band length: {dist:.4f} m, Stretch: {dist - self.L0:.4f} m")
            #time.sleep(10)
            if dist < 1e-6:
                continue
            
            nu = 0.45
            mu = self.E / (2 * (1 + nu))

            direction = delta / dist
            stretch = dist - current_L0
            spring_area = self.A / len(self.local_offsets_A)
            # if stretch <= 0:
            #     continue  # tension only
            # # if stretch <= 0:
            # #     continue  # tension only
            # # if stretch > 0:
            # #     # NORMAL TENSION (Neo-Hookean)
                
            # # else:
            # #     # COMPRESSION (The "Collision" Force)
            # #     # We use a linear high-stiffness "Bumper" to mimic mesh compression
            # #     # This simulates the physical volume of the ligament being squashed
            # #     compression_stiffness = self.E * self.A * 10 # 10x multiplier for hard contact
            # #     Fs = compression_stiffness * stretch # stretch is negative here, creating pushing force

            # # Linear spring
            # #Fs = self.k * (stretch** self.exponent)
            # # In step()
            # Fs = spring_area * mu * (lambda_stretch - 1/(lambda_stretch**2))
            # #Fs = spring_area * mu * (lambda_stretch - 1/lambda_stretch**2)
            # #Fs = self.A *mu * (lambda_stretch - 1/lambda_stretch**2)
            # # Relative velocity along spring direction
            # rel_vel = np.dot((velB - velA), direction)
            # actual_damping = self.c * (lambda_stretch ** 2)/len(self.local_offsets) # Damping increases with stretch
            # Fd = actual_damping * rel_vel
            # #Fd = self.c * rel_vel

            # force_mag = Fs + Fd
            if stretch > 0:
                # Match the FEM material law exactly
                # F = Area * mu * (lambda - 1/lambda^2)
                Fs = (self.A / self.num_springs) * mu * (lambda_stretch - 1/(lambda_stretch**2))
                
                # Critical: Damping must be scaled to the new stiffness
                # c = 2 * sqrt(k * m)
                rel_vel = np.dot((velB - velA), direction)
                Fd = (self.c / self.num_springs) * rel_vel
                
                force_mag = max(0.0, Fs + Fd)
                force_mag = max(0.0, force_mag)
                total_force_mag += force_mag
                total_stretch += stretch
                active_springs += 1

                forc_vec = force_mag * direction
                F_vec = forc_vec.copy()
                self.last_force_vector = F_vec.copy()


            # Apply Force
                # p.applyExternalForce(self.bodyA, self.linkA, (-F_vec).tolist(), posA.tolist(), p.WORLD_FRAME)
                # p.applyExternalForce(self.bodyB, self.linkB, (F_vec).tolist(), posB.tolist(), p.WORLD_FRAME)
                p.applyExternalForce(self.bodyA, self.linkA, (-F_vec).tolist(),
                                        worldA.tolist(), p.WORLD_FRAME)

                p.applyExternalForce(self.bodyB, self.linkB, (F_vec).tolist(),
                                        worldB.tolist(), p.WORLD_FRAME)

                #color = [1, 0, 0] if stretch > 0 else [0, 1, 0] 
                
                width = max(1, int(1)) 
                # make each spring a different color for visualization
                colour = [1,0,0] if i == 0 else [0,1,0] if i == 1 else [0,0,1]
                p.addUserDebugLine(worldA, worldB, colour, 2,lifeTime=0.1) 
                #print(stretch, force_mag)
            else: 
                continue
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
        
        if active_springs > 0:
            return total_stretch / active_springs, total_force_mag

        # Return default values if no stretch detected
        return stretch, force_mag

    def get_force(self):
        """Returns the magnitude of the current tension."""
        return np.linalg.norm(self.last_force_vector)
    
    