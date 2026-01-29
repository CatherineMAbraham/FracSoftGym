import pybullet as p
import numpy as np
from gym_fracture.envs import create_lig
def vec(x): return np.array(x, dtype=float)

class ElasticBand:
    def __init__(self, bodyA, linkA, bodyB, linkB,
                 young_modulus, area, rest_length,
                 damping_ratio=0.5, exponent=1.5):
        self.bodyA = bodyA
        self.linkA = linkA
        self.bodyB = bodyB
        self.linkB = linkB

        # Physical Properties
        self.E = young_modulus
        self.A = area
        self.L0 = rest_length
        self.k = (self.E * self.A) / self.L0
        
        # Realism Parameters
        self.exponent = exponent  # 1.0 = linear, 1.5-2.0 = realistic tissue/rubber
        self.damping_ratio = damping_ratio
        self.last_force_vector = np.zeros(3)

    def _get_pose_vel(self, body, link):
        if link == -1:
            pos, _ = p.getBasePositionAndOrientation(body)
            vel, _ = p.getBaseVelocity(body)
        else:
            st = p.getLinkState(body, link, computeLinkVelocity=True)
            pos = st[0]
            vel = st[6]
        return vec(pos), vec(vel)

    def step(self):
        posA, velA = self._get_pose_vel(self.bodyA, self.linkA)
        posB, velB = self._get_pose_vel(self.bodyB, self.linkB)

        delta = posB - posA
        L = np.linalg.norm(delta)
        
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
        F = np.clip(F, 0, 100) # Prevents simulation explosion
        
        F_vec = F * d
        self.last_force_vector = F_vec.copy()

        # Apply Force
        p.applyExternalForce(self.bodyA, self.linkA, (-F_vec).tolist(), posA.tolist(), p.WORLD_FRAME)
        p.applyExternalForce(self.bodyB, self.linkB, (F_vec).tolist(), posB.tolist(), p.WORLD_FRAME)

    def get_force(self):
        """Returns the magnitude of the current tension."""
        return np.linalg.norm(self.last_force_vector)
    
    import numpy as np

def estimate_ee_force(self,robot_id, ee_link_index, joint_indices):
    # joint_indices: list of controlled joint indices in the same order used by controllers
    q = []
    dq = []
    taus = []
    for j in joint_indices:
        js = p.getJointState(robot_id, j)
        q.append(js[0])
        dq.append(js[1])
        taus.append(js[3])   # measured joint torque

    q = list(q); dq = list(dq)
    ddq = [0.0] * len(q)

    # localPosition at EE (zero vector -> Jacobian at link origin)
    local_pos = [0.0, 0.0, 0.0]
    J_lin, J_ang = p.calculateJacobian(robot_id, ee_link_index, local_pos, q, dq, ddq)
    J = np.array(J_lin)    # shape (3, n)
    tau = np.array(taus)   # shape (n,)

    # Solve for force: F = pinv(J.T) @ tau
    # (J.T) is (n x 3), so pseudoinverse yields (3 x n)
    Ft = np.linalg.pinv(J.T) @ tau    # shape (3,)
    resultant_force = np.linalg.norm(Ft)
    #print("Estimated EE force:", resultant_force)
    return Ft  # world-frame force vector (approx)

def estimate_ee_force_compensated(self, robot_id, ee_link_index, joint_indices):
    # 1. Gather current state
    q = []
    dq = []
    taus_measured = []
    for j in joint_indices:
        js = p.getJointState(robot_id, j)
        q.append(js[0])
        dq.append(js[1])
        taus_measured.append(js[3]) # Total torque (Gravity + Dynamics + External)

    # 2. Calculate "Free-Space" Torques
    # We want to know what torques are needed just for gravity and Coriolis.
    # We pass desired acceleration (ddq) as zeros to isolate static/velocity effects.
    ddq_zeros = [0.0] * len(joint_indices)
    taus_dynamic = p.calculateInverseDynamics(robot_id, q, dq, ddq_zeros)
    
    # Convert to numpy for vector math
    tau_measured_vec = np.array(taus_measured)
    tau_dynamic_vec = np.array(taus_dynamic)

    # 3. Isolate External Torque
    # Tau_ext = Tau_measured - Tau_gravity_and_coriolis
    tau_ext = tau_measured_vec - tau_dynamic_vec

    # 4. Map to Cartesian Force using the Jacobian
    local_pos = [0.0, 0.0, 0.0]
    J_lin, _ = p.calculateJacobian(robot_id, ee_link_index, local_pos, q, dq, ddq_zeros)
    J = np.array(J_lin)
    
    # F = pinv(J.T) @ tau_ext
    force_vector = np.linalg.pinv(J.T) @ tau_ext
    
    return force_vector


