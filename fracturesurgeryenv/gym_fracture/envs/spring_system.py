import pybullet as p
import numpy as np

def vec(x): return np.array(x, dtype=float)

class ElasticBand:
    def __init__(self, bodyA, linkA, bodyB, linkB,
                 young_modulus, area, rest_length,
                 damping_ratio=1.0):
        self.bodyA = bodyA
        self.linkA = linkA
        self.bodyB = bodyB
        self.linkB = linkB

        self.E = young_modulus
        self.A = area
        self.L0 = rest_length
        self.k = (self.E * self.A) / self.L0

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
        if L < 1e-9:
            return
        d = delta / L

        # === Spring Force ===
        spring_force = self.k * (L - self.L0)
        if spring_force < 0:
            spring_force = 0.0 # No compression force

        # === Damping Force ===
        mA = p.getDynamicsInfo(self.bodyA, self.linkA)[0]
        mB = p.getDynamicsInfo(self.bodyB, self.linkB)[0]
        m_eff = (mA * mB) / (mA + mB)

        c = self.damping_ratio * 2.0 * np.sqrt(self.k * m_eff)

        rel_vel = np.dot((velB - velA), d)
        damping_force = c * rel_vel

        # Final scalar force
        F = spring_force + damping_force
        F_vec = F * d

        # Save for robot measurement
        self.last_force_vector = F_vec.copy()

        # Apply equal and opposite forces
        p.applyExternalForce(self.bodyA, self.linkA,
                             forceObj=(-F_vec).tolist(),
                             posObj=posA.tolist(),
                             flags=p.WORLD_FRAME)

        p.applyExternalForce(self.bodyB, self.linkB,
                             forceObj=F_vec.tolist(),
                             posObj=posB.tolist(),
                             flags=p.WORLD_FRAME)

    def get_force(self):
        """Return current tension force vector (world frame)."""
        return self.last_force_vector.copy()
    
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
    print("Estimated EE force:", resultant_force)
    return Ft  # world-frame force vector (approx)

