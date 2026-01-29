import pybullet as p
import numpy as np
from gym_fracture.envs import create_lig

class MultiLinkLigament:
    def __init__(self, bodyA, linkA, bodyB, linkB, E=200e6, Area=1e-6):
        # 1. Generate and Load URDF
        urdf_path = create_lig.create_ligament_urdf(num_links=5, link_length=0.01)
        self.id = p.loadURDF(urdf_path, flags=p.URDF_USE_SELF_COLLISION)
        
        self.bodyA = bodyA # e.g., the Leg
        self.bodyB = bodyB # e.g., the Pin/Foot
        self.num_joints = p.getNumJoints(self.id)
        
        # 2. Map Young's Modulus to Stiffness
        # E (Pa), Area (m^2), L (m)
        self.E = E
        self.Area = Area
        
        # 3. Create Anchors (Point-to-Point Constraints)
        # Anchor start of chain to Leg
        p.createConstraint(bodyA, linkA, self.id, 0, p.JOINT_POINT2POINT, 
                           [0,0,0], [0,0,0], [0,0,0])
        
        # Anchor end of chain to Pin
        p.createConstraint(bodyB, linkB, self.id, self.num_joints-1, p.JOINT_POINT2POINT, 
                           [0,0,0], [0,0,0], [0,0,0.00])

    def update_stiffness(self, total_length):
        """Controls elasticity by adjusting joint motor force based on E."""
        # Calculate axial stiffness k = (E * A) / L
        k_total = (self.E * self.Area) / total_length
        
        # For joints in series, each joint must be stiffer to maintain k_total
        # We add a small damping to prevent high-frequency oscillations
        joint_stiffness = k_total * self.num_joints
        damping = joint_stiffness * 0.05 
        
        for i in range(self.num_joints):
            # We use POSITION_CONTROL with target=0 to act as a restorative spring
            p.setJointMotorControlMultiDof(
                self.id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1], # Neutral quaternion
                targetVelocity=[0, 0, 0],
                force=[joint_stiffness]*3,
                positionGain=0.1,
                velocityGain=damping
            )

    def get_tension(self):
        """Calculates tension force based on joint reaction forces."""
        # Extracting reaction force from the first joint anchor
        reaction = p.getJointState(self.id, 0)[2] # [6 elements: force x,y,z, torque x,y,z]
        return np.linalg.norm(reaction[:3])