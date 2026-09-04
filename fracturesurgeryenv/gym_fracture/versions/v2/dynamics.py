import pybullet as p

def change_leg_dynamics(env):
    p.changeDynamics(env.leg, 0, 
                     mass=0, 
                     contactStiffness=500, 
                     contactDamping=150, 
                     lateralFriction=0.5,
                     linearDamping=0.01,
                     angularDamping=0.01,
                     collisionMargin=0.0001)
    p.setPhysicsEngineParameter(contactSlop=0.0005)
    #p.setCollisionFilterGroupMask(env.leg, -1, collisionFilterGroup=0, collisionFilterMask=0) 
 
def change_foot_dynamics(env):
    if env.randomise_foot_dynamics:
        mass = env.np_random.uniform(0.01 * 0.8, 0.01 * 1.2)if env.patient ==132 else env.np_random.uniform(0.276 * 0.8, 0.276 * 1.2)
        
        # Friction DR
        joint_friction = env.np_random.uniform(0.3, 1.0)
        sole_friction = env.np_random.uniform(0.5, 1.2)
        spin_friction = env.np_random.uniform(0.001, 0.02)
        
        # Contact Compliance DR (±20% variation)
        stiffness_link1 = env.np_random.uniform(4000, 6000)
        stiffness_joint = env.np_random.uniform(240, 360)
        damping = env.np_random.uniform(80, 120)
        restitution = env.np_random.uniform(0.0, 0.15)
    else:
        # If not randomizing, use default values
        mass = 0.0276 if env.patient ==126 or env.patient == 132 else 0.276
        sole_friction, joint_friction, spin_friction = 1.0, 0.5, 0.005
        stiffness_link1, stiffness_joint, damping = 5000, 300, 100
        restitution = 0.0

    # Toe / Sole segment
    p.changeDynamics(env.foot, 1, 
                              mass=0.001, 
                              lateralFriction=2, # Lower this! 5.0 is causing the 50N spikes
                              contactStiffness=5000, 
                              contactDamping=100,
                              collisionMargin=0.1) # Standardized from 0.1 to avoid phantom collisions

    # Main Foot Joint segment
    p.changeDynamics(env.foot, env.footjoint, 
                     mass=mass, 
                     lateralFriction=joint_friction,
                     spinningFriction=spin_friction,
                     contactStiffness=stiffness_joint, 
                     contactDamping=damping,
                     restitution=restitution,
                     collisionMargin=0.0001)
   # print(p.getLinkState(env.foot, 1))
    
     
def change_robot_dynamics(env):
    for i in [9, 10]:
        p.changeDynamics(env.pandaUid, i, 
                         lateralFriction=5.0, # Enough to grip, but not 'glued'
                         contactStiffness=5000, 
                         contactDamping=100,
                         collisionMargin=0.0001)

def change_ligament_dynamics(name):
    p.changeDynamics(name, -1, mass=0.1, linearDamping=0.7, angularDamping=0.7)
    p.setPhysicsEngineParameter(contactERP=0.4,useSplitImpulse=1,
                                splitImpulsePenetrationThreshold=0.001)#, CFM=0.0011)#, cfm=0.5)#, 
    p.setCollisionFilterGroupMask(name, -1, collisionFilterGroup=0, collisionFilterMask=0) # Disable collisions for soft body to prevent explosion during tuning
   
       
        
        






#