import pybullet as p

def change_leg_dynamics(env):
    p.changeDynamics(env.leg, 0, 
                     mass=0, 
                     contactStiffness=5000, 
                     contactDamping=1000, 
                     lateralFriction=0.5,
                     linearDamping=0.01,
                     angularDamping=0.01,
                     collisionMargin=0.0001)
    p.setPhysicsEngineParameter(contactSlop=0.0005)
    #p.setCollisionFilterGroupMask(env.leg, -1, collisionFilterGroup=0, collisionFilterMask=0) 
 
def change_foot_dynamics(env):
    
    p.changeDynamics(env.foot, -1, 
                     mass=0.001, 
                     lateralFriction=2, # Lower this! 5.0 is causing the 50N spikes
                     contactStiffness=5000, 
                     contactDamping=100,
                     collisionMargin=0.001)
    p.changeDynamics(env.foot, 1, 
                     mass=0.276, 
                     lateralFriction=0.5, # Lower this! 5.0 is causing the 50N spikes
                     contactStiffness=3000, 
                     contactDamping=300,
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