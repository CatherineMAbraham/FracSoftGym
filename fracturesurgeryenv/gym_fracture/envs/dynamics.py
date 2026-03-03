import pybullet as p

# def fix_robot_physics(self, robot_id):
#         num_joints = p.getNumJoints(robot_id)
#         for i in range(num_joints):
#             mass = p.getDynamicsInfo(robot_id, i)[0]
            
#             # Soften the contact constraints
#             # contactStiffness: lower = softer 'squish'
#             # contactDamping: higher = less 'bounce'
#             p.changeDynamics(robot_id, i, 
#                             contactStiffness=3000, 
#                             contactDamping=100,
#                             lateralFriction=0.1,
#                             jointDamping=0.01)
            
#             if mass > 0:
#                 new_inertia = [mass * 0.001, mass * 0.001, mass * 0.001]
#                 p.changeDynamics(robot_id, i, localInertiaDiagonal=new_inertia)
#             else:
#                 p.changeDynamics(robot_id, i, mass=0.001, localInertiaDiagonal=[1e-6]*3)
         ## check and tune this 
            # Also soften the object you are moving!
            # p.changeDynamics(self.objectUid, -1, 
            #                 contactStiffness=3000, 
            #                 contactDamping=100,
            #                 lateralFriction=1)
            
def change_leg_dynamics(self):
    p.changeDynamics(self.leg, 0, 
                     mass=0, 
                     contactStiffness=1000000, 
                     contactDamping=1000, 
                     lateralFriction=0.5,
                     linearDamping=0.01,
                     angularDamping=0.01)
    p.setPhysicsEngineParameter(contactSlop=0.001)
 
def change_foot_dynamics(self):
    
    p.changeDynamics(self.objectUid, -1, 
                     mass=0.276, 
                     lateralFriction=2, # Lower this! 5.0 is causing the 50N spikes
                     contactStiffness=1000000, 
                     contactDamping=1000,
                     collisionMargin=0.01)
     
def change_robot_dynamics(self):
    for i in [9, 10]:
        p.changeDynamics(self.pandaUid, i, 
                         lateralFriction=2.0, # Enough to grip, but not 'glued'
                         contactStiffness=1000000, 
                         contactDamping=1000,
                         collisionMargin=0.01)
    #p.changeDynamics(self.pandaUid, 9, jointLowerLimit=0.00, jointUpperLimit=0.004,contactStiffness=2000, 
                    # contactDamping=100)
    #p.changeDynamics(self.pandaUid, 10, jointLowerLimit=0.0, jointUpperLimit=0.0042,contactStiffness=2000, 
                     #contactDamping=100)

       
        
        






#