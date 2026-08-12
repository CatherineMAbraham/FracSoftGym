import pybullet as p
import os

patient = 110
## File to make it easy to load new patient models into the environment.

def load_patient_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    leg_path = os.path.join(current_dir, f"Assets/Patient{patient}/proximal.urdf")
    foot_path = os.path.join(current_dir, f"Assets/Patient{patient}/distal.urdf")
    return leg_path, foot_path

