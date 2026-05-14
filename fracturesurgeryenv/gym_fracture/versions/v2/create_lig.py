import pybullet as p
import numpy as np
import os

def create_ligament_urdf(num_links=5, link_length=0.01, radius=0.002, mass=0.01):
    """Generates a multi-link chain URDF string."""
    urdf_str = f"""<?xml version="1.0"?><robot name="ligament">"""
    
    for i in range(num_links):
        urdf_str += f"""
        <link name="link_{i}">
            <contact><lateral_friction value="1.0"/></contact>
            <inertial>
                <origin xyz="0 0 {link_length/2}" rpy="0 0 0"/>
                <mass value="{mass}"/>
                <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
            </inertial>
            <visual>
                <origin xyz="0 0 {link_length/2}" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <geometry><capsule radius="{radius}" length="{link_length}"/></geometry>
                <material name="red"><color rgba="0.8 0.2 0.2 1"/></material>
            </visual>
        </link>"""
        
        if i > 0:
            urdf_str += f"""
        <joint name="joint_{i}" type="spherical">
            <parent link="link_{i-1}"/>
            <child link="link_{i}"/>
            <origin xyz="0 0 {link_length}"/>
            <axis xyz="0 0 1"/>
        </joint>"""
            
    urdf_str += "</robot>"
    
    with open("ligament.urdf", "w") as f:
        f.write(urdf_str)
    return "ligament.urdf"