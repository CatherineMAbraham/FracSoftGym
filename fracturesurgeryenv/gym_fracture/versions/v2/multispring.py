import pybullet as p
import numpy as np
import math

def create_ligament_chain(start_pos, end_pos, num_segments, total_mass):

    direction = np.array(end_pos) - np.array(start_pos)
    L0 = np.linalg.norm(direction)
    direction /= L0

    segment_length = L0 / num_segments
    segment_mass = total_mass / num_segments

    radius = 0.005

    col = p.createCollisionShape(p.GEOM_CAPSULE,
                                 radius=radius,
                                 height=segment_length)

    vis = p.createVisualShape(p.GEOM_CAPSULE,
                              radius=radius,
                              length=segment_length,
                              rgbaColor=[0,1,0,1])

    bodies = []

    for i in range(num_segments):
        pos = start_pos + direction * (i + 0.5) * segment_length

        body = p.createMultiBody(
            baseMass=segment_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos.tolist()
        )

        bodies.append(body)

    return bodies, L0
def apply_axial_springs(bodies, k_total, L0, damping):

    N = len(bodies)
    k_segment = k_total * N
    L_segment = L0 / N

    for i in range(N - 1):

        posA, ornA = p.getBasePositionAndOrientation(bodies[i])
        posB, ornB = p.getBasePositionAndOrientation(bodies[i+1])

        velA, _ = p.getBaseVelocity(bodies[i])
        velB, _ = p.getBaseVelocity(bodies[i+1])

        posA = np.array(posA)
        posB = np.array(posB)

        delta = posB - posA
        dist = np.linalg.norm(delta)

        if dist < 1e-6:
            continue

        direction = delta / dist
        stretch = dist - L_segment

        Fs = k_segment * stretch
        rel_vel = np.dot(np.array(velB) - np.array(velA), direction)
        Fd = damping * rel_vel

        F = (Fs + Fd) * direction

        p.applyExternalForce(bodies[i], -1, F.tolist(), posA.tolist(), p.WORLD_FRAME)
        p.applyExternalForce(bodies[i+1], -1, (-F).tolist(), posB.tolist(), p.WORLD_FRAME)