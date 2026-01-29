import numpy as np
import pybullet as p
import math


class new_band:
    def __init__(self):
        self.band_id = None
        self.force_text_id = None

    def apply_elastic(self, bodyA, bodyB, YOUNGS_MODULUS, rest_length, max_length, max_force,VISCOSITY):

        posA, _ = p.getBasePositionAndOrientation(bodyA)
        posB, _ = p.getBasePositionAndOrientation(bodyB)

        velA, _ = p.getBaseVelocity(bodyA)
        velB, _ = p.getBaseVelocity(bodyB)

        posA = np.array(posA)
        posB = np.array(posB)
        velA = np.array(velA)
        velB = np.array(velB)

        delta = posB - posA
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return

        direction = delta / dist

        # -------------------------------
        # Geometry
        # -------------------------------
        radius_mm = 5.0  # Band radius in mm
        radius = radius_mm / 1000.0
        area = math.pi * radius ** 2

        # Spring stiffness (Young's modulus)
        k = (YOUNGS_MODULUS * area) / rest_length

        # Internal resistance (viscoelastic)
        c = (VISCOSITY * area) / rest_length

        # -------------------------------
        # Forces
        # -------------------------------
        stretch = max(0.0, min(dist - rest_length, max_length - rest_length))
        Fs = k * stretch

        stretch_rate = np.dot((velB - velA), direction)
        Fv = c * stretch_rate

        force_mag = Fs + Fv
        force_mag = np.clip(force_mag, 0, max_force)

        force = force_mag * direction

        p.applyExternalForce(
            bodyA, -1, force, posA, p.WORLD_FRAME
        )

        # -------------------------------
        # Visual band
        # -------------------------------
        color = [1, 0, 0] if stretch > 0 else [0, 1, 0]
        width = max(1, int(radius_mm))

        if self.band_id is None:
            self.band_id = p.addUserDebugLine(posA, posB, color, width)
        else:
            p.addUserDebugLine(
                posA, posB, color, width,
                replaceItemUniqueId=self.band_id
            )

        # -------------------------------
        # Force readout
        # -------------------------------
        text = f"Band force: {force_mag:.1f} N"
        if self.force_text_id is None:
            self.force_text_id = p.addUserDebugText(
                text, posA + [0, 0, 0.15], textSize=1.2
            )
        else:
            p.addUserDebugText(
                text, posA + [0, 0, 0.15],
                textSize=1.2,
                replaceItemUniqueId=self.force_text_id)