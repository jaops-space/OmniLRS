import numpy as np
from math import pi as PI
from scipy.constants import c as C
import carb
from pxr import UsdGeom, Gf, Vt
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCone, DynamicCylinder
from omni.physx import get_physx_scene_query_interface  # for raycasting e.g raycast_closest()


#..........G = the gain of the antenna (its maximum gain)
#..........e = its efficiency
# waveLength = the wavelength for which it emits/receives
#.......mode = 1 if emitter, 0 if receiver
# theta, phi = angles defining the direction it emits/receives
#........pow = receivable power
class Antenna(DynamicCylinder):
    def __init__(self, prim_path, name, waveLength, G, e, position=None, orientation=None, scale=None, color=None, radius=None, height=None, mode=0, power=0):
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation, scale=scale, color=color, radius=radius, height=height)
        self.waveLength = waveLength
        self.G = G
        self.e = e
        self.L = waveLength/2
        self.frequency = C/self.waveLength
        self.directivity = G/e
        self.mode = mode
        self.theta = 0
        self.phi = 0
        self.power = 0
    
    def getLength(self):
        return self.L
    
    def getGain(self):
        return self.G
    
    def getFrequency(self):
        return self.frequency
    
    def getDirectivity(self):
        return self.directivity
    
    def getMinimumPower(self):
        return self.power
    
    def setMode(self, mode):
        if mode == 0 or mode == 1:
            self.mode = mode
        else:
            raise ValueError("The mode must be 0 (emitter) or 1 (receiver).")
    
    def setDirection(self, theta, phi):
        if 0 > theta or theta > PI:
            raise ValueError("Theta must have a value between 0 and pi.")
        if 0 > phi or phi > 2*PI:
            raise ValueError("Phi must have a value between 0 and 2*pi.")
        self.theta = theta
        self.phi = phi
        
    def setPower(self, powerValue):
        if powerValue < 0:
            raise ValueError("Antenna receivable power must be larger than 0")
        self.power = powerValue
        
    def performRaycast(self, target_antenna, stage, distance_increment):
        """
        Performs a raycast from the current antenna (self) towards another antenna (target_antenna).

        Args:
            target_antenna (Antenna): The target antenna.
            stage: The current USD scene.
            distance_increment (float): Distance to shift the starting position of the ray.

        Returns:
            tuple: (Path of the hit object, distance traveled by the ray)
        """
        # Move the ray's origin slightly forward
        object1_position = self.moveOriginRaycast(target_antenna, distance_increment)
        object2_position, _ = target_antenna.get_world_pose()
        
        # Calculate the normalized direction
        direction = np.array(object2_position) - np.array(object1_position)
        distance = np.linalg.norm(direction)
        if distance == 0:
            return None, 10000.0  # Edge case where antennas overlap

        direction_normalized = direction / distance
        rayDir = carb.Float3(*direction_normalized)

        # Perform the raycast
        hit = get_physx_scene_query_interface().raycast_closest(object1_position, rayDir, distance)

        if hit["hit"]:
            usdGeom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
            return usdGeom.GetPath().pathString, hit["distance"]

        return None, 10000.0  # No object was hit

    def moveOriginRaycast(self, target_antenna, increment):
        """
        Calculates the new ray origin shifted by a given increment.

        Args:
            target_antenna (Antenna): The receiving antenna.
            increment (float): Distance to shift the origin.

        Returns:
            tuple: New position of the ray's origin.
        """
        object1_position, _ = self.get_world_pose()
        object2_position, _ = target_antenna.get_world_pose()

        direction = np.array(object2_position) - np.array(object1_position)
        direction_normalized = direction / np.linalg.norm(direction)
        new_origin = np.array(object1_position) + direction_normalized * increment

        return tuple(new_origin)

    def checkRaycastCoherence(self, output_ray, target_path):
        """
        Verifies if the raycast hit the expected target.

        Args:
            output_ray (tuple): Result of `performRaycast`, contains the hit path.
            target_path (str): Expected path of the target object.

        Returns:
            bool: True if the ray hit the expected object, False otherwise.
        """
        path1 = output_ray[0]

        if path1 is None or target_path is None:
            return False
        
        object_name1 = path1.split('/')[-1]
        object_name2 = target_path.split('/')[-1]

        return object_name1 == object_name2

    def displayRaycastCoherence(self, output_ray, target_path):
        """
        Prints whether the signal was received or not based on raycast coherence.

        Args:
            output_ray (tuple): Result of `performRaycast`, contains the hit path.
            target_path (str): Expected path of the target object.
        """
        if self.checkRaycastCoherence(output_ray, target_path):
            print("The signal has been received.")
        else:
            print("The signal has not been received.")

    def power_losses(self, distance):
        """
        Computes signal power loss over a given distance.

        Args:
            distance (float): Distance traveled by the signal.

        Returns:
            float: Power loss factor.
        """
        if distance == 0:
            return 0
        return 1 / (distance ** 2)

    def checkPowerFeasibility(self, distance, minimum_power):
        """
        Checks if the received power is above the required threshold.

        Args:
            distance (float): Distance traveled by the signal.
            minimum_power (float): Minimum required power.

        Returns:
            bool: True if the signal is strong enough, False otherwise.
        """
        return self.power_losses(distance) >= minimum_power

    def displayPowerFeasibility(self, output_ray, target_antenna):
        """
        Prints whether the signal is strong enough to be detected, only if the raycast was successful.
        
        Args:
            output_ray (tuple): Result of `performRaycast`, containing the hit path and distance.
            target_path (str): Path of the target antenna.
            target_antenna_minPower (float): Minimum required power.
        """
        if not self.checkRaycastCoherence(output_ray, target_antenna.prim_path):
            print("The signal has not reached the target antenna.")
            return
    
        distance = output_ray[1]
        if self.checkPowerFeasibility(output_ray[1], target_antenna.getMinimumPower()):
            print("The signal is detectable by the receiver.")
        else:
            print("The signal is not detectable by the receiver.")

