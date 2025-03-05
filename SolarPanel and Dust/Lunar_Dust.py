import numpy as np
from scipy.interpolate import PchipInterpolator
<<<<<<< HEAD
import typing
import matplotlib.pyplot as plt
import carb
from pprint import pprint

from pxr import Usd, UsdGeom, Gf
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCone
from omni.timeline import get_timeline_interface
from omni.isaac.core.utils.stage import get_current_stage
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.mesh as mesh_utils
from omni.usd import get_prim_at_path
from omni.physx import get_physx_scene_query_interface  # for raycasting


class LunarDust():

=======
import omni.isaac.core.utils.prims as prims_utils
import typing
from pxr import Usd, UsdGeom, Gf
import matplotlib.pyplot as plt
from omni.isaac.core import World
from omni.physx import get_physx_scene_query_interface
import carb


class LunarDust:
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
    def __init__(self, world):
        """
        Initializes the lunar dust accumulation simulation.
        
        :param surface_area: Surface area exposed to dust (m^2)
        :param time_step: Time step for accumulation updates (s)
        """
<<<<<<< HEAD
        # Setup world
        self.world = world
        
        # Lunar Dust behaviour options
        self.limit_deposition = True
        
        
        # Fixed parameters for dust behavior
        self.dust_density = 1500  # (kg/m^3) - 
=======
        # Lunar Dust behaviour options
        self.limit_deposition = False #True

        # Setup world
        self.world = world
        
        # Fixed parameters for dust behavior
        self.dust_density = 1500  # (kg/m^3) -
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        self.dust_diameter = 40e-6  # (m) - Approximate lunar dust particle diameter
        self.altitude_to_deposition_rate = 500
        
        # Altitude-to-accumulation rate interpolation data points
        alt_points = np.array([0, 0.6, 1, 1.9, 100])  # Altitudes (m)
        taux_points = np.array([10000, 500, 100, 20, 0.001])*1e-5  # Corresponding accumulation rates
<<<<<<< HEAD
        # at altitude 0 the accumulation rate has been set as 10000. there's no data on altitudes lower than 0.6 m, 
=======
        # at altitude 0 the accumulation rate has been set as 10000. there's no data on altitudes lower than 0.6 m,
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        # this is only to start the interpolation at zero
        
        # Create an interpolation function for accumulation rate based on altitude
        self.pchip_interp = PchipInterpolator(alt_points, taux_points)
        
<<<<<<< HEAD
       # Create an interpolation function for accumulation limit (by dust diameter) in a year based on data 
        if self.limit_deposition:
            print("DUST COVERAGE LIMITED BY DUST SIZE")
            dust_radius_points = np.array([10, 50, 100]) 
=======
       # Create an interpolation function for accumulation limit (by dust diameter) in a year based on data
        if self.limit_deposition:
            print("DUST COVERAGE LIMITED BY DUST SIZE")
            dust_radius_points = np.array([10, 50, 100])
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
            limit_points = np.array([15, 3, 1.5])
            self.pchip_limit = PchipInterpolator(dust_radius_points, limit_points)
            self.max_coverage = self.pchip_limit(self.dust_diameter*1e6/2) / 100
            print("Max surface coverage for ", self.dust_diameter, " m dust: ", self.max_coverage)

        else:
            self.max_coverage = 1

<<<<<<< HEAD
        # Speed-to-accumulation rate interpolation data points 
=======
        # Speed-to-accumulation rate interpolation data points
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        speed_points = np.array([0.5, 1, 3, 5, 7])
        altitude_reached_points = np.array([0.015, 0.024, 0.32, 1.32, 3.2])
        
        # Create an interpolation function for accumulation rate based on speed
        self.pchip_altitude_reached = PchipInterpolator(speed_points, altitude_reached_points)
        
        # Variables for dust accumulation tracking
        self.coverage = 0.0  # (%) Covered surface
        self.taux = 0.0  # Current accumulation rate
<<<<<<< HEAD
        self.total_time_passed = 0 
        self.prev_simulation_time = 0
                
=======
        self.total_time_passed = 0
        self.prev_simulation_time = 0

>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        # Data for color changing
        self.retrieved_color = False
        self.final_color = Gf.Vec3f((0.85, 0.83, 0.8))
        self.final_roughness = 1.0
<<<<<<< HEAD
        
        
        self.reduction_altitude = 0.0 ###### for testing purposes

        
    def plot(self) :
    
=======

    def plot(self) :
        
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        # Print Plots (choose one)
        plt.figure()
        plt.plot(np.linspace(0,1.9, 100), self.pchip_interp(np.linspace(0,1.9, 100)) * 1e5)
        plt.grid(True)
        plt.xlabel("Altitude")
        plt.ylabel("Deposition rate")
        plt.title("Deposition rate by altitude")
        plt.show()
<<<<<<< HEAD
    
=======

>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        plt.figure()
        plt.plot(np.linspace(0, 7, 100), self.pchip_altitude_reached(np.linspace(0, 7, 100)))
        plt.grid(True)
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Altitude reached")
        plt.title("Deposition rate by altitude")
        plt.show()
<<<<<<< HEAD
    
=======
        
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        if self.limit_deposition:
            plt.figure()
            plt.plot(np.linspace(10,100, 100), self.pchip_limit(np.linspace(10,100, 100)))
            plt.grid(True)
            plt.xlabel("Dust diameter")
            plt.ylabel("Max possible covered surface (%)")
            plt.title("Max covered surface by dust diameter")
            plt.show()
<<<<<<< HEAD

        

=======
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
    
    def update(self, simulation_time: float, prim: Usd.Prim):
        """
        Updates the dust accumulation and surface coverage based on the current altitude.
        
        :param time_elapsed: Time elapsed since the last update (s)
        :param prim: The USD Prim representing the object in the simulation
        """
        
        time_elapsed = (simulation_time - self.prev_simulation_time)
        self.total_time_passed += time_elapsed #/ (86400*365)
        self.prev_simulation_time = simulation_time

        if self.total_time_passed < 3*60*60:
            print("Time Passed: ", round(self.total_time_passed / 60, 2), " minutes")
        elif self.total_time_passed < 86400*3:
            print("Time Passed: ", round(self.total_time_passed / (60*60), 2), " hours")
        elif self.total_time_passed < 86400*365/2:
            print("Time Passed: ", round(self.total_time_passed / 86400, 2), " days")
<<<<<<< HEAD
        else: 
            print("Time Passed: ", round(self.total_time_passed / (86400*365), 2), " years")


        ################## WARNING
        # The altitude is calculated with the distance from the ground where the ground coordinates come from a raycast from under it
        # However, the environment of simulation is very large and to visualize better the prims, they have to be of an unrealistic size:
        # this means that the lunar dust accumulation is not at all representative of its effects. We decided, for the sake of this test, to
        # reduce manually the altitude calculations in order for it to provide a good value results in the tests. The testing prim appears 
        # to be at altitude 28 m, so we will reduce it 
        altitude = self.get_surface_altitude(prim) - self.reduction_altitude ################### WARNING
=======
        else:
            print("Time Passed: ", round(self.total_time_passed / (86400*365), 2), " years")
            
        altitude = self.get_surface_altitude(prim)
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        print("Altitude: ", round(altitude,2), " m")
        
        self.taux = self.compute_dust_accumulation_rate(altitude) # + compute_dust_accumulation_rate_by_movement(speed) ## WORK IN PROGRESS
        print("Taux de deposition: ", round(self.taux*1e5, 2), " micrograms/cm^2*y")

        self.update_coverage(time_elapsed)
        print("Covered surface by dust: ", self.coverage*100, "%")
        print("")

            
    def get_surface_altitude(self, prim: Usd.Prim) -> float:
        """
        Retrieves the altitude of a given prim by extracting its world position.

        :param prim: The USD Prim to retrieve the position for.
        :return: Altitude (z-coordinate in meters)
        """
        xform = UsdGeom.Xformable(prim)
        time = Usd.TimeCode.Default() # The time at which we compute the bounding box
        world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
        translation: Gf.Vec3d = world_transform.ExtractTranslation()
        #rotation: Gf.Rotation = world_transform.ExtractRotation()
        #scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))  

        # Perform raycast to get the real altitude
        # raycast starts from way underground under the surface and goes in the Z+ direction in order to hit the ground under the surface
        self.world.step(render=True)
        

        direction = (0,0,1)
<<<<<<< HEAD
        origin = carb.Float3(translation[0], translation[1], -100) 
        max_distance = 1000000000 
=======
        origin = carb.Float3(translation[0], translation[1], -100)
        max_distance = 1000000000
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        
        
        hit_info = get_physx_scene_query_interface().raycast_closest(origin, direction, max_distance+1)
        self.world.step(render=True)
        if hit_info["hit"]:
            ground_coordinates = hit_info["position"]
        else:
            print("object has no ground under it")
            ground_coordinates = (carb.Float3(translation[0], translation[1], -100000))

<<<<<<< HEAD
        altitude = translation[2]-ground_coordinates[2] 
=======
        altitude = translation[2]-ground_coordinates[2]
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
        return altitude

    def compute_dust_accumulation_rate(self, altitude: float) -> float:
        """
        Computes the dust accumulation rate based on the current altitude using interpolation.

        :param altitude: Current altitude (m)
        :return: Accumulation rate (kg/m^2*y)
        """
        return self.pchip_interp(altitude)
    
    def compute_dust_accumulation_rate_by_movement(self, speed: float) -> float: ### WORK IN PROGRESS
        """
        Computes the dust accumulation rate based on the current speed using interpolation.

        :param altitude: Current speed (m/s)
        :return: Accumulation rate (kg/m^2*y)
        """
        return self.pchip_altitude_reached(speed) * self.altitude_to_deposition_rate
        
    def update_coverage(self, time_elapsed: float):
        """
        Updates the surface coverage percentage based on dust accumulation.
        
        :param time_elapsed: Time elapsed since the last update (s)
        """
        # Ensure taux and other parameters are defined
        if self.taux <= 0:
            return

        # Compute the fraction of the surface covered over time
        coverage_increase = (3/2) * (self.taux / ( self.dust_diameter * self.dust_density)) * \
                            (time_elapsed / (365 * 24 * 3600))  # Convert to years
        
        # Update total coverage
        self.coverage = min(self.max_coverage, self.coverage + coverage_increase)  # Ensure it doesn't exceed 100%


    def reset_accumulation(self):
        """
        Resets the dust accumulation and coverage to zero.
        """
        self.accumulated_dust = 0.0
        self.coverage = 0.0

    def update_color(self, shader_path):
        if self.retrieved_color is False:
            self.original_color = prims_utils.get_prim_attribute_value(shader_path, 'inputs:diffuseColor')
            self.original_roughness = prims_utils.get_prim_attribute_value(shader_path, 'inputs:roughness')
<<<<<<< HEAD
=======
            #print("Original color: ", self.original_color)
            #print("Original roughness: ", self.original_roughness)
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
            self.retrieved_color = True
            
        new_color = self.original_color + self.coverage * (self.final_color - self.original_color)
        #print("New color: ", new_color)
        new_color = Gf.Vec3f(float(new_color[0]), float(new_color[1]), float(new_color[2]))

        new_roughness = self.original_roughness + self.coverage * (self.final_roughness - self.original_roughness)
        #print("New roughness: ", new_roughness)
        
        prims_utils.set_prim_attribute_value(shader_path, 'inputs:diffuseColor', new_color)
        prims_utils.set_prim_attribute_value(shader_path, 'inputs:roughness', new_roughness)

        self.world.render()

<<<<<<< HEAD
    def set_color(self, shader_path, new_color):
        new_color = Gf.Vec3f(float(new_color[0]), float(new_color[1]), float(new_color[2]))
        prims_utils.set_prim_attribute_value(shader_path, 'inputs:diffuseColor', new_color)
=======
>>>>>>> 316c952bd8f514ddccb7c0caeabf19cde2f42625
