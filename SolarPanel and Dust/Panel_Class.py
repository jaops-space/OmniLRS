import omni.isaac.core.utils.mesh as mesh_utils
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.rotations as rotations_utils
from omni.physx import get_physx_scene_query_interface
import numpy as np
from Lunar_Dust import LunarDust


class Panel() :
    """
    Class responsible for keeping track of the battery status
    prim_path (string) : Path to the panel prim
    sun_path (string) : Path to the Sun
    max_storage (int) : Battery capacity (Wh)
    current_storage (int) : Actual charge of the battery (Wh) 
    power_consumption (int) : Power consumed by the rover (W)
    power_produced (int) : Power produced by the rover (W)
    dust_coverage (float) : Percentage of the panel surface covered by dust (between 0 and 1)
    """
    def __init__(self, panel_prim_path, sun_path = "/World/Sun", nb_cells = 100, max_storage = 1000, 
                 current_storage = 0, power_consumption = 10, power_produced = 0, dust_coverage = 0,
                 dust = False, timeline = None) :
        self.prim_path = panel_prim_path
        self.sun_path = sun_path
        self.length, self.width, self.height = prims_utils.get_prim_attribute_value(panel_prim_path, "xformOp:scale")
        self.panel_surface = self.length * self.width
        self.nb_cells = nb_cells
        self.cell_area = self.panel_surface / self.nb_cells
        self.max_storage = max_storage
        self.current_storage = current_storage
        self.power_consumption = power_consumption
        self.current_charge = round(100*current_storage/max_storage, 2)
        self.power_produced = power_produced

        self.dust = dust
        self.dust_coverage = dust_coverage
        if dust :
            self.lunar_dust = LunarDust()
            self.prev_simulation_time = 0
            self.timeline = timeline

    def update(self, stage, time_elapsed) :
        """
        Execute one simulation step for the solar panel (currently doesn't support moving the panel to follow the sun)

        Args :
            time_elapsed : time elapsed since last update
            stage : The world stage

        Returns :
            None
        """

        if self.dust :
            self.lunar_dust.update(self.timeline.get_current_time(), self.prim_path)
            self.dust_coverage = self.lunar_dust.coverage
        self.update_power_produced(stage)
        self.update_state(time_elapsed)
        
    
    def update_state(self, time_elapsed) :
        """
        Simulate battery evolution during time_elapsed

        Args :
            time_elapsed : time elapsed since last state update

        Returns :
            None
        """
        self.current_storage += (self.power_produced - self.power_consumption)* (time_elapsed / 3600)
        if self.current_storage > self.max_storage :
            self.current_storage = self.max_storage
        if self.current_storage < 0 :
            self.current_storage = 0
        self.current_charge = round(100*self.current_storage/self.max_storage, 2)
        if self.current_storage == 0 :
            print("Battery is empty")

    def update_power_produced(self, stage) :
        """
        Update the power produced by the solar pannel based on the stage state

        Args : 
            stage : The world stage

        Returns :
            None
        """
        face_vertex_indices = prims_utils.get_prim_attribute_value(self.prim_path, "faceVertexIndices")
        # Get mesh vertices
        mesh_prim = stage.GetPrimAtPath(self.prim_path)
        world_prim = stage.GetPrimAtPath("/World")
        points_world = mesh_utils.get_mesh_vertices_relative_to(mesh_prim, world_prim)

        # Get quad centers
        num_quads = len(face_vertex_indices) // 4
        quad_centers = np.array([np.mean(points_world[list(face_vertex_indices[i * 4 : (i + 1) * 4])], axis=0) for i in range(num_quads)])
        # In this case we know that the first 100 elements are the faces on the top of the pannel, be careful with other usd.
        quad_centers = quad_centers[:self.nb_cells] # Keep only faces on the top -> avoid unnecessary computations
        dot_products = self.compute_sun_to_mesh(quad_centers)
        
        power_generation = self.compute_power(dot_products)
        self.power_produced = power_generation

    def set_power_produced(self, power_produced) :
        """
        Manually change the power produced

        Args :
            power_consumed (float) : The power produced by the panel

        Returns :
            None
        """
        self.power_produced = power_produced

    def set_power_consumed(self, power_consumed) :
        """
        Manually change the power consumed

        Args :
            power_consumed (float) : The power consumed by the electrics

        Returns :
            None
        """
        self.power_consumed = power_consumed

    def set_panel_orientation_euler(self, angles, degrees = True) :
        """
        Set the panel orientation using Euler angles.

        Args:
            angles (tuple or list): The Euler angles (in radians or degrees) to set the orientation.
            degrees (bool, optional): If True, the angles are interpreted as degrees.
                                      If False, the angles are interpreted as radians.
    
        Returns:
            None
        """
        prims_utils.set_prim_attribute_value(
            self.prim_path, "xformOp:orient",
            rotations_utils.euler_angles_to_quat(angles, degrees=degrees),
        )
        
    def display_state(self) :
        """
        Display the current state of the battery
        """
        print(f"Battery is at {self.current_charge}%, actual production (W) : {self.power_produced}, actual consumption (W): {self.power_consumption} | current_storage (Wh) : {self.current_storage} ")

    def compute_sun_to_mesh(self, mesh_face_centers, output="dotproducts"):
        """
        Uses ray-tracing to compute the dot products between the normals of mesh face centers and the direction from the Sun to each face center.
    
        Parameters:
        mesh_face_centers (array-like): An array of 3D coordinates representing the centers of the mesh faces.
        path_sun (str): The path to the Sun object in the scene. Default is "/World/Sun".
        output (str): The type of output to return. Options are "dotproducts", "dotproducts_normals", or "angles_deg".
                      Default is "dotproducts".
    
        Returns:
        numpy.ndarray or tuple: Depending on the output parameter:
            - "dotproducts": Returns an array of dot products.
            - "dotproducts_normals": Returns a tuple containing an array of dot products and an array of normals.
            - "angles_deg": Returns an array of angles in degrees.
    
        Notes:
        - If a mesh face center is in the shadow of an obstruction, the corresponding dot product and normal are set to NaN.
        - If no hit is detected for a mesh face center, the corresponding dot product and normal are set to NaN.
        - The function prints a message if no hit is detected for a mesh face center.
        """
        sun_coord = prims_utils.get_prim_attribute_value(self.sun_path, "xformOp:translate")
    
        dot_products = np.zeros(len(mesh_face_centers))
        normals = np.zeros((len(mesh_face_centers), 3))
        hit_distances = []  # for debug
    
        for i, center in enumerate(mesh_face_centers):
            vector_to_cell = center - sun_coord
            distance = np.linalg.norm(vector_to_cell)
            direction = vector_to_cell / distance
    
            # Cast a ray from the Sun to the cell center
            hit_info = get_physx_scene_query_interface().raycast_closest(
                sun_coord, direction, distance + 1
            )  # increase max distance to avoid ray not reaching the surface
    
            if hit_info["hit"]:
                hit_distance = np.linalg.norm(hit_info["position"] - center)
                hit_distances.append(hit_distance)
                if (
                    hit_distance < 0.001
                ):  # consider that there is no obstruction between the sun and the cell center
                    normals[i] = np.array(hit_info["normal"])
                    dot_products[i] = np.dot(
                        normals[i], -1 * direction
                    )  # direction is sun to center, so need to flip it
                else:  # cell is in shadow of obstruction
                    dot_products[i] = np.nan
                    normals[i] = np.nan
            else:
                print(f"no hit, angle for cell center at {center} is undefined")
                dot_products[i] = np.nan  # No hit, angle is undefined
                normals[i] = np.nan
    
        if output == "dotproducts":
            return dot_products
        if output == "dotproducts_normals":
            return dot_products, normals
        elif output == "angles_deg":
            sun_angles = np.arccos(dot_products) * (180 / np.pi)  # Convert to degrees
            return sun_angles
        else:
            print(
                "possible outputs are only 'dotproducts' , 'dotproducts_normals' or 'angles_deg'"
            )
            return
            
    def compute_power(self, cos_theta) :
        """
        Compute the power produced by the solar panel

        Args : 
            cos_theta : array of the cosine of the incidence angle for each solar cell

        Returns :
            Power produced by the solar panel
        """
        # P = A x I x n x cos(theta) x covering
        solar_intensity = 1376 # (W/m2)
        solar_cell_efficiency = 0.3 
        cos_theta[cos_theta < 0] = 0
        power_produced = self.cell_area * solar_intensity * solar_cell_efficiency * cos_theta * (1 - self.dust_coverage)
        return np.nansum(power_produced[:self.nb_cells])