from scipy.constants import c as C
from math import pi as PI
from omni.physx import get_physx_scene_query_interface  # for raycasting e.g raycast_closest()

class Antenna(DynamicCylinder):
    def __init__(self, prim_path, name, waveLength, G, e, position=None, orientation=None, scale=None, color=None, radius=None, height=None, mode=0):
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
    
    def getLength(self):
        return self.L
    
    def getGain(self):
        return self.G
    
    def getFrequency(self):
        return self.frequency
    
    def getDirectivity(self):
        return self.directivity
    
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


from pxr import UsdGeom, Gf, Vt

def perform_raycast(stage, object1, object2):

    object1_position,_ = object1.get_world_pose()
    object2_position,_ = object2.get_world_pose()
    
    # Calculate direction from object1 to object2
    direction = (object2_position[0] - object1_position[0], 
                 object2_position[1] - object1_position[1], 
                 object2_position[2] - object1_position[2])
    
    # Normalize the direction vector
    distance = np.linalg.norm(direction)
    direction_normalized = tuple(d / distance for d in direction)

    # Origin of the ray (from object1)
    origin = carb.Float3(object1_position[0], object1_position[1], object1_position[2])
    rayDir = carb.Float3(direction_normalized[0], direction_normalized[1], direction_normalized[2])
    
    # Perform the raycast and check for a hit
    hit = get_physx_scene_query_interface().raycast_closest(origin, rayDir, distance)

    if hit["hit"]:
        usdGeom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
        hitColor = Vt.Vec3fArray([Gf.Vec3f(1., 1., 0.0)])
        usdGeom.GetDisplayColorAttr().Set(hitColor)
        world.render()
        
        objects_distance = hit["distance"]
        return usdGeom.GetPath().pathString, objects_distance

    # No hit, return None and a large distance value
    return None, 10000.0

def move_origin_raycast(object1, object2, increment):

    object1_position,_ = object1.get_world_pose()
    object2_position,_ = object2.get_world_pose()
    
    direction = (object2_position[0] - object1_position[0], 
                 object2_position[1] - object1_position[1], 
                 object2_position[2] - object1_position[2])
    origin = np.array(object1_position)

    direction_normalized = direction / np.linalg.norm(direction)

    new_origin = origin + direction_normalized * increment
    
    return tuple(new_origin)

def check_raycast_coherence(output_ray, path2):
    path1 = output_ray[0]

    if path1 is None or path2 is None:
        return 0
    
    object_name1 = path1.split('/')[-1]
    object_name2 = path2.split('/')[-1]

    if object_name2 == object_name1:
        return 1
    else:
        return 0

def power_losses(distance):
    return 1/distance**2

def check_power_feasibility(distance, minimum_power):

    if power_losses(distance) >= minimum_power:
        return 1
    else:
        return 0
