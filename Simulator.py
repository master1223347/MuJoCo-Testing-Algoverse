from __future__ import annotations
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import xml.etree.ElementTree as ET
import Scene
from typing import Dict
import logging




class Simulator:
    def __init__(self, scene_id: str):
        """
        Initialize the Simulation class with the provided scene ID.
        The model is automatically loaded based on the scene_id.
        """
        self.scene_id = scene_id
        self.model_path = self.get_model_path(scene_id)
        
        try:
            # Load model with error handling
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Initialize viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.start_pos = np.copy(self.data.qpos)
            
        except Exception as e:
            logging.error(f"MuJoCo initialization failed: {e}")
            raise

    def get_model_path(self, scene_id: str) -> str:
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            scenes_dir = os.path.join(script_dir, "Scenes")
            
            # Extract scene number and construct XML path
            scene_number = scene_id.split("_")[-1]
            xml_path = os.path.join(scenes_dir, f"Scene{scene_number}", f"scene{scene_number}.xml")
            
            # Verify if the file exists
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Scene XML not found at: {xml_path}")
            
            return xml_path.replace("\\", "/")
        except Exception as e:
            logging.error(f"Path construction failed: {e}")
            raise


        
    def render(self):
        """Render the current simulation frame."""
        self.viewer.sync()
        return self.viewer.capture_frame()

    def get_displacement(self, object_id: str) -> float:
        """
        Calculate the displacement of a given object in the simulation.
        
        Args:
            object_id (str): The identifier of the object to measure displacement.
        
        Returns:
            float: The displacement of the object in meters (or your unit of measurement).
        """
        # Example logic: calculate displacement based on object's initial and current position
        initial_position = self.get_position(object_id)  # Assume this returns a dictionary with x, y, z
        current_position = self.get_position(object_id)  # Get the position at the current time step
        
        # Simple Euclidean distance formula for displacement
        displacement = ((current_position['x'] - initial_position['x']) ** 2 + 
                        (current_position['y'] - initial_position['y']) ** 2 + 
                        (current_position['z'] - initial_position['z']) ** 2) ** 0.5
        return displacement

    def compute_force(self, object_id: str, mass: float) -> Dict[str, float]:
        """
        Compute the force on an object using F = ma (Force = mass * acceleration).
        
        Args:
            object_id (str): The identifier of the object to compute force.
            mass (float): The mass of the object (in kilograms or your unit of mass).
        
        Returns:
            Dict[str, float]: The force applied to the object in the x, y, and z directions.
        """
        # Example logic: calculate force based on acceleration (you could use more detailed physics here)
        # Get the velocity of the object at the current time step
        velocity = self.get_velocity(object_id)  # Assuming `get_velocity` is implemented
        
        
        acceleration = self.get_acceleration(object_id)  # New method needed
        return {
        "x": mass * acceleration['x'],
        "y": mass * acceleration['y'],
        "z": mass * acceleration['z']
    }


    def set_velocity(self, obj_name, velocity_vector):
        """
        Set the velocity of an object.
        
        Parameters:
        - obj_name (str): The name of the object.
        - velocity_vector (np.ndarray): The velocity to apply to the object.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
        self.data.qvel[obj_id * 6: obj_id * 6 + 3] = velocity_vector  # Apply velocity
        mujoco.mj_forward(self.model, self.data)  # Update simulation state

    def apply_force(self, obj_name, force_vector):
        """
        Apply a force to an object.
        
        Parameters:
        - obj_name (str): The name of the object.
        - force_vector (np.ndarray): The force to apply to the object.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
        self.data.xfrc_applied[obj_id, :3] = force_vector  # Apply force

    def get_velocity(self, obj_name):
        """
        Retrieve the velocity of an object.
        
        Parameters:
        - obj_name (str): The name of the object.
        
        Returns:
        - velocity (np.ndarray): The velocity of the object.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
        return self.data.qvel[obj_id]

    def detect_collision(self, obj1_name, obj2_name):
        """
        Detect collision between two objects and apply simple elastic forces.

        Parameters:
        - obj1_name (str): The name of the first object.
        - obj2_name (str): The name of the second object.
        
        Returns:
        - bool: True if collision detected, False otherwise.
        """
        obj1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj1_name)
        obj2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj2_name)

        if obj1_id == -1 or obj2_id == -1:
            raise ValueError("One or both objects not found.")

        for contact in self.data.contact:
            if (contact.geom1 == obj1_id and contact.geom2 == obj2_id) or \
               (contact.geom1 == obj2_id and contact.geom2 == obj1_id):
                # Apply simple elastic response
                normal_force = contact.frame[:3] * contact.dist
                self.apply_force(obj1_name, -normal_force)
                self.apply_force(obj2_name, normal_force)
                return True
        return False

    def set_permissions(self, permissions):
        """
        Set permissions for object parameter access.
        
        Parameters:
        - permissions (dict): A dictionary of permissions for objects.
        """
        self.permissions = permissions

    def get_parameters(self, obj_name):
        """
        Retrieve parameters of an object, respecting scene-defined permissions.
        
        Parameters:
        - obj_name (str): The name of the object.
        
        Returns:
        - dict: Parameters such as mass, bounding box, and type.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")

        # Check permissions for parameter access
        permissions = self.permissions.get(obj_name, {})
        if not permissions.get("get_parameters", False):
            raise PermissionError(f"Access to parameters of '{obj_name}' is not allowed.")

        return {
            "mass": self.model.body_mass[obj_id],
            "bounding_box": self.model.body_inertia[obj_id],  # Adjust based on actual attributes
            "type": self.model.body_parentid[obj_id]
        }

    def move_object(self, name, x, y, z):
        """
        Move an object to a new position.
        
        Parameters:
        - name (str): The name of the object.
        - x (float): The new x-coordinate.
        - y (float): The new y-coordinate.
        - z (float): The new z-coordinate.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id == -1:
            raise ValueError(f"Object '{name}' not found.")
        self.data.qpos[obj_id * 7] = x
        self.data.qpos[obj_id * 7 + 1] = y
        self.data.qpos[obj_id * 7 + 2] = z
        mujoco.mj_forward(self.model, self.data)

    def get_position(self, name):
        """
        Get the position of an object.
        
        Parameters:
        - name (str): The name of the object.
        
        Returns:
        - tuple: Position (x, y, z) and time.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id == -1:
            raise ValueError(f"Object '{name}' not found.")
        return (self.data.qpos[obj_id * 7], self.data.qpos[obj_id * 7 + 1], self.data.qpos[obj_id * 7 + 2]), self.data.time

    def reset_sim(self):
        """Reset the simulation to its initial state."""
        self.data.qpos[:] = self.start_pos  # Reset position
        self.data.qvel[:] = 0  # Stop movement
        mujoco.mj_forward(self.model, self.data)
        self.time = 0

    def step(self, duration):
        """
        Step the simulation forward by a specified duration.
        
        Parameters:
        - duration (float): The time to advance the simulation.
        """
        num_steps = int(duration / self.model.opt.timestep)  # Compute steps based on dt
        remaining_time = duration - (num_steps * self.model.opt.timestep)  # Compute leftover time

        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        if remaining_time > 0:  # Handle small leftover time
            mujoco.mj_step(self.model, self.data)  
            if self.viewer is not None:
                self.viewer.sync()

        self.time += duration  # Now accurately reflects total time
        def get_acceleration(self, obj_name):
    """
    Retrieve the acceleration of an object using finite differences.

    Parameters:
    - obj_name (str): The name of the object.

    Returns:
    - acceleration (dict): The acceleration in x, y, and z directions.
    """
    obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if obj_id == -1:
        raise ValueError(f"Object '{obj_name}' not found.")

    vel_prev = np.copy(self.get_velocity(obj_name))  # Previous velocity
    mujoco.mj_step(self.model, self.data)  # Step simulation
    vel_curr = np.copy(self.get_velocity(obj_name))  # Current velocity

    dt = self.model.opt.timestep
    acceleration = (vel_curr - vel_prev) / dt

    return {"x": acceleration[0], "y": acceleration[1], "z": acceleration[2]}


    def get_kinetic_energy(self, obj_name, mass):
        """
        Calculate the kinetic energy of an object.

        Parameters:
        - obj_name (str): The name of the object.
        - mass (float): The mass of the object.

        Returns:
        - float: The kinetic energy of the object.
        """
        velocity = self.get_velocity(obj_name)
        return 0.5 * mass * np.sum(velocity**2)


    def get_potential_energy(self, obj_name, mass, gravity=9.81):
        """
        Calculate the potential energy of an object.
    
        Parameters:
        - obj_name (str): The name of the object.
        - mass (float): The mass of the object.
        - gravity (float): The gravitational acceleration (default is 9.81 m/s²).
    
        Returns:
        - float: The potential energy of the object.
        """
        position, _ = self.get_position(obj_name)
        return mass * gravity * position[2]  # Using z as height


    def get_momentum(self, obj_name, mass):
        """
        Calculate the linear momentum of an object.
    
        Parameters:
        - obj_name (str): The name of the object.
        - mass (float): The mass of the object.
    
        Returns:
        - dict: The momentum in x, y, and z directions.
        """
        velocity = self.get_velocity(obj_name)
        return {"x": mass * velocity[0], "y": mass * velocity[1], "z": mass * velocity[2]}
    
    
    def get_torque(self, obj_name):
        """
        Calculate the torque acting on an object.
    
        Parameters:
        - obj_name (str): The name of the object.
    
        Returns:
        - dict: The torque in x, y, and z directions.
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
    
        torque = self.data.qfrc_applied[obj_id * 6 + 3: obj_id * 6 + 6]
        return {"x": torque[0], "y": torque[1], "z": torque[2]}
    
    
    def get_center_of_mass(self):
        """
        Calculate the center of mass of the entire scene.
    
        Returns:
        - dict: The center of mass in x, y, and z directions.
        """
        total_mass = np.sum(self.model.body_mass)
        weighted_positions = np.sum(self.model.body_mass[:, None] * self.data.xpos, axis=0)
        
        center_of_mass = weighted_positions / total_mass
        return {"x": center_of_mass[0], "y": center_of_mass[1], "z": center_of_mass[2]}
    
    
    def get_angular_momentum(self, obj_name, mass):
        """
        Calculate the angular momentum of an object.
    
        Parameters:
        - obj_name (str): The name of the object.
        - mass (float): The mass of the object.
    
        Returns:
        - dict: The angular momentum in x, y, and z directions.
        """
        position, _ = self.get_position(obj_name)
        velocity = self.get_velocity(obj_name)
    
        angular_momentum = np.cross(position, mass * velocity)
        return {"x": angular_momentum[0], "y": angular_momentum[1], "z": angular_momentum[2]}


    #change pos
    def change_position(self, obj_name, dx, dy, dz, in_world_frame=True):
    """
    Change the position of an object by a given displacement.
    
    Parameters:
    - obj_name (str): The name of the object.
    - dx (float): The displacement in the x-direction.
    - dy (float): The displacement in the y-direction.
    - dz (float): The displacement in the z-direction.
    - in_world_frame (bool): Whether the displacement is in world frame (True) 
                             or object's local frame (False).
    """
    obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if obj_id == -1:
        raise ValueError(f"Object '{obj_name}' not found.")
    
    # Get current position
    pos_x = self.data.qpos[obj_id * 7]
    pos_y = self.data.qpos[obj_id * 7 + 1]
    pos_z = self.data.qpos[obj_id * 7 + 2]
    
    if in_world_frame:
        # Apply displacement directly in world frame
        self.data.qpos[obj_id * 7] = pos_x + dx
        self.data.qpos[obj_id * 7 + 1] = pos_y + dy
        self.data.qpos[obj_id * 7 + 2] = pos_z + dz
    else:
        # Apply displacement in local frame
        # Get quaternion (w, x, y, z format)
        quat = self.data.qpos[obj_id * 7 + 3:obj_id * 7 + 7]
        
        # Create rotation matrix from quaternion
        def quat_to_rot_matrix(q):
            # Normalize quaternion to ensure it represents a valid rotation
            q = q / np.linalg.norm(q)
            w, x, y, z = q
            
            # Construct rotation matrix
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ])
        
        rot_matrix = quat_to_rot_matrix(quat)
        
        # Rotate displacement vector
        local_disp = np.array([dx, dy, dz])
        world_disp = rot_matrix @ local_disp
        
        # Apply rotated displacement
        self.data.qpos[obj_id * 7] = pos_x + world_disp[0]
        self.data.qpos[obj_id * 7 + 1] = pos_y + world_disp[1]
        self.data.qpos[obj_id * 7 + 2] = pos_z + world_disp[2]
    
    # Update simulation
    mujoco.mj_forward(self.model, self.data)



    


