from __future__ import annotations
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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
            self.time = 0
            self.prev_velocities = {}  # Store previous velocities for acceleration calculations
            
        except Exception as e:
            logging.error(f"MuJoCo initialization failed: {e}")
            raise

    def get_model_path(self, scene_id: str) -> str:
        """Generate the model path based on the scene_id."""
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
        """Calculate the displacement of a given object in the simulation."""
        # Get initial and current positions
        initial_position, _ = self.get_position(object_id)
        current_position, _ = self.get_position(object_id)
        
        # Calculate Euclidean distance
        displacement = ((current_position[0] - initial_position[0]) ** 2 + 
                        (current_position[1] - initial_position[1]) ** 2 + 
                        (current_position[2] - initial_position[2]) ** 2) ** 0.5
        return displacement

    def compute_force(self, object_id: str, mass: float) -> Dict[str, float]:
        """Compute the force on an object using F = ma."""
        acceleration = self.get_acceleration(object_id)
        return {
            "x": mass * acceleration["x"],
            "y": mass * acceleration["y"],
            "z": mass * acceleration["z"]
        }

    def get_acceleration(self, obj_name: str) -> Dict[str, float]:
        """Retrieve the acceleration of an object using finite differences."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")

        vel_prev = np.copy(self.get_velocity(obj_name))  # Previous velocity
        mujoco.mj_step(self.model, self.data)  # Step simulation
        vel_curr = np.copy(self.get_velocity(obj_name))  # Current velocity

        dt = self.model.opt.timestep
        acceleration = (vel_curr - vel_prev) / dt

        return {"x": acceleration[0], "y": acceleration[1], "z": acceleration[2]}

    def set_velocity(self, obj_name: str, velocity_vector):
        """Set the velocity of an object."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
        self.data.qvel[obj_id * 6: obj_id * 6 + 3] = velocity_vector
        mujoco.mj_forward(self.model, self.data)

    def apply_force(self, obj_name: str, force_vector):
        """Apply a force to an object."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
        self.data.xfrc_applied[obj_id, :3] = force_vector

    def get_velocity(self, obj_name: str):
        """Retrieve the velocity of an object."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
        return self.data.qvel[obj_id * 6: obj_id * 6 + 3]

    def detect_collision(self, obj1_name: str, obj2_name: str):
        """Detect collision between two objects and apply simple elastic forces."""
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
        """Set permissions for object parameter access."""
        self.permissions = permissions

    def get_parameters(self, obj_name: str):
        """Retrieve parameters of an object, respecting scene-defined permissions."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")

        # Check permissions for parameter access
        permissions = getattr(self, 'permissions', {}).get(obj_name, {})
        if not permissions.get("get_parameters", True):  # Default to allowed
            raise PermissionError(f"Access to parameters of '{obj_name}' is not allowed.")

        return {
            "mass": float(self.model.body_mass[obj_id]),
            "bounding_box": self.model.body_inertia[obj_id].tolist(),
            "type": int(self.model.body_parentid[obj_id])
        }

    def move_object(self, name: str, x: float, y: float, z: float):
        """Move an object to a new position."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id == -1:
            raise ValueError(f"Object '{name}' not found.")
        self.data.qpos[obj_id * 7] = x
        self.data.qpos[obj_id * 7 + 1] = y
        self.data.qpos[obj_id * 7 + 2] = z
        mujoco.mj_forward(self.model, self.data)

    def get_position(self, name: str):
        """Get the position of an object."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id == -1:
            raise ValueError(f"Object '{name}' not found.")
        return (self.data.qpos[obj_id * 7], self.data.qpos[obj_id * 7 + 1], self.data.qpos[obj_id * 7 + 2]), self.data.time

    def reset_sim(self):
        """Reset the simulation to its initial state."""
        self.data.qpos[:] = self.start_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.time = 0

    def step(self, duration: float):
        """Step the simulation forward by a specified duration."""
        num_steps = int(duration / self.model.opt.timestep)
        remaining_time = duration - (num_steps * self.model.opt.timestep)

        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        if remaining_time > 0:
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.time += duration

    def get_kinetic_energy(self, obj_name: str, mass: float):
        """Calculate the kinetic energy of an object."""
        velocity = self.get_velocity(obj_name)
        return 0.5 * mass * np.sum(velocity**2)

    def get_potential_energy(self, obj_name: str, mass: float, gravity: float = 9.81):
        """Calculate the potential energy of an object."""
        position, _ = self.get_position(obj_name)
        return mass * gravity * position[2]  # Using z as height

    def get_momentum(self, obj_name: str, mass: float):
        """Calculate the linear momentum of an object."""
        velocity = self.get_velocity(obj_name)
        return {"x": mass * velocity[0], "y": mass * velocity[1], "z": mass * velocity[2]}

    def get_torque(self, obj_name: str):
        """Calculate the torque acting on an object."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found.")
    
        torque = self.data.qfrc_applied[obj_id * 6 + 3: obj_id * 6 + 6]
        return {"x": torque[0], "y": torque[1], "z": torque[2]}

    def get_center_of_mass(self):
        """Calculate the center of mass of the entire scene."""
        total_mass = np.sum(self.model.body_mass)
        weighted_positions = np.sum(self.model.body_mass[:, None] * self.data.xpos, axis=0)
        
        center_of_mass = weighted_positions / total_mass
        return {"x": center_of_mass[0], "y": center_of_mass[1], "z": center_of_mass[2]}

    def get_angular_momentum(self, obj_name: str, mass: float):
        """Calculate the angular momentum of an object."""
        position, _ = self.get_position(obj_name)
        velocity = self.get_velocity(obj_name)
    
        # Convert position to numpy array for cross product
        pos_array = np.array(position)
        angular_momentum = np.cross(pos_array, mass * velocity)
        return {"x": angular_momentum[0], "y": angular_momentum[1], "z": angular_momentum[2]}

    def change_position(self, obj_name: str, dx: float, dy: float, dz: float, in_world_frame: bool = True):
        """Change the position of an object by a given displacement."""
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

    def load_scene(self, scene_id: str):
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.close()
    
            scene_id = str(scene_id)  # ensure it's a string
            self.model_path = self.get_model_path(scene_id)
            logging.info(f"Loading model from: {self.model_path}")
    
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
    
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.start_pos = np.copy(self.data.qpos)
            self.time = 0
    
        except Exception as e:
            logging.error(f"Failed to load scene {scene_id}: {e}")

    def __del__(self):
        """Clean up resources when the Simulator object is destroyed."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
