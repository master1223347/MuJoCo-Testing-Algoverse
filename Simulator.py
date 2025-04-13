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
        """Generate the model path based on the scene_id (returns a string)"""
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
        """Render the current simulation frame (returns nothing specific)"""
        self.viewer.sync()
        return self.viewer.capture_frame()

    def step(self, duration: float = 1.0):
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

    def get_displacement(self, object_id: str) -> dict:
        """Calculate the displacement of a given object in the simulation."""
        # Get initial and current positions
        initial_position, _ = self.get_position(object_id)
        current_position, _ = self.get_position(object_id)
        
        # Calculate Euclidean distance
        displacement = ((current_position[0] - initial_position[0]) ** 2 + 
                        (current_position[1] - initial_position[1]) ** 2 + 
                        (current_position[2] - initial_position[2]) ** 2) ** 0.5
        return {"displacement": displacement}

    def compute_force(self, object_id: str, mass: float) -> dict:
        """Compute the force on an object using F = ma."""
        acceleration = self.get_acceleration(object_id)
        return {
            "x": mass * acceleration["x"],
            "y": mass * acceleration["y"],
            "z": mass * acceleration["z"]
        }

    def get_acceleration(self, object_id: str) -> dict:
        """Retrieve the acceleration of an object."""
        # Get velocity data for acceleration calculation
        vel_prev = np.copy(self.get_velocity(object_id))  # Previous velocity
        mujoco.mj_step(self.model, self.data)  # Step simulation
        vel_curr = np.copy(self.get_velocity(object_id))  # Current velocity

        dt = self.model.opt.timestep
        acceleration = (vel_curr - vel_prev) / dt

        return {"x": acceleration[0], "y": acceleration[1], "z": acceleration[2]}

    def set_velocity(self, object_id: str, velocity_vector) -> dict:
        """Set the velocity of an object."""
        self.data.qvel[object_id * 6: object_id * 6 + 3] = velocity_vector
        mujoco.mj_forward(self.model, self.data)
        return {"status": "velocity_set", "object_id": object_id, "velocity": velocity_vector}

    def apply_force(self, object_id: str, force_vector) -> dict:
        """Apply a force to an object."""
        self.data.xfrc_applied[object_id, :3] = force_vector
        return {"status": "force_applied", "object_id": object_id, "force": force_vector}

    def apply_torque(self, object_id: str, torque_vector) -> dict:
        """Apply a torque to an object."""
        self.data.xfrc_applied[object_id, 3:6] = torque_vector
        return {"status": "torque_applied", "object_id": object_id, "torque": torque_vector}

    def get_velocity(self, object_id: str) -> dict:
        """Retrieve the velocity of an object."""
        velocity = self.data.qvel[object_id * 6: object_id * 6 + 3]
        return {"velocity": velocity}

    def detect_collision(self, obj1_id: str, obj2_id: str) -> dict:
        """Detect collision between two objects and apply simple elastic forces."""
        for contact in self.data.contact:
            if (contact.geom1 == obj1_id and contact.geom2 == obj2_id) or \
               (contact.geom1 == obj2_id and contact.geom2 == obj1_id):
                # Apply simple elastic response
                normal_force = contact.frame[:3] * contact.dist
                self.apply_force(obj1_id, -normal_force)
                self.apply_force(obj2_id, normal_force)
                return {"collision_detected": True}
        return {"collision_detected": False}

    def set_permissions(self, permissions) -> dict:
        """Set permissions for object parameter access (returns nothing specific)"""
        self.permissions = permissions
        return {"status": "permissions_set"}

    def get_parameters(self, object_id: str) -> dict:
        """Retrieve parameters of an object, respecting scene-defined permissions."""
        # Check permissions for parameter access
        permissions = getattr(self, 'permissions', {}).get(object_id, {})
        if not permissions.get("get_parameters", True):  # Default to allowed
            raise PermissionError(f"Access to parameters of object with ID {object_id} is not allowed.")

        return {
            "mass": float(self.model.body_mass[object_id]),
            "bounding_box": self.model.body_inertia[object_id].tolist(),
            "type": int(self.model.body_parentid[object_id])
        }

    def move_object(self, object_id: str, x: float, y: float, z: float) -> dict:
        """Move an object to a new position."""
        self.data.qpos[object_id * 7] = x
        self.data.qpos[object_id * 7 + 1] = y
        self.data.qpos[object_id * 7 + 2] = z
        mujoco.mj_forward(self.model, self.data)
        return {"position": (x, y, z)}

    def get_position(self, object_id: str) -> dict:
        """Get the position of an object."""
        return (self.data.qpos[object_id * 7], 
                self.data.qpos[object_id * 7 + 1], 
                self.data.qpos[object_id * 7 + 2]), self.data.time

    def get_kinetic_energy(self, object_id: str, mass: float) -> dict:
        """Calculate the kinetic energy of an object."""
        velocity = self.get_velocity(object_id)
        kinetic_energy = 0.5 * mass * np.sum(velocity**2)
        return {"kinetic_energy": kinetic_energy}

    def get_potential_energy(self, object_id: str, mass: float, gravity: float = 9.81) -> dict:
       """Calculate the potential energy of an object."""
        position = self.get_position(object_id)
        potential_energy = mass * gravity * position[2]  # Using z as height
        return {"potential_energy": potential_energy}

    def get_momentum(self, object_id: str, mass: float):
        """Calculate the linear momentum of an object."""
        velocity = self.get_velocity(object_id)
        momentum = {"x": mass * velocity[0], "y": mass * velocity[1], "z": mass * velocity[2]}
        return {"momentum": momentum}

    def get_torque(self, object_id: str):
        """Calculate the torque acting on an object."""
        torque = self.data.qfrc_applied[object_id * 6 + 3: object_id * 6 + 6]
        torque_dict = {"x": torque[0], "y": torque[1], "z": torque[2]}
        return {"torque": torque_dict}
    
    def get_center_of_mass(self):
        """Calculate the center of mass of the entire scene."""
        total_mass = np.sum(self.model.body_mass)
        weighted_positions = np.sum(self.model.body_mass[:, None] * self.data.xpos, axis=0)
        
        center_of_mass = weighted_positions / total_mass
        center_of_mass_dict = {"x": center_of_mass[0], "y": center_of_mass[1], "z": center_of_mass[2]}
        return {"center_of_mass": center_of_mass_dict}
    
    def get_angular_momentum(self, object_id: str, mass: float):
        """Calculate the angular momentum of an object."""
        position, _ = self.get_position(object_id)
        velocity = self.get_velocity(object_id)
        
        # Convert position to numpy array for cross product
        pos_array = np.array(position)
        angular_momentum = np.cross(pos_array, mass * velocity)
        
        angular_momentum_dict = {"x": angular_momentum[0], "y": angular_momentum[1], "z": angular_momentum[2]}
        return {"angular_momentum": angular_momentum_dict}
    
    def change_position(self, object_id: str, dx: float, dy: float, dz: float, in_world_frame: bool = True):
        """Change the position of an object by a given displacement."""
        pos_x = self.data.qpos[object_id * 7]
        pos_y = self.data.qpos[object_id * 7 + 1]
        pos_z = self.data.qpos[object_id * 7 + 2]
    
    if in_world_frame:
        # Apply displacement directly in world frame
        self.data.qpos[object_id * 7] = pos_x + dx
        self.data.qpos[object_id * 7 + 1] = pos_y + dy
        self.data.qpos[object_id * 7 + 2] = pos_z + dz
    else:
        # Apply displacement in local frame
        quat = self.data.qpos[object_id * 7 + 3: object_id * 7 + 7]
        rot_matrix = self.quat_to_rot_matrix(quat)
        local_disp = np.array([dx, dy, dz])
        world_disp = rot_matrix @ local_disp
        self.data.qpos[object_id * 7] = pos_x + world_disp[0]
        self.data.qpos[object_id * 7 + 1] = pos_y + world_disp[1]
        self.data.qpos[object_id * 7 + 2] = pos_z + world_disp[2]
    
        # Update simulation
        mujoco.mj_forward(self.model, self.data)
        new_position = {"x": self.data.qpos[object_id * 7], "y": self.data.qpos[object_id * 7 + 1], "z": self.data.qpos[object_id * 7 + 2]}
        return {"new_position": new_position}

    def quat_to_rot_matrix(self, q) -> dict:
        """Convert a quaternion to a rotation matrix."""
        q = q / np.linalg.norm(q)  # Normalize quaternion
        w, x, y, z = q
        return {
            "rotation_matrix": np.array([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
            ])
        }

    def reset_sim(self):
        """Reset the simulation to its initial state."""
        self.data.qpos[:] = self.start_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.time = 0

    def scene_step(self) -> str:
        """
        Advance the simulation by one timestep.
        Useful for progressing the scene in fine increments.
        """
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
        self.viewer.sync()
        self.time += self.model.opt.timestep
        return f"Scene stepped by {self.model.opt.timestep:.4f} seconds."

    def __del__(self):
        """Clean up resources when the Simulator object is destroyed."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
