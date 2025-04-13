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
        parameters = super().get_parameters(object_id)
        return parameters

    def move_object(self, object_id: str, x: float, y: float, z: float) -> dict:
        """Move an object to a new position."""
        super().move_object(object_id, x, y, z)
        return {"position": (x, y, z)}

    def get_position(self, object_id: str) -> dict:
        """Get the position of an object."""
        position, _ = super().get_position(object_id)
        return {"position": position}

    def get_kinetic_energy(self, object_id: str, mass: float) -> dict:
        """Calculate the kinetic energy of an object."""
        kinetic_energy = super().get_kinetic_energy(object_id, mass)
        return {"kinetic_energy": kinetic_energy}

    def get_potential_energy(self, object_id: str, mass: float, gravity: float = 9.81) -> dict:
        """Calculate the potential energy of an object."""
        potential_energy = super().get_potential_energy(object_id, mass, gravity)
        return {"potential_energy": potential_energy}

    def get_momentum(self, object_id: str, mass: float) -> dict:
        """Calculate the linear momentum of an object."""
        momentum = super().get_momentum(object_id, mass)
        return {"momentum": momentum}

    def get_torque(self, object_id: str) -> dict:
        """Calculate the torque acting on an object."""
        torque = super().get_torque(object_id)
        return {"torque": torque}

    def get_center_of_mass(self) -> dict:
        """Calculate the center of mass of the entire scene."""
        center_of_mass = super().get_center_of_mass()
        return {"center_of_mass": center_of_mass}

    def get_angular_momentum(self, object_id: str, mass: float) -> dict:
        """Calculate the angular momentum of an object."""
        angular_momentum = super().get_angular_momentum(object_id, mass)
        return {"angular_momentum": angular_momentum}

    def change_position(self, object_id: str, dx: float, dy: float, dz: float, in_world_frame: bool = True) -> dict:
        """Change the position of an object by a given displacement."""
        super().change_position(object_id, dx, dy, dz, in_world_frame)
        return {"new_position": (dx, dy, dz)}

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

    def load_scene(self, scene_id: str) -> dict:
        """Load the scene (returns nothing specific)"""
        super().load_scene(scene_id)
        return {"status": "scene_loaded", "scene_id": scene_id}

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
        """Clean up resources when the Simulator object is destroyed (returns nothing specific)"""
        super().__del__()
        return {"status": "simulator_destroyed"}
