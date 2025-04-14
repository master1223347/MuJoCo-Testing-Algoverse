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

    def get_body_id(self, object_id: str) -> int:
        object_id = str(object_id)
        name = object_id
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body with name '{name}' not found in the scene.")
        return body_id

    def move_object(self, object_id: str, x: float, y: float, z: float) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on the object name

            # Get the joint corresponding to the object and set its position
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_id}_joint")
            if joint_id == -1:
                return {"error": f"No joint named {object_id}_joint"}

            joint_qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[joint_qpos_addr:joint_qpos_addr+3] = np.array([x, y, z])
            mujoco.mj_forward(self.model, self.data)
            return {"position": (x, y, z)}
        
        except Exception as e:
            return {"error": str(e)}

    def compute_force(self, object_id: str, mass: float) -> dict:
        """Compute the force on an object using F = ma."""
        object_id = str(object_id)  # Ensure object_id is a string
        acceleration = self.get_acceleration(object_id)
        return {
            "x": mass * acceleration["x"],
            "y": mass * acceleration["y"],
            "z": mass * acceleration["z"]
        }

    def get_acceleration(self, object_id: str) -> dict:
        try:
            object_id = str(object_id)
            body_id = self.get_body_id(object_id)
            dof_adr = self.model.body_dofadr[body_id]
            
            # Save previous velocity
            prev_vel = np.copy(self.data.qvel[dof_adr:dof_adr+3])
            
            # Step the sim slightly forward
            mujoco.mj_step(self.model, self.data)
            new_vel = self.data.qvel[dof_adr:dof_adr+3]
            dt = self.model.opt.timestep
            
            acc = (new_vel - prev_vel) / dt
            return {"x": float(acc[0]), "y": float(acc[1]), "z": float(acc[2])}
        except Exception as e:
            return {"error": str(e)}
        
    def get_center_of_mass(self) -> dict:
        try:
            com = self.data.subtree_com[0]
            return {"center_of_mass": com.tolist()}
        except Exception as e:
            return {"error": str(e)}

    def set_velocity(self, object_id: str, velocity_vector: list) -> dict:
        """Set the velocity of an object."""
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get the body ID based on object_id

            # The velocity vector is passed in as an array of 3 elements (x, y, z)
            # Accessing the correct portion of qvel for the given body (body_id * 6 for 6 DoFs)
            self.data.qvel[body_id * 6: body_id * 6 + 3] = velocity_vector  # Set the linear velocity

            # After updating, we recompute the forward kinematics
            mujoco.mj_forward(self.model, self.data)

            return {"status": "velocity_set", "object_id": object_id, "velocity": velocity_vector}
        except Exception as e:
            return {"error": str(e)}

    def apply_force(self, object_id: str, force_vector) -> dict:
        """Apply a force to an object."""
        object_id = str(object_id)  # Ensure object_id is a string
        self.data.xfrc_applied[int(object_id), :3] = force_vector
        return {"status": "force_applied", "object_id": object_id, "force": force_vector}

    def apply_torque(self, object_id: str, torque_vector) -> dict:
        """Apply a torque to an object."""
        object_id = str(object_id)  # Ensure object_id is a string
        self.data.xfrc_applied[int(object_id), 3:6] = torque_vector
        return {"status": "torque_applied", "object_id": object_id, "torque": torque_vector}

    def get_velocity(self, object_id: str) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            vel = self.data.cvel[body_id][:3]  # Get the velocity for the body (linear part)
            return {"velocity": vel.tolist()}
        except Exception as e:
            return {"error": str(e)}


    def detect_collision(self, obj1_id: str, obj2_id: str) -> dict:
        """Detect collision between two objects and apply simple elastic forces."""
        obj1_id = str(obj1_id)
        obj2_id = str(obj2_id)
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

    def get_position(self, object_id: str) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            pos = self.data.xpos[body_id]  # Get the position of the body
            return {"position": pos.tolist(), "time": self.data.time}
        
        except Exception as e:
            return {"error": str(e)}

    def get_kinetic_energy(self, object_id: str, mass: float) -> dict:
        """Calculate the kinetic energy of an object."""
        object_id = str(object_id)  # Ensure object_id is a string
        velocity = self.get_velocity(object_id)
        kinetic_energy = 0.5 * mass * np.sum(velocity["velocity"]**2)
        return {"kinetic_energy": kinetic_energy}

    def get_potential_energy(self, object_id: str, mass: float, gravity: float = 9.81) -> dict:
        """Calculate the potential energy of an object."""
        object_id = str(object_id)  # Ensure object_id is a string
        position = self.get_position(object_id)
        potential_energy = mass * gravity * position["position"][2]  # Using z as height
        return {"potential_energy": potential_energy}

    def get_momentum(self, object_id: str, mass: float):
        """Calculate the linear momentum of an object."""
        object_id = str(object_id)  # Ensure object_id is a string
        velocity = self.get_velocity(object_id)
        momentum = {"x": mass * velocity["velocity"][0], "y": mass * velocity["velocity"][1], "z": mass * velocity["velocity"][2]}
        return {"momentum": momentum}

    def get_torque(self, object_id: str):
        """Calculate the torque acting on an object."""
        object_id = str(object_id)  # Ensure object_id is a string
        torque = self.data.qfrc_applied[int(object_id) * 6 + 3: int(object_id) * 6 + 6]
        torque_dict = {"x": torque[0], "y": torque[1], "z": torque[2]}
        return {"torque": torque_dict}
    
    def get_displacement(self, object_id: str) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            current_pos = self.data.xpos[body_id]  # Get the current position
            start_pos = self.start_pos[:3]  # Get the initial position from start state
            displacement = np.linalg.norm(current_pos - start_pos)
            return {"displacement": float(displacement)}
        except Exception as e:
            return {"error": str(e)}
    
    def get_angular_momentum(self, object_id: str, mass: float) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            angvel = self.data.cvel[body_id][3:6]  # Angular velocity part of the body
            ang_momentum = mass * angvel  # Angular momentum = mass * velocity
            return {"angular_momentum": ang_momentum.tolist()}
        except Exception as e:
            return {"error": str(e)}

    def change_position(self, object_id: str, dx: float, dy: float, dz: float, in_world_frame: bool) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_id}_joint")
            if joint_id == -1:
                return {"error": f"No joint named {object_id}_joint"}
            
            joint_qpos_addr = self.model.jnt_qposadr[joint_id]
            if in_world_frame:
                self.data.qpos[joint_qpos_addr:joint_qpos_addr+3] += np.array([dx, dy, dz])
            else:
                self.data.qpos[joint_qpos_addr] += dx
                self.data.qpos[joint_qpos_addr+1] += dy
                self.data.qpos[joint_qpos_addr+2] += dz
            
            mujoco.mj_forward(self.model, self.data)
            return {"new_position": self.data.qpos[joint_qpos_addr:joint_qpos_addr+3].tolist()}
        except Exception as e:
            return {"error": str(e)}


    def quat_to_rot_matrix(self, q: list[float]) -> dict:
        try:
            q_np = np.array(q)
            mat = np.zeros((3, 3))
            mujoco.mju_matQuat(mat, q_np)
            return {"rotation_matrix": mat.tolist()}
        except Exception as e:
            return {"error": str(e)}

    def reset_sim(self):
        """Reset the simulation to its initial state."""
        self.data.qpos[:] = self.start_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.time = 0
    
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
