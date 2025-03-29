import mujoco
import mujoco.viewer
import numpy as np
import time
import os

class Simulation:
    def __init__(self, scene_id: str):
        """
        Initialize the Simulation class with the provided scene ID.
        The model is automatically loaded based on the scene_id.
        """
        self.scene_id = scene_id

        # Dynamically determine the model path using the scene_id
        self.model_path = self.get_model_path(scene_id)

        # Load the model and create the MjModel and MjData
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Launch the viewer in passive mode
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.start_pos = np.copy(self.data.qpos)
        self.time = 0

    def get_model_path(self, scene_id: str) -> str:
        """Generate the model path based on the scene_id."""
        model_dir = "/path/to/models"  # Adjust to where your models are stored
        model_file = f"scene_{scene_id}.xml"  # Assuming the file follows this naming pattern
        model_path = os.path.join(model_dir, model_file)

        # Ensure the path is valid
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for scene {scene_id} not found at {model_path}")

        return model_path
    
    def render(self):
        """Render the current simulation frame."""
        self.viewer.sync()
        return self.viewer.capture_frame()

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


