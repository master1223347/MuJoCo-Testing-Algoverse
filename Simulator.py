import mujoco
import mujoco.viewer
import numpy as np
import time

#BIG TODO USE CHATGPT TO WRITE A VERY DETAILED DOCSTRING FOR EVERY METHOD HERE, SHOULD INCLUDE PARAMETERS AND TYPES AS WELL AS RETURN AND RETURN TYPE

class Simulation:
    def __init__(self, model_path): #TAKE IN SCENE ID ONLY
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.start_pos = np.copy(self.data.qpos)
        self.time = 0
    
    def render(self):
        self.viewer.sync()
        return self.viewer.capture_frame()

    def set_velocity(self, obj_name, velocity_vector):
    obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if obj_id == -1:
        raise ValueError("Object not found")
    self.data.qvel[obj_id * 6 : obj_id * 6 + 3] = velocity_vector  # Apply velocity to the object
    mujoco.mj_forward(self.model, self.data)  # Update simulation state
        
    def apply_force(self, obj_name, force_vector):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError("Object not found")
        self.data.xfrc_applied[obj_id, :3] = force_vector  # Apply force to the body
    
    def get_velocity(self, obj_name):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError("Object not found")
        return self.data.qvel[obj_id]
    
    def detect_collision(self, obj1_name, obj2_name):
        obj1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj1_name)
        obj2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj2_name)
        
        if obj1_id == -1 or obj2_id == -1:
            raise ValueError("One or both objects not found")
        
        for contact in self.data.contact:
            if (contact.geom1 == obj1_id and contact.geom2 == obj2_id) or (contact.geom1 == obj2_id and contact.geom2 == obj1_id):
                # Apply opposite forces (simple elastic response)
                normal_force = contact.frame[:3] * contact.dist
                self.apply_force(obj1_name, -normal_force)
                self.apply_force(obj2_name, normal_force)
                return True
        return False

    def set_permissions(self, permisisons):
        self.permissions = permissions

    
    def get_parameters(self, obj_name):
    """Retrieve parameters of an object, respecting scene-defined permissions."""
    obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if obj_id == -1:
        raise ValueError("Object not found")

    # Check if the scene-defined permissions allow access to this object's parameters
    permissions = self.permissions.get(obj_name, {})
    if not permissions.get("get_parameters", False):
        raise PermissionError(f"Access to parameters of '{obj_name}' is not allowed.")

    return {
        "mass": self.model.body_mass[obj_id],
        "bounding_box": self.model.body_inertia[obj_id],  # Ensure this is the correct attribute
        "type": self.model.body_parentid[obj_id]
    }

    
    def move_object(self, name, x, y, z):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id == -1:
            raise ValueError("Object not found")
        self.data.qpos[obj_id * 7] = x
        self.data.qpos[obj_id * 7 + 1] = y
        self.data.qpos[obj_id * 7 + 2] = z
        mujoco.mj_forward(self.model, self.data)
    
    def get_position(self, name):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id == -1:
            raise ValueError("Object not found")
        return (self.data.qpos[obj_id * 7], self.data.qpos[obj_id * 7 + 1], self.data.qpos[obj_id * 7 + 2]), self.data.time
    
    def reset_sim(self):
        self.data.qpos[:] = self.start_pos  # Reset position
        self.data.qvel[:] = 0  # Stop movement
        mujoco.mj_forward(self.model, self.data)
        self.time = 0
    

    def step(self, duration):
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

