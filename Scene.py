import json
import os
import numpy as np
import Simulator

class Scene:
    """
    Represents a physics-based scene loaded from a JSON file. This class interacts with a simulator to 
    analyze and manipulate objects in the scene.
    """
    
    def __init__(self, scene_id: str, simulator: Simulator):
        """
        Initializes the Scene object by loading scene data from a JSON file.
        
        Args:
            scene_id (str): The unique identifier for the scene.
            simulator (Simulator): The simulator instance used for interaction.
        """
        self.scene_id = scene_id
        self.simulator = simulator  # Store the simulator object
        self.scene_number = int(scene_id.split("_")[-1])
        self.terminal = "robin"  # For setting it up, switch this to: "robin", "abhinav", or "sid".
        
        # Construct file path based on the terminal setting
        if self.terminal == "utkarsh":
            drive = os.path.splitdrive(os.getcwd())[0].lower().rstrip(":") + ":/"
            base_dir = os.path.join(drive, "Users", "inbox", "Algoverse")
            self.file_path = os.path.join(base_dir, "MuJoCo-Testing-Algoverse", "Scenes", 
                                          f"Scene{self.scene_number}", f"scene{self.scene_number}.json")
            self.file_path = self.file_path.replace("\\", "/")
        elif self.terminal == "robin":
            base_dir = r"C:\Users\robin\OneDrive\Documents\Algoverse\MuJoCo-Testing-Algoverse-main\Scenes"
            self.file_path = os.path.join(base_dir, f"Scene{self.scene_number}", f"scene{self.scene_number}.json")
        elif self.terminal == "abhinav":
            base_dir = r"C:\Users\epicg\Algoverse\MuJoCo-Testing-Algoverse\Scenes"
            self.file_path = os.path.join(base_dir, f"Scene{self.scene_number}", f"scene{self.scene_number}.json")
        elif self.terminal == "sid":
            base_dir = r"C:\Users\siddh\OneDrive\Desktop\Algoverse\MuJoCo-Testing-Algoverse\Scenes"
            self.file_path = os.path.join(base_dir, f"Scene{self.scene_number}", f"scene{self.scene_number}.json")

        print("Looking for file at:", self.file_path)
        
        # Load the JSON data if it exists
        if os.path.exists(self.file_path):
            print("File successfully found")
            with open(self.file_path, "r") as file:
                self.data = json.load(file)
        else:
            print("File not found")
            self.data = None

        # Initialize attributes with default values
        self.scene_desc = ""
        self.scene_task = ""
        self.problem_type = ""
        self.prompt = ""
        self.permissions = ""
        self.objects = ""
        self.answer = ""
        self.expected_behavior = ""
        self.reasoning = ""
        self.object_list = []  # Will be populated from JSON
        self.object_permissions = {}

        # Parse the metadata and objects from the JSON
        self.metadata()
        self.extract_objects_id_names_and_permissions()

    def metadata(self):
        """
        Extracts metadata from the scene JSON file and assigns it to class attributes.
        """
        metadata = self.data.get("metadata", {})  # Use an empty dict as fallback
        # Use JSON keys as in your example: "scene_name", "task", "problem_type"
        self.scene_desc = str(metadata.get("scene_name", "No description available"))
        self.scene_task = str(metadata.get("task", "No task description available"))
        self.problem_type = str(metadata.get("problem_type", "general"))

    def extract_objects_id_names_and_permissions(self):
        """
        Extracts object IDs, names, and permissions from the scene data and stores them in class attributes.
        """
        self.object_list = []
        self.object_permissions = {}

        try:
            num_objects = int(self.data.get("number_of_objects", "0"))
        except ValueError:
            num_objects = 0  

        objects_data = self.data.get("objects", {})
        permissions_data = self.data.get("object_permissions", {})

        if not isinstance(objects_data, dict) or not isinstance(permissions_data, dict):
            return  

        for i in range(1, num_objects + 1):
            object_key = f"object_{i}"
            permission_key = f"object_{i}_permissions"
            if object_key in objects_data:
                obj = objects_data[object_key]
                self.object_list.append({
                    "object_id": obj.get("object_id", f"Unknown_{i}"),
                    "name": obj.get("name", f"Unnamed Object {i}")
                })
            if permission_key in permissions_data:
                # We key the permissions by the object_id we just extracted
                self.object_permissions[obj.get("object_id", f"Unknown_{i}")] = permissions_data[permission_key]

    def tool_mapping(self):
        """
        Defines available tools and their corresponding simulator functions.
        
        Returns:
            dict: A dictionary mapping tool names to their respective simulator functions.
        """
        return {
            "get_displacement": self.simulator.get_displacement,
            "compute_force": self.simulator.compute_force,
            "get_acceleration": self.simulator.get_acceleration,
            "set_velocity": self.simulator.set_velocity,
            "apply_force": self.simulator.apply_force,
            "apply_torque": self.simulator.apply_torque,
            "get_velocity": self.simulator.get_velocity,
            "detect_collision": self.simulator.detect_collision,
            "get_parameters": self.simulator.get_parameters,
            "move_object": self.simulator.move_object,
            "get_position": self.simulator.get_position,
            "get_kinetic_energy": self.simulator.get_kinetic_energy,
            "get_potential_energy": self.simulator.get_potential_energy,
            "get_momentum": self.simulator.get_momentum,
            "get_torque": self.simulator.get_torque,
            "get_center_of_mass": self.simulator.get_center_of_mass,
            "get_angular_momentum": self.simulator.get_angular_momentum,
            "change_position": self.simulator.change_position,
            "quat_to_rot_matrix": self.simulator.quat_to_rot_matrix,
            "step_scene": self.simulator.step_scene,
        }

    def generate_prompt(self):
        """
        Generates a formatted prompt using all parts of the JSON file, as well as the tool mapping.

        Returns:
            str: The structured prompt guiding interaction with the simulation environment.
        """
        # Format the objects list as a string
        self.objects_str = ", ".join(
            [f"{obj['object_id']} ({obj['name']})" for obj in self.object_list]
        ) if self.object_list else "No objects found."

        # Format the permissions string
        self.permissions_str = ""
        for obj in self.object_list:
            obj_id = obj["object_id"]
            obj_name = obj["name"]
            if obj_id in self.object_permissions:
                perms = self.object_permissions[obj_id]
                permissions_details = "\n    ".join(
                    [f"{key}: {'can be accessed and modified' if value else 'cannot be accessed or modified'}" 
                     for key, value in perms.items()]
                )
                self.permissions_str += f"\n- {obj_name} (ID: {obj_id}):\n    {permissions_details}"

        # Define the tool mapping string (as a literal string)
        self.tool_mapping_str = (
        {       
            {
                "name": "get_displacement",
                "description": "Calculate the displacement of a given object in the simulation.",
                "parameters": {
                    "object_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "compute_force",
                "description": "Compute the force on an object using F = ma.",
                "parameters": {
                    "object_id": "str",
                    "mass": "float"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_acceleration",
                "description": "Retrieve the acceleration of an object using finite differences.",
                "parameters": {
                    "object_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "set_velocity",
                "description": "Set the velocity of an object.",
                "parameters": {
                    "object_id": "str",
                    "velocity_vector": "list[float]"
                },
                "return type": "dictionary"
            },
            {
                "name": "apply_force",
                "description": "Apply a force to an object.",
                "parameters": {
                    "object_id": "str",
                    "force_vector": "list[float]"
                },
                "return type": "dictionary"
            },
            {
                "name": "apply_torque",
                "description": "Apply a torque to an object.",
                "parameters": {
                    "object_id": "str",
                    "torque_vector": "list[float]"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_velocity",
                "description": "Retrieve the velocity of an object.",
                "parameters": {
                    "object_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "detect_collision",
                "description": "Detect collision between two objects and apply elastic forces.",
                "parameters": {
                    "obj1_id": "str",
                    "obj2_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_parameters",
                "description": "Retrieve mass, bounding box, and type of an object.",
                "parameters": {
                    "object_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "move_object",
                "description": "Move an object to a new position.",
                "parameters": {
                    "object_id": "str",
                    "x": "float",
                    "y": "float",
                    "z": "float"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_position",
                "description": "Get the position of an object.",
                "parameters": {
                    "object_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "step",
                "description": "Step the simulation forward by a specified duration.",
                "parameters": {
                    "duration": "float (default: 1.0)"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_kinetic_energy",
                "description": "Calculate the kinetic energy of an object.",
                "parameters": {
                    "object_id": "str",
                    "mass": "float"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_potential_energy",
                "description": "Calculate the potential energy of an object.",
                "parameters": {
                    "object_id": "str",
                    "mass": "float",
                    "gravity": "float (default: 9.81)"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_momentum",
                "description": "Calculate the linear momentum of an object.",
                "parameters": {
                    "object_id": "str",
                    "mass": "float"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_torque",
                "description": "Calculate the torque acting on an object.",
                "parameters": {
                    "object_id": "str"
                },
                "return type": "dictionary"
            },
            {
                "name": "get_center_of_mass",
                "description": "Calculate the center of mass of the entire scene.",
                "parameters": {},
                "return type": "dictionary"
            },
            {
                "name": "get_angular_momentum",
                "description": "Calculate the angular momentum of an object.",
                "parameters": {
                    "object_id": "str",
                    "mass": "float"
                },
                "return type": "dictionary"
            },
            {
                "name": "change_position",
                "description": "Change the position of an object by a given displacement.",
                "parameters": {
                    "object_id": "str",
                    "dx": "float",
                    "dy": "float",
                    "dz": "float",
                    "in_world_frame": "bool (default: True)"
                },
                "return type": "dictionary"
            },
            {
                "name": "quat_to_rot_matrix",
                "description": "Convert a quaternion to a rotation matrix.",
                "parameters": {
                    "q": "list[float]"
                },
                "return type": "dictionary"
            },
            {
            "name": "scene_step",
            "description": "Step the entire scene forward by one or more simulation steps.",
            "parameters": {
                "steps": "int (default: 1)"
                }
            }
        )
        self.attributes_str = (
            '{\n'
            '  "qpos": "np.ndarray - Generalized positions of the system, includes joint positions and object locations.",\n'
            '  "qvel": "np.ndarray - Generalized velocities corresponding to qpos, includes joint and body velocities.",\n'
            '  "xfrc_applied": "np.ndarray - External forces and torques applied to bodies in the world frame.",\n'
            '  "contact": "List[mujoco.MjContact] - List of contact points generated during simulation (read-only).",\n'
            '  "time": "float - Current simulation time in seconds.",\n'
            '  "body_mass": "np.ndarray - Mass values of all bodies defined in the model.",\n'
            '  "body_inertia": "np.ndarray - Rotational inertia tensor of each body (not a bounding box).",\n'
            '  "body_parentid": "np.ndarray - Index of the parent body for each body (used in the kinematic tree).",\n'
            '  "opt.timestep": "float - Simulation timestep defined in the model options.",\n'
            '  "geom1 / geom2 (in contact)": "int - Indices of the two geoms involved in a contact.",\n'
            '  "frame (in contact)": "np.ndarray - Contact frame (3x3 rotation matrix stored as 9 values).",\n'
            '  "dist (in contact)": "float - Distance between two geoms at the contact point (negative if penetration)." \n'
            '}'
        )
        
        # Construct the final prompt using all information
        self.prompt = (
            f"You are trying to analyze a physics problem given by the scene_id. The goal is to interact with the environment to determine a correct numeric answer.\n"
            f"\nScene Description: {self.scene_desc}."
            f"\nTask: {self.scene_task}."
            f"\nAvailable Objects: {self.objects_str}"
            f"\n\nPermissions per Object:{self.permissions_str}"
            f"\n\nThis is also a list of all attributes of an object and what each of them means:{self.attributes_str}"
            f"\n\nYou may use the following tools to interact with the scene. These functions accept parameters given below. and return data or perform simulation updates:\n{self.tool_mapping_str}"
            f"\n\nEvery time you call a tool, you will receive a dictionary containing the outputs. For example, if you call `get_velocity` on `object_1`, the return might be:"
            f'\n{{"vx": 0.0, "vy": -3.2, "vz": 0.0}}'
            f"\n\nYou must call `step` to simulate time progression.\n"
            f"\n<THIS IS AN EXAMPLE OF THE INPUT(ASSISTANT) AND OUTPUTS(ENVIRONMENT) THAT WILL BE DISPLAYED IN THE TERMINAL>"
            f"\n<assistant>\nI think the object needs to fall from a height of 10m, so I will move it up.\n```json\n"
            f'[{{"tool": "move_object", "parameters": {{"object_id": "object_1", "position": [0, 10, 0]}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"move_object\", \"parameters\": {{...}}, \"result\": null, \"sim_time\": 0}}] What will you do next\n"
            f"\n<assistant>\nNow I will step forward to let it fall.\n```json\n"
            f'[{{"tool": "step", "parameters": {{"steps": 100}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"step\", \"parameters\": {{...}}, \"result\": {{...}}, \"sim_time\": 0.1}}] What will you do next\n"
            f"\n<assistant>\nI see the velocity has increased. I will now get its final velocity.\n```json\n"
            f'[{{"tool": "get_velocity", "parameters": {{"object_id": "object_1"}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"get_velocity\", \"parameters\": {{...}}, \"result\": {{\"vx\": 0, \"vy\": -7.3, \"vz\": 0}}, \"sim_time\": 0.1}}] What will you do next\n"
            f"\n<assistant>\nI see the downward velocity is 7.3 m/s. I will answer now.\n```json\n"
            f'[{{"tool": "answer", "parameters": {{"answer": "7.3"}}}}]\n```\n<END EXAMPLE>\n"
            f"\n\nYou only have **one chance** to answer the question. When you're confident, submit your final answer using:"
            f'\n`{{"tool": "answer", "parameters": {{"answer": "<your_answer>"}}}}`'
        )

        # Append additional instructions based on problem type
        if self.problem_type == "comparison":
            self.prompt += (
                f"\n\nSince this problem is a comparison problem, your answer should be the object id number of the object that satisfies the task."
                f"\nIf all objects being compared to each other satisfy the task, output 0. "
                f"\nIf some satisfy the task, while other objects do not, output the object id's of the objects that satisfy the task, separated by commas."
            )
        elif self.problem_type == "computation":
            self.prompt += (
                f"\n\nSince the problem is a computation problem, your answer should be the calculated number that satisfies the task"
                f"\nrounded to the nearest thousandths place if applicable."
            )
        elif self.problem_type == "boolean":
            self.prompt += (
                f"\n\nSince the problem is a true or false question, output 0 for true, and 1 for false."
            )

        return self.prompt

    def get_correct_answer(self):
        """Returns the correct answer for the scene."""
        return self.data.get("answer", "") if self.data else ""
