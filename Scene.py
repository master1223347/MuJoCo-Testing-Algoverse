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
        self.terminal = "utkarsh"  # For setting it up, switch this to: "robin", "abhinav", or "sid".
        
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
                    [f"{key}: {'can be accessed' if value else 'cannot be accessed'}" 
                     for key, value in perms.items()]
                )
                self.permissions_str += f"\n- {obj_name} (ID: {obj_id}):\n    {permissions_details}"

        # Define the tool mapping string (as a literal string)
        tools = [
            {"name": "step", "description": "keeps on moving the simulator forward in time", "arguments": {"duration": "float"}, "return type": {"results" : None}},
            {"name": "apply_force", "description": "applies a force vector to an object", "arguments": {"object_id": "str", "force_vector": "list[float]"}, "return type": {"status": "str", "object_id": "int", "force": "list[float]"}},
            {"name": "get_velocity", "description": "retrieves the velocity vector of an object", "arguments": {"object_id": "str"}, "return type": {"velocity": "array"}},
            {"name": "detect_collision", "description": "checks if two objects have collided", "arguments": {"obj1_id": "str", "obj2_id": "str"}, "return type": {"collision_detected": "bool"}},
            {"name": "get_parameters", "description": "fetches physical parameters like mass, bounding box, and type", "arguments": {"object_id": "str"}, "return type": {"mass": "float", "bounding_box": "list[float]", "type": "int"}},
            {"name": "move_object", "description": "sets an object's position to a new coordinate", "arguments": {"object_id": "str", "x": "float", "y": "float", "z": "float"}, "return type": {"position": "tuple[float, float, float]"}},
            {"name": "get_position", "description": "gets the current position and time of an object", "arguments": {"object_id": "str"}, "return type": {"position": "tuple[float, float, float]", "time": "float"}},
            {"name": "get_displacement", "description": "gets how far an object has moved from its initial position", "arguments": {"object_id": "str"}, "return type": {"displacement": "float"}},
            {"name": "compute_force", "description": "calculates the force on an object using F = ma", "arguments": {"object_id": "str", "mass": "float"}, "return type": {"x": "float", "y": "float", "z": "float"}},
            {"name": "get_acceleration", "description": "returns the current acceleration vector of an object", "arguments": {"object_id": "str"}, "return type": {"x": "float", "y": "float", "z": "float"}},
            {"name": "set_velocity", "description": "sets the velocity vector of an object", "arguments": {"object_id": "str", "velocity_vector": "list[float]"}, "return type": {"status": "str", "object_id": "int", "velocity": "list[float]"}},
            {"name": "apply_torque", "description": "applies a torque to an object", "arguments": {"object_id": "str", "torque_vector": "list[float]"}, "return type": {"status": "str", "object_id": "int", "torque": "list[float]"}},
            {"name": "get_torque", "description": "returns the torque acting on an object", "arguments": {"object_id": "str"}, "return type": {"torque": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "get_center_of_mass", "description": "gets the center of mass of the entire scene", "arguments": {}, "return type": {"center_of_mass": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "get_angular_momentum", "description": "returns the angular momentum of an object", "arguments": {"object_id": "str", "mass": "float"}, "return type": {"angular_momentum": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "change_position", "description": "translates an object by some delta in the local or world frame", "arguments": {"object_id": "str", "dx": "float", "dy": "float", "dz": "float", "in_world_frame": "bool"}, "return type": {"new_position": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "quat_to_rot_matrix", "description": "converts a quaternion into a 3x3 rotation matrix", "arguments": {"q": "list[float]"}, "return type": {"rotation_matrix": "array[3][3]"}},
            {"name": "answer", "description": "submits an answer back to the system for checking or logging", "arguments": {"answer": "str or float"}, "return type": {"acknowledged": "bool"}}
        ]

        tools_str = json.dumps(tools, indent=2)

        permission_explanations_dict = {
            "density": "This is the permission to figure out the density of the object.",
            "mass": "This is the permission to find out how much the object weighs.",
            "radius": "This lets you access the object's radius, which may affect rolling or collisions.",
            "type": "This is the permission to know what type of object this is (e.g., sphere, plane).",
            "name": "This lets you retrieve the name of the object.",
            "pos": "This allows access to the current position of the object.",
            "size": "This gives access to the object's bounding dimensions (like length, width, height).",
            "vel": "This allows access to the linear velocity of the object.",
            "acc": "This allows access to the linear acceleration of the object.",
            "rot": "This gives access to the object's orientation, usually in quaternion or rotation matrix form.",
            "angvel": "This gives access to the object's angular velocity (how fast it's spinning).",
            "angacc": "This gives access to the object's angular acceleration (how quickly its rotation is changing).",
            "inertia": "This describes the object's rotational inertia (resistance to changes in rotation).",
            "friction": "This gives access to how much the object resists sliding (static/dynamic friction coefficients).",
            "restitution": "This describes how bouncy the object is — how much energy it retains after collision.",
            "material": "This lets you access the material properties (e.g., metal, rubber) which affect physics and visuals.",
            "color": "This allows access to the object's visual appearance in terms of RGB color.",
            "texture": "This gives access to the texture applied to the object’s surface.",
            "contact": "This provides information about whether the object is in contact with another and details of that contact.",
            "geom": "This refers to the object's collision geometry — what shape MuJoCo uses to detect contact.",
            "joint": "This allows access to the joint configuration if the object is part of an articulated system.",
            "qpos": "This is the generalized position state — includes both location and joint states.",
            "qvel": "This is the generalized velocity state — includes both linear and angular motion.",
            "xfrc_applied": "This provides access to external forces and torques applied to the object.",
            "torque": "This gives access to the torque currently acting on the object.",
            "force": "This provides information about the total net force acting on the object.",
            "com": "This gives the position of the object’s center of mass in world coordinates.",
            "parentid": "This allows access to the object's parent in the scene hierarchy (for multi-body systems).",
            "childid": "This gives access to the object's child links or bodies, if any."
        }

        # Collect union of all permission keys from all objects
        used_permissions = set()
        for perms in self.object_permissions.values():
            used_permissions.update(perms.keys())

        # Output unique permission explanations once
        permission_explanations = "\n".join([
            f"  - {key}: {permission_explanations_dict.get(key, 'No description available.')}"
            for key in sorted(used_permissions)
        ])
        
        # Construct the final prompt using all information
        self.prompt = (
            f"You are trying to analyze a physics problem given by the scene_id. The goal is to interact with the environment to determine a correct numeric answer.\n"
            f"\nScene Description: {self.scene_desc}."
            f"\nTask: {self.scene_task}."
            f"\nAvailable Objects: {self.objects_str}"
            f"\n\nPermissions per Object:{self.permissions_str}"
            f"\n\nThis is also a list of all the description of an object's permissions and what each of them means:{permission_explanations}"
            f"\n\nYou may use the following tools along with their descriptionto interact with the scene. These functions accept parameters given below, and return data or perform simulation updates:\n{tools_str}"
            f"\n\nEvery time you call a tool, you will receive a dictionary containing the outputs. For example, if you call `get_velocity` on `object_1`, the return might be:"
            f'\n{{"vx": 0.0, "vy": -3.2, "vz": 0.0}}'
            f"\n\nYou must call `step` to simulate time progression.\n"
            f"\n<THIS IS AN EXAMPLE OF THE INPUT(ASSISTANT) AND OUTPUTS(ENVIRONMENT)>"
            f"\nProblem: You are given a ball and a ground surface for reference. Drop the ball from a height of 10 units and figure out the velocity of the object after 0.5 seconds."
            f"\n<assistant>\nI see that I have to move the ball up 10 units so I will do that.\n```json\n"
            f'[{{"tool": "move_object", "parameters": {{"object_id": "object_1", "x": 0, "y": 10, "z": 0}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"move_object\", \"parameters\": {{...}}, \"result\": {{\"position\": [0, 10, 0]}}, \"sim_time\": 0}}] What will you do next\n"
            f"\n<assistant>\nNow I will simulate by using the step function to go 0.5 seconds forward.\n```json\n"
            f'[{{"tool": "step", "parameters": {{"duration": 0.5}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"step\", \"parameters\": {{...}}, \"result\": null, \"sim_time\": 0.5}}] What will you do next\n"
            f"\n<assistant>\nNow I will use the get velocity function to figure out what I should output as my answer.\n```json\n"
            f'[{{"tool": "get_velocity", "parameters": {{"object_id": "object_1"}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"get_velocity\", \"parameters\": {{...}}, \"result\": {{\"velocity\": [0, -4.9, 0]}}, \"sim_time\": 0.5}}] What will you do next\n"
            f"\n<assistant>\nNow I will call back the answer.\n```json\n"
            f'[{{"tool": "answer", "parameters": {{"answer": "-4.9"}}}}]\n```\n<END EXAMPLE>\n"'
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

        self.prompt += (
            f'\n When you are trying to call functions, use a string that goes as object_{{object_id}} for the object id, and use the name of the function as the tool name.'
            f'\nRemember that YOU MUST PROVIDE YOUR REASONING FOR EVERY ACTION you take, and then make sure to add a valid JSON format of an array of tool calls.')

        return self.prompt

    def get_correct_answer(self):
        """Returns the correct answer for the scene."""
        return self.data.get("answer", "") if self.data else ""
    