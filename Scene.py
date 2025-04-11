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
            '{\n'
            '    "get_displacement": "Computes the displacement of an object from its initial position.",\n'
            '    "compute_force": "Computes the net force applied to an object using physical parameters.",\n'
            '    "get_acceleration": "Retrieves the linear acceleration of a specified object.",\n'
            '    "set_velocity": "Sets a new velocity for a specified object.",\n'
            '    "apply_force": "Applies a force to a specified object in the scene.",\n'
            '    "apply_torque": "Applies a torque (rotational force) to a specified object.",\n'
            '    "get_velocity": "Retrieves the velocity of a specified object.",\n'
            '    "detect_collision": "Detects whether two or more objects have collided.",\n'
            '    "set_permissions": "Defines or updates access and modification permissions for scene objects.",\n'
            '    "get_parameters": "Retrieves physical parameters of an object (e.g., mass, inertia).",\n'
            '    "move_object": "Moves an object to a specified position within the simulation.",\n'
            '    "get_position": "Retrieves the current position of an object.",\n'
            '    "get_kinetic_energy": "Calculates the kinetic energy of an object.",\n'
            '    "get_potential_energy": "Calculates the potential energy of an object in the simulation.",\n'
            '    "get_momentum": "Calculates the linear momentum of an object.",\n'
            '    "get_torque": "Retrieves the net torque acting on an object.",\n'
            '    "get_center_of_mass": "Returns the center of mass position of the object.",\n'
            '    "get_angular_momentum": "Computes the angular momentum of an object.",\n'
            '    "change_position": "Applies a positional offset to an object along x, y, z directions.",\n'
            '    "quat_to_rot_matrix": "Converts a quaternion rotation into a 3x3 rotation matrix.",\n'
            '}'
        )
        
        # Create a string for the expected experiment results format
        self.exp_results_format = (
            f"\n\n### Expected Experiment Results Format"
            f"\nThe results of each experiment should be structured as follows:"
            f"\n```json"
            f"\n{{"
            f'\n  "correct": <boolean>,  # Whether the correct answer was found'
            f'\n  "timeout": <boolean>,  # Whether the experiment timed out after max iterations'
            f'\n  "num_tool_calls": <integer>,  # Total number of tool calls made'
            f'\n  "iterations": <integer>  # Total iterations performed'
            f"\n}}"
            f"\n```"
            f"\n\nTo process and evaluate the experiment results, you should call back to the `run_experiments` function in the `Experimental` class."
            f"\nThis function utilizes a dictionary to track and analyze the experiment's progress."
        )

        # Construct the final prompt using all information
        self.prompt = (
            f"You are trying to analyze a physics problem given by the scene_id. Your goal is to interact with the environment and figure out the correct answer. The answers are either integers or float values."
            f"\n\nThe scene's description is: {self.scene_desc}."
            f"\nYour task is defined as: {self.scene_task}."
            f"\nThese are the object IDs and object names found in the scene: {self.objects_str}"
            f"\n\nEach object has specific permissions defining whether its properties can be accessed or modified. Here are the permissions for each object:{self.permissions_str}"
            f"\n\nThese are the different tools and functions you can use to interact with the environment/scene rendered in MuJoCo:\n{self.tool_mapping_str}"
            f"\n\nThe names of the tools above correspond to functions in the simulator class that you can call when interacting with the environment."
            f"\n\nHowever, you might encounter an error if you call a function requiring permissions to modify objects that you are not allowed to access."
            f"\n\nThe environment will be set up by rendering an XML file within MuJoCo - your goal is to change different attributes of certain objects to figure out how to answer the problem."
            f"\n\nFor example, if you want to determine the final velocity of a sphere impacting the surface from a height of 10 units/meters, the scene will start with the sphere on the ground."
            f"\nYou will then move the sphere up 10 units using the `move_object` function from the simulator class and render it in MuJoCo, which will simulate the sphere falling from a height of 10 units."
            f"\n\nYou have **one attempt** to solve the problem. Once you input your answer, you will call the following functions from the Scene class:"
            f"\n- `scene.answer`"
            f"\n- `scene.expected_behavior`"
            f"\n- `scene.reasoning`"
            f"\n\n**Logging & Analysis:**"
            f"\n- Log all interactions with the environment, including time-step results and the functions used. NOTE: when calling functions, be sure to use the OBJECT ID as your parameter and NOT the OBJECT NAME!"
            f"\n- Provide reasoning for every interaction you apply to the environment AS YOU ARE RUNNING THROUGH YOUR ITERATIONS!!!"
            f"\n- Format all experimental data into a table or matrix for visualization."
            f"\n- Use Python libraries such as `matplotlib` and `scikit-learn` to generate plots or diagrams."
            f"\n{self.exp_results_format}"
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
