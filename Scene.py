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
        self.terminal = "utkarsh" #For setting it up, switch this to the following - robin, abhinav, or sid.

        # Get the absolute path of the current working directory through a bunch of if else statements.
        if (self.terminal == "utkarsh"):
            drive = os.path.splitdrive(os.getcwd())[0].lower().rstrip(":") + ":/"  # Ensure single ':'
        
            # Construct the correct base directory
            base_dir = os.path.join(drive, "Users", "inbox", "Algoverse")
        
            # Construct the full file path
            self.file_path = os.path.join(
                base_dir, "MuJoCo-Testing-Algoverse", "Scenes", 
                f"Scene{self.scene_number}", f"scene{self.scene_number}.json"
            )
        
            # Normalize the path: replace backslashes with forward slashes
            self.file_path = self.file_path.replace("\\", "/")

        else if(self.terminal == "robin"):
            
            base_dir = r"C:\Users\robin\OneDrive\Documents\Algoverse\MuJoCo-Testing-Algoverse\Scenes"

            self.file_path = os.path.join(base_dir, f"Scene{self.scene_number}", f"scene{self.scene_number}.json")

        else if(self.terminal == "abhinav"):

            base_dir = r"C:\Users\epicg\Algoverse\MuJoCo-Testing-Algoverse\Scenes"

            self.file_path = os.path.join(base_dir, f"Scene{self.scene_number}", f"scene{self.scene_number}.json")

        else if(self.terminal == "sid"):

            base_dir = r"C:\Users\siddh\OneDrive\Desktop\Algoverse\MuJoCo-Testing-Algoverse\Scenes"

            self.file_path = os.path.join(base_dir, f"Scene{self.scene_number}", f"scene{self.scene_number}.json")
        
        # Debugging: Print the file path to verify correctness
        print("Looking for file at:", self.file_path)
        
        # Check if file exists
        if os.path.exists(self.file_path):
            print("File successfully found")
            
            # Load JSON data
            with open(self.file_path, "r") as file:
                self.data = json.load(file)
        else:
            print("Not successful")
            self.data = None

        self.scene_desc = ""
        self.scene_task = ""
        self.problem_type = ""
        self.prompt = ""
        self.permissions = ""
        self.objects = ""
        self.answer = ""
        self.expected_behavior = ""
        self.reasoning = ""
        self.object_list = []  # Reset object list before populating
        self.permissions = {}

    def metadata(self):
        """
        Extracts metadata from the scene JSON file and assigns it to class attributes.
        """
        metadata = self.data.get("metadata", {})  # Default to empty dict
        self.scene_desc = metadata.get("scene_desc", "")
        self.scene_task = metadata.get("scene_task", "")
        self.problem_type = metadata.get("problem_type", "")

    def tool_mapping(self):
        """
        Defines available tools and their corresponding simulator functions.
        
        Returns:
            dict: A dictionary mapping tool names to their respective simulator functions.
        """
        return {
            "apply_force": self.simulator.apply_force,
            "get_velocity": self.simulator.get_velocity,
            "detect_collision": self.simulator.detect_collision,
            "get_parameters": self.simulator.get_parameters,
            "move_object": self.simulator.move_object,
            "get_position": self.simulator.get_position,
            "reset_sim": self.simulator.reset_sim,
            "step": self.simulator.step,
            "load_scene": self.simulator.load_scene,
            "get_displacement": self.simulator.get_displacement,
            "compute_force": self.simulator.compute_force,
            "set_permissions": self.simulator.set_permissions,
        }

    def extract_objects_id_names_and_permissions(self):
        """
        Extracts object IDs, names, and permissions from the scene data and stores them in class attributes.
        """
        self.object_list = []  # Reset object list
        self.object_permissions = {}  # Reset permissions

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
                    "object_id": obj.get("object_id", "Unknown"),
                    "name": obj.get("name", "Unnamed Object")
                })

            if permission_key in permissions_data:
                self.object_permissions[obj.get("object_id", "Unknown")] = permissions_data[permission_key]

    def object_permissions_for_sim(self):
        """
        Sets object permissions in the simulator.
        """
        self.permissions = self.data.get("permissions")
        self.simulator.set_permissions(self.permissions)

    def answer(self):
        """
        Retrieves the answer dataset of the scene.
        
        Returns:
            any: The answer dataset, or None if not available.
        """
        return self.data.get("answer") if self.data else None

    def expected_behavior(self):
        """
        Retrieves the expected behavior dataset of the scene.
        
        Returns:
            any: The expected behavior dataset, or None if not available.
        """
        return self.data.get("expected_behavior") if self.data else None

    def reasoning(self):
        """
        Retrieves the reasoning dataset of the scene.
        
        Returns:
            any: The reasoning dataset, or None if not available.
        """
        return self.data.get("reasoning") if self.data else None

    def generate_prompt(self):
        """
        Generates a formatted prompt using all parts of the JSON file, as well as the tool mapping.

        Returns:
            str: The structured prompt guiding interaction with the simulation environment.
        """
    
        # Format object list as a string
        self.objects_str = ", ".join(
            [f"{obj['object_id']} ({obj['name']})" for obj in self.object_list]
        ) if self.object_list else "No objects found."

        # Format permissions (Ensure every permission is fully listed)
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

        self.tool_mapping_str = (
        '{\n'
        '    "apply_force": "Applies a force to a specified object in the scene.",\n'
        '    "get_velocity": "Retrieves the velocity of a specified object.",\n'
        '    "detect_collision": "Detects whether two or more objects have collided.",\n'
        '    "get_parameters": "Retrieves physical parameters of an object (e.g., mass, inertia).",\n'
        '    "move_object": "Moves an object to a specified position within the simulation.",\n'
        '    "get_position": "Retrieves the current position of an object.",\n'
        '    "reset_sim": "Resets the simulation environment to its initial state.",\n'
        '    "step": "Advances the simulation by a single time step.",\n'
        '    "load_scene": "Loads a predefined scene setup into the simulation.",\n'
        '    "get_displacement": "Computes the displacement of an object from its initial position.",\n'
        '    "compute_force": "Computes the force applied to an object based on its motion parameters.",\n'
        '    "set_permissions": "Defines or updates access and modification permissions for scene objects."\n'
        '}'
    )

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

        self.prompt = (
            f"You are trying to analyze a physics problem that is given when a scene_number is entered. Your goal is to interact with the environment and figure out the correct answer. The answers are either integers or float values."
            f"\n\nThe scene's description is: {self.scene_desc}."
            f"\nYour task is defined as: {self.scene_task}."
            f"\nThese are the object IDs and object names found in the scene: {self.objects_str}"
            f"\n\nEach object has specific permissions defining whether its properties can be accessed or modified. "
            f"Here are the permissions for each object:{self.permissions_str}"
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
            f"\n- Log all interactions with the environment, including time-step results and the functions used."
            f"\n- Provide reasoning for every interaction you apply to the environment."
            f"\n- Format all experimental data into a table or matrix for visualization."
            f"\n- Use Python libraries such as `matplotlib` and `scikit-learn` to generate plots or diagrams."
            f"\n{self.exp_results_format}"
        )
    
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
