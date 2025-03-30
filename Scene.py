import json
import os
import numpy as np

class Scene:
    def __init__(self, scene_id: str, simulator: Simulator):
        self.scene_id = scene_id
        self.simulator = simulator  # Store the simulator object
        self.scene_number = int(scene_id.split("_")[-1])

        # Get the absolute path of the current working directory
        try:
            drive = os.path.splitdrive(os.getcwd())[0].lower().rstrip(":") + ":/"  # Ensure single ':'
        except Exception as e:
            print(f"Warning! {e}. Retrying Scene Class . . . ")
            scene = Scene(self, scene_id: str, simulator: Simulator)
        
        # Construct the correct base directory
        base_dir = os.path.join(drive, "Users", "inbox", "Algoverse")
        
        # Construct the full file path
        self.file_path = os.path.join(
            base_dir, "MuJoCo-Testing-Algoverse", "Scenes", 
            f"Scene{self.scene_number}", f"scene{self.scene_number}.json"
        )
        
        # Normalize the path: replace backslashes with forward slashes
        self.file_path = self.file_path.replace("\\", "/")
        
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

    def generate_tool_descriptions(self) -> str:
        """Generate a formatted description of all available tools."""
        return "\n".join([f"- {tool}: {func.__doc__}" for tool, func in self.simulator.tool_mapping.items()])

    def metadata(self):
        """Defines the data found in the metadata array"""
        metadata = self.data.get("metadata", {})  # Default to empty dict
        self.scene_desc = metadata.get("scene_desc", "")
        self.scene_task = metadata.get("scene_task", "")
        self.problem_type = metadata.get("problem_type", "")

    def tool_mapping(self):
        """Defines available tools and their corresponding simulator functions."""
        return {
            "render": self.simulator.render,
            "apply_force": self.simulator.apply_force,
            "get_velocity": self.simulator.get_velocity,
            "detect_collision": self.simulator.detect_collision,
            "get_parameters": self.simulator.get_parameters,
            "move_object": self.simulator.move_object,
            "get_position": self.simulator.get_position,
            "reset_sim": self.simulator.reset_sim,
            "step": self.simulator.step,
            "load_scene": self.simulator.load_scene,
        }

    def extract_objects_id_names_and_permissions(self):
        """Extracts objects and their permissions."""
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
        self.permissions = self.data.get("permissions")
        self.simulator.set_permissions(self.permissions)

    def answer(self):
        """Returns the answer dataset of the scene."""
        return self.data.get("answer") if self.data else None

    def expected_behavior(self):
        """Returns the expected_behavior dataset of the scene."""
        return self.data.get("expected_behavior") if self.data else None

    def reasoning(self):
        """Returns the reasoning dataset of the scene."""
        return self.data.get("reasoning") if self.data else None
        
    def generate_prompt(self):
    """Generates a formatted prompt using all parts of the JSON file, as well as the tool mapping."""

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

    # Generate tool descriptions dynamically
    self.tool_mapping_str = self.generate_tool_descriptions()

    self.prompt = (
        f"You are trying to analyze a physics problem that is given when a scene_number is entered. Your goal is to interact with the environment and figure out the correct answer. The answers are either integers or float values."
        f"\n\nThe scene's description is of the following: {self.scene_desc}."
        f"Your task is defined as: {self.scene_task}."
        f"These are the object IDs and object names found in the scene: {self.objects_str}"
        f"\n\nEach object has specific permissions defining whether its properties can be accessed or modified. "
        f"Here are the permissions for each object:{self.permissions_str}"
        f"\n\nThese are the different tools and functions you can use to interact with the environment/scene rendered in MuJoCo:\n{self.tool_mapping_str}"
        f"\n\nThe names of the tools above are functions that are found in the simulator class that you can call back when interacting with the environment."
        f"\n\nHowever, you might get an error if you call back a function to interact with the environment if it requires accessing or modifying permissions of objects that you are not allowed to."
        f"\n\nThe environment will be set up by rendering an XML file within MuJoCo - your goal is to change different attributes of certain objects to figure out how to answer the problem."
        f"\n\nFor example, say you want to figure out the final velocity of a sphere when impacting the surface at a height of 10 units/meters. The scene will start off with the sphere on the ground"
        f"\nand you will move the sphere up 10 units using the move_object function from the simulator class, and render it in MuJoCo, in which it would show the sphere falling from 10 units above the surface."
        f"\n\nYou will be given 1 attempt to solve the problem, and once you input the answer, you will then call back to the answer, expected behavior, and reasoning functions of the scene class, defined as"
        f"\n scene.answer, scene.expected_behavior, and scene.reasoning"
        f"\n\nRemember to log all your interactions with the environment, the results of the interaction per time step of the simulator, the functions you called back on from all classes, as well as your reasoning"
        f"\nfor the different interactions you apply to the environment. Format all data from every experiment run you do into a table or matrix so that it can be visualized into a plot or diagram of some sort"
        f"\nusing python libraries, such as matplotlib and sci-kit learn."
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
