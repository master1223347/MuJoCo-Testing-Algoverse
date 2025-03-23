import os
import json
import mujoco

class Scene:
    def __init__(self, scene_number):
        self.xml_path = r"C:\Users\robin\OneDrive\Documents\VSCode projects\Algoverse\Scenes\Scene1\xml\scene1.xml"
        self.json_path = r"C:\Users\robin\OneDrive\Documents\VSCode projects\Algoverse\Scenes\Scene1\json\scene1.json"

def find_existing_file(self, search_paths, filename):
    """Searches for the file in multiple possible directories and returns the correct path."""
    for path in search_paths:
        full_path = os.path.join(path, filename)
        print(f"Searching for {filename} in: {full_path}")  # Debugging print
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"❌ {filename} not found in any expected directory!")


    def load_xml(self):
        """Loads the MuJoCo XML model file."""
        return mujoco.MjModel.from_xml_path(self.xml_path)
        self.model = self.load_xml()

    def load_json(self):
        """Loads the JSON metadata for the scene (e.g., permissions, question prompts)."""
        with open(self.json_path, "r") as json_file:
            return json.load(json_file)
        self.permissions = self.load_json()

    def print_scene_info(self):
        """Prints scene metadata and model details for debugging."""
        print(f"✅ Scene {self.scene_number} Loaded Successfully!")
        print(f"📂 XML Path: {self.xml_path}")
        print(f"📂 JSON Path: {self.json_path}")
        print(f"🔐 Permissions: {self.permissions}")



if __name__ == "__main__":
    scene_number = input("What is the scene number? ")
    Scene(scene_number)
    scene_number.print_scene_info()
    
# Example Usage
# scene = Scene(5)  # Loads scene5.xml and scene5.json
# scene.print_scene_info()
