import json
import os

class Scene:
    def __init__(self):
        # Get scene number from user input
        self.scene_number = int(input("Enter the scene number: "))
        
        # Get the absolute path of the current working directory
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

    def metadata(self):
        """Returns the metadata of the scene."""
        return self.data.get("metadata") if self.data else None

    def objects(self):
        """Returns the objects data of the scene."""
        return self.data.get("objects") if self.data else None

    def answer(self):
        """Returns the answer dataset of the scene."""
        return self.data.get("answer") if self.data else None

    def expected_behavior(self):
        """Returns the expected_behavior dataset of the scene."""
        return self.data.get("expected_behavior") if self.data else None

    def reasoning(self):
        """Returns the reasoning dataset of the scene."""
        return self.data.get("reasoning") if self.data else None

# Example Usage
if __name__ == "__main__":
    scene = Scene()
    print("Metadata:", scene.metadata())
    print("Objects:", scene.objects())
    print("Answer:", scene.answer())
    print("Expected Behavior:", scene.expected_behavior())
    print("Reasoning:", scene.reasoning())
