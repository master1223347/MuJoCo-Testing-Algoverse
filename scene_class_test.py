import openai
import mujoco
import json
import os

class Scene:
    def __init__(self, scene_number, openai_api_key):
        self.scene_number = scene_number
        self.xml_path = f"scene{scene_number}.xml"
        self.json_path = f"scene{scene_number}.json"
        self.openai_api_key = openai_api_key
        
        self.model = None
        self.data = None
        self.permissions = None
        
        openai.api_key = openai_api_key
        
        self.load_scene()

    def load_scene(self):
        """Loads the XML and JSON data for the scene."""
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML file {self.xml_path} not found.")

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file {self.json_path} not found.")

        # Load the MuJoCo model from XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Load permissions and settings from JSON
        with open(self.json_path, "r") as json_file:
            self.permissions = json.load(json_file)

    def get_permission(self, key):
        """Retrieves permission settings for the scene."""
        return self.permissions.get(key, None)

    def run_simulation(self, steps=1000):
        """Runs the MuJoCo simulation for a set number of steps."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        """Visualizes the MuJoCo simulation."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
    
    def generate_code(self, prompt, model="gpt-4", max_tokens=300, temperature=0.2):
        """Generates code based on a given physics-related prompt."""
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    scene_number = input("Enter scene number: ")
    openai_api_key = input("Enter OpenAI API key: ")  # Ensure API key is entered securely
    
    scene = Scene(scene_number, openai_api_key)
    print(f"Loaded scene {scene_number} with permissions: {scene.permissions}")
    
    physics_prompt = "Describe a physics problem to generate related code: "
    generated_code = scene.generate_code(input(physics_prompt))
    print("Generated Code:\n", generated_code)
    
    scene.render()
