import os
import json
# Assuming all files are in the same directory and correctly implemented
from openai_agent import OpenAIAgent
from Scene import Scene
from Simulator import Simulator
from Experimental import Experimental



def main():
    # Predefined list of scene IDs to iterate through
    scene_ids = ["Scene1", "Scene2", "Scene3"]  # Replace with actual scene IDs
    
    # Initialize a dictionary to store aggregated results
    aggregated_results = {}
    
    for scene_id in scene_ids:
        # Initialize an Experimental object for the current scene
        experiment = Experimental(scene_id)
        
        # Run the experiment
        results = experiment.run_experiment()
        
        # Store the results in the aggregated_results dictionary
        aggregated_results[scene_id] = results
        
        # Optionally, print or log the results for each scene
        print(f"Results for Scene {scene_id}: {results}")
    
    # Optionally, save the aggregated results to a file
    with open("aggregated_results.json", "w") as file:
        json.dump(aggregated_results, file, indent=4)

if __name__ == "__main__":
    main()
