"""
Experiment Runner Script

This script iterates through a predefined list of scene IDs, runs experiments 
using the `Experimental` class, and aggregates the results. The results are 
stored in a dictionary and optionally saved to a JSON file.

Dependencies:
- openai_agent.py
- scene.py
- simulator.py
- experimental.py

Functions:
- main(): Initializes and runs experiments for each scene ID, stores results, 
  and saves them to a JSON file.

Usage:
Run the script directly to execute experiments and generate results.

Example:
$ python script.py

Output:
- Results for each scene are printed to the console.
- Aggregated results are saved in `aggregated_results.json`.

"""

import os
import json
# Assuming all files are in the same directory and correctly implemented
from OpenAIAgent import OpenAIAgent
from Scene import Scene
from Simulator import Simulator
from Experimental import Experimental

def main():
    """
    Executes experiments for predefined scene IDs, collects results, 
    and saves them to a JSON file.
    """
    
    # Predefined list of scene IDs to iterate through
    scene_ids = ["Scene_1"]  # Replace with actual scene IDs
    
    # Initialize a dictionary to store aggregated results
    aggregated_results = {}
    
    for scene_id in scene_ids:
      
        simulator = Simulator(scene_id)  # Initialize the simulator
        scene = Scene(scene_id, simulator)  # Initialize the scene with the simulator
    
        # Print the generated prompt to the terminal
        prompt = scene.generate_prompt()  # Generate the prompt from the Scene class
        print(prompt)  # This will print the prompt to the terminal
      
        # Initialize an Experimental object for the current scene
        experiment = Experimental(scene_id)
        
        # Run the experiment
        results = experiment.run_experiment()


        if results['answer_found']:
            print("\n=== Answer Summary ===")
            print(f"LLM's Answer: {results['llm_answer']}")
            print(f"Correct Answer: {results['correct_answer']}")
            print(f"Answer Correct: {results['correct']}")
        else:
            print("\nNo answer was provided by the LLM.")
        
        # Store the results in the aggregated_results dictionary
        aggregated_results[scene_id] = results
        
        # Optionally, print or log the results for each scene
        print(f"Results for Scene {scene_id}: {results}")
    
    # Optionally, save the aggregated results to a file
    with open("aggregated_results.json", "w") as file:
        json.dump(aggregated_results, file, indent=4)

if __name__ == "__main__":
    main()
