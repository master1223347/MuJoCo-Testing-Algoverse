import json
import os
import logging
from dotenv import load_dotenv
from simulator import Simulator
from scene import Scene
from openai_agent import OpenAIAgent
from typing import Any, Dict, List

# Load environment variables from the .env file
load_dotenv()

# API Key loaded once globally
api_key = os.getenv('OPENAI_API_KEY')

class Experimental:
    def __init__(self, scene_id: str, max_iterations: int = 5):
        """
        Initialize the Experimental class with a Scene ID and set up the necessary components.
        
        Args:
            scene_id (str): The unique identifier for the simulation scene.
            max_iterations (int): The maximum number of iterations allowed for the experiment (default is 5).
        """
        self.max_iterations = max_iterations
        self.simulator = Simulator()  # Create the Simulator object
        self.scene = Scene(scene_id, self.simulator)  # Initialize Scene with the simulator
        self.agent = OpenAIAgent(api_key)  # Initialize AI agent with the API key

        # Tool mapping to bind methods from the simulator to be called dynamically
        self.tool_mapping = {
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

    def execute_tool_calls(self, tool_calls_json: str) -> List[Dict[str, Any]]:
        """
        Execute the provided tool calls, log the results, and return them.

        Args:
            tool_calls_json (str): A JSON string representing the tool calls to be executed.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the results of each tool call, 
                                   including the tool name, parameters, result, and simulation time.
        """
        tool_calls = json.loads(tool_calls_json)  # Parse the JSON string into a list of tool calls
        aggregated_results = []  # Initialize a list to store the results of the tool calls

        for call in tool_calls:
            tool = call['tool']
            params = call['parameters']
            result = None

            try:
                # Attempt to find and execute the tool if it exists in the mapping
                if tool in self.tool_mapping:
                    func = self.tool_mapping[tool]
                    result = func(**params)  # Execute the function dynamically with the parameters
                else:
                    raise ValueError(f"Unknown tool '{tool}'")
            except Exception as e:
                # If an exception occurs during the execution of a tool, log it and return an error result
                logging.error(f"Exception during '{tool}': {str(e)}")
                result = {"error": str(e)}

            # Append the result, including the tool name, parameters, and result to the aggregated results
            aggregated_results.append({
                "tool": tool,
                "parameters": params,
                "result": result,
                "sim_time": self.simulator.time  # Record the simulation time during this call
            })

        return aggregated_results

    def extract_json_response(self, llm_output: str) -> str:
        """
        Extract a JSON response from the output of the LLM (Large Language Model).

        Args:
            llm_output (str): The raw output string from the LLM.

        Returns:
            str: A valid JSON string representing the response extracted from the LLM output.

        Raises:
            ValueError: If the LLM output is not in valid JSON format.
        """
        try:
            # Find the first occurrence of a JSON object and extract it
            json_start = llm_output.index("{")
            json_part = llm_output[json_start:]  # Extract JSON part from the output string
            json_obj = json.loads(json_part)  # Parse it into a dictionary
            return json.dumps(json_obj.get("response", {}))  # Return the "response" field from the JSON object
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")  # Raise an error if the JSON is invalid
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")  # Raise an error for unexpected issues

    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the experiment using the simulator and AI agent. This method orchestrates the experiment by 
        interacting with the simulation and utilizing the AI agent to decide the next steps.

        The loop will continue until the correct answer is found or the maximum number of iterations is reached.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the experiment, including whether the 
                             correct answer was found, if a timeout occurred, the number of tool calls made, 
                             and the number of iterations performed.
        """
        self.simulator.reset_sim()  # Reset the simulator to its initial state
        correct_answer_found = False  # Flag to track if the correct answer was found
        timeout_occurred = False  # Flag to track if the maximum number of iterations was reached

        # Get the scene prompt and tool descriptions from the Scene object
        scene_prompt = self.scene.get_prompt()
        tool_descriptions = self.scene.generate_tool_descriptions()
        full_prompt = f"{scene_prompt}\n\nAvailable Tools:\n{tool_descriptions}"  # Construct the full prompt

        results = []  # List to store the results of tool calls during each iteration
        num_tool_calls = 0  # Counter to track the number of tool calls made during the experiment
        
        for itr in range(self.max_iterations):
            # Construct the prompt for the LLM, either including previous results or starting fresh
            if itr == 0:
                llm_response = self.agent.interact(f"{full_prompt}\nWhat should I do next?")
            else:
                llm_response = self.agent.interact(f"{scene_prompt}\nPrevious Results: {json.dumps(results, indent=2)}\nWhat should I do next?")

            try:
                # Extract JSON tool calls from the LLM response
                tool_calls_json = self.extract_json_response(llm_response)
            except ValueError as e:
                logging.error(f"Error extracting JSON: {e}")
                continue  # Skip this iteration and request new instructions from the LLM

            logging.info(f"\n=== Executing Tool Calls (Iteration {itr + 1}) ===")

            # Answer logic: Check if any tool call contains an answer and check if it's correct
            answer_found = False
            correct_answer_found = False
            for call in tool_calls_json:
                if call['tool'] == 'answer':
                    final_answer = call['parameters'].get('answer')  # Get the answer from parameters
                    correct_answer = self.scene.get_correct_answer()  # Retrieve correct answer from scene
                    
                    # Mark the answer as found
                    answer_found = True
                    correct_answer_found = final_answer.strip().lower() in correct_answer.strip().lower() if final_answer and correct_answer else False
                    break  # Stop the experiment as soon as we get an answer (whether correct or not)

            # If an answer is found (correct or not), exit the loop early
            if answer_found:
                break  # Stop looping once an answer is provided by the LLM

            # If no answer is found, execute the tool calls as planned
            if not answer_found:
                results = self.execute_tool_calls(tool_calls_json)  # Execute tool calls and get results
                num_tool_calls += len(results)  # Increment the tool call count after execution

        else:  # If the loop completes without finding the answer, set the timeout flag
            timeout_occurred = True

        # Return the results of the experiment, including whether the correct answer was found and other statistics
        experiment_results = {
            'correct': correct_answer_found,  # Whether the correct answer was found
            'timeout': timeout_occurred,  # Whether the experiment timed out after max iterations
            'num_tool_calls': num_tool_calls,  # Total number of tool calls made
            'iterations': itr + 1 if not timeout_occurred else self.max_iterations  # Total iterations performed
        }

        return experiment_results
