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
        Initialize the Experimental class with a Scene ID.
        """
        self.max_iterations = max_iterations
        self.scene = Scene(scene_id)  # Scene takes scene_id as input
        self.simulator = Simulator(self.scene)  # Pass Scene object into Simulator constructor THIS IS THE OPPOSITE OF WHAT WE TALKED ABOUT
        self.agent = OpenAIAgent(api_key)

        # Bind instance methods from self.simulator
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

    def generate_tool_descriptions(self) -> str:
        """Generate a formatted description of all available tools."""
        return "\n".join([f"- {tool}: {func.__doc__}" for tool, func in self.tool_mapping.items()])

    def execute_tool_calls(self, tool_calls_json: str) -> List[Dict[str, Any]]:
        """Execute the provided tool calls and log the results."""
        tool_calls = json.loads(tool_calls_json)
        aggregated_results = []

        for call in tool_calls:
            tool = call['tool']
            params = call['parameters']
            result = None

            try:
                if tool in self.tool_mapping:  # Access instance-bound methods
                    func = self.tool_mapping[tool]
                    result = func(**params)  # Now calls methods directly
                else:
                    raise ValueError(f"Unknown tool '{tool}'")
            except Exception as e:
                logging.error(f"Exception during '{tool}': {str(e)}")
                result = {"error": str(e)}

            aggregated_results.append({
                "tool": tool,
                "parameters": params,
                "result": result,
                "sim_time": self.simulator.time  # Logging current timestep
            })

        return aggregated_results

    def extract_json_response(self, llm_output: str) -> str:
        """Extract JSON response from LLM's output."""
        try:
            json_start = llm_output.index("{")
            json_part = llm_output[json_start:]
            json_obj = json.loads(json_part)
            return json.dumps(json_obj.get("response", {}))  # Use .get() to avoid KeyError
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")

    def run_experiment(self) -> Dict[str, Any]:
        """Run the experiment using the simulator and AI agent."""
        self.simulator.reset_sim()  # Reset the simulation
        correct_answer_found = False
        timeout_occurred = False

        # Get prompt and tool descriptions from the Scene
        scene_prompt = self.scene.get_prompt()
        tool_descriptions = self.generate_tool_descriptions() #FIX THIS WILL NOW BE DONE IN SCENE.PY SO NO NEED
        full_prompt = f"{scene_prompt}\n\nAvailable Tools:\n{tool_descriptions}"

        results = []
        for itr in range(self.max_iterations):
            user_input = f"{full_prompt}\nPrevious Results: {json.dumps(results, indent=2)}\nWhat should I do next?" #FIX WE ONLY WANT FULL PROMPT THE FRIST TIME AROUND
            llm_response = self.agent.interact(user_input)

            try:
                tool_calls_json = self.extract_json_response(llm_response)
            except ValueError as e:
                logging.error(f"Error extracting JSON: {e}")
                # DONT BREAK RETURN THIS ERROR TO LLM
                break

            logging.info(f"\n=== Executing Tool Calls (Iteration {itr + 1}) ===")

            # Check if an answer was provided
            for call in results: #DONT CHECK RESULTS CHECK TOOL_CALLS_JSON
                if call['tool'] == 'answer':
                    final_answer = call['parameters'].get('answer')  # Use .get() to avoid KeyError
                    if final_answer is not None:
                        correct_answer = self.scene.get_correct_answer()
                        correct_answer_found = str(final_answer).strip().lower() == str(correct_answer).strip().lower()
                    break  # Stop immediately if 'answer' tool was used

            if correct_answer_found:
                break  # Stop looping if we found a correct answer
                
            #I MOVED THIS TO ONLY EXECUTE IF THERE IS NO ANSWER
            results = self.execute_tool_calls(tool_calls_json)
            #MAKE THE NEXT PROMPT DOWN HERE
        
        #THIS ELSE HAS NO IF??
        else:  # No break if answer not found
            timeout_occurred = True

        # Return the results of the experiment
        experiment_results = {
            'correct': correct_answer_found,
            'timeout': timeout_occurred,
            'num_tool_calls': len(results), #WRONG NUMBER
            'iterations': len(results) if not timeout_occurred else self.max_iterations #WRONG NUMBER
        }
        return experiment_results

