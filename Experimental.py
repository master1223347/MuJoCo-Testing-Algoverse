import json
from dotenv import load_dotenv
import os
from simulator import Simulator
from scene import Scene
from openai_agent import OpenAIAgent
from typing import Any, Dict, List
import logging

# Load environment variables from the .env file
load_dotenv()

# API Key loaded once globally
api_key = os.getenv('OPENAI_API_KEY')

# Tool mapping only for existing simulator methods
tool_mapping = {
    "step": Simulator.step,
    "run_steps": Simulator.run_steps,
    "reset": Simulator.reset,
    "pause": Simulator.pause,
    "resume": Simulator.resume,
    "save_state": Simulator.save_state,
    "load_state": Simulator.load_state,
    "get_actions": Simulator.get_actions,
    "get_metadata": Simulator.get_metadata,
    "get_parameters": Simulator.get_parameters,
    "get_observation": Simulator.get_observation,
    "enable_logging": Simulator.enable_logging,
    "disable_logging": Simulator.disable_logging,
    "get_logs": Simulator.get_logs,
    "set_parameters": Simulator.set_parameters,
    "load_config": Simulator.load_config,
    "save_config": Simulator.save_config,
    "render": Simulator.render,
    "export_results": Simulator.export_results,
}

# Function to generate tool descriptions
def generate_tool_descriptions() -> str:
    """Generate a formatted description of all available tools."""
    return "\n".join([f"- {tool}: {func.__doc__}" for tool, func in tool_mapping.items()])

class Experimental:
    def __init__(self, scene_id: str, max_iterations: int = 5):
        """
        Initialize the Experimental class with a Scene ID.
        """
        self.max_iterations = max_iterations
        self.scene = Scene(scene_id)  # Scene takes scene_id as input
        self.simulator = Simulator(self.scene)  # Pass Scene object into Simulator constructor
        self.agent = OpenAIAgent(api_key)

    def execute_tool_calls(self, tool_calls_json: str) -> List[Dict[str, Any]]:
        """Execute the provided tool calls and log the results."""
        tool_calls = json.loads(tool_calls_json)
        aggregated_results = []

        for call in tool_calls:
            tool = call['tool']
            params = call['parameters']

            try:
                if tool in tool_mapping:
                    func = tool_mapping[tool]
                    result = func(self.simulator, **params)  # Pass the simulator instance explicitly
                else:
                    raise ValueError(f"Unknown tool '{tool}'")
            except Exception as e:
                logging.error(f"Exception during '{tool}': {str(e)}")
                result = {"error": str(e)}

            aggregated_results.append({
                "tool": tool,
                "parameters": params,
                "result": result,
                "sim_time": self.simulator.time  # logging current timestep (from tool execution)
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
        self.simulator.reset()  # Reset the simulation
        correct_answer_found = False
        timeout_occurred = False

        # Get prompt and tool descriptions from the Scene
        scene_prompt = self.scene.get_prompt()
        tool_descriptions = generate_tool_descriptions()
        full_prompt = f"{scene_prompt}\n\nAvailable Tools:\n{tool_descriptions}"

        results = []
        for itr in range(self.max_iterations):
            # Generate user input prompt dynamically
            user_input = f"{full_prompt}\nPrevious Results: {json.dumps(results, indent=2)}\nWhat should I do next?"

            llm_response = self.agent.interact(user_input)  # Ask AI agent for action

            # Extract the tool calls from the LLM's response
            tool_calls_json = self.extract_json_response(llm_response)

            # Execute tool calls and aggregate the results
            logging.info(f"\n=== Executing Tool Calls (Iteration {itr+1}) ===")
            results = self.execute_tool_calls(tool_calls_json)

            # Check for the correct answer in the results
            answer_found = False
            for call in results:
                if call['tool'] == 'answer':
                    final_answer = call['parameters']['answer']
                    correct_answer = self.scene.get_correct_answer()
                    correct_answer_found = (final_answer == correct_answer)
                    answer_found = True
                    break

            logging.info("\n=== Aggregated Results ===")
            logging.debug(json.dumps(results, indent=2))

            if answer_found:
                break

        else:  # No break if answer not found
            timeout_occurred = True

        # Return the results of the experiment
        experiment_results = {
            'correct': correct_answer_found,
            'timeout': timeout_occurred,
            'num_tool_calls': len(results),
            'iterations': len(results) if not timeout_occurred else self.max_iterations
        }
        return experiment_results





def generate_prompt.self
self.scene.prompt = scene.prompt


#utkarsh tooling-convention
