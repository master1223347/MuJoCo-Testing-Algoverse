import json
from dotenv import load_dotenv
import os
from simulator import Simulator  # Import the Simulator class
from scene import Scene  # Import the Scene class
from openai_agent import OpenAIAgent  # Import the OpenAI Agent class
from typing import Any, Dict, List

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# Answer checking tool
def answer_tool(answer: Any, correct_answer: Any):
    if answer == correct_answer:
        return("[Answer Tool] ✅ Correct answer!")
    else:
        return(f"[Answer Tool] ❌ Incorrect answer! Got {answer}, expected {correct_answer}")

# Generalized execution of tool calls with try-except and timestep logging
def execute_tool_calls(sim: Simulator, tool_calls_json: str) -> List[Dict[str, Any]]:
    tool_calls = json.loads(tool_calls_json)
    aggregated_results = []

    tool_mapping = {
        # Core functionalities
        "get_state": sim.get_state,
        "set_state": sim.set_state,
        "step": sim.step,
        "run_steps": sim.run_steps,

        # Simulation control
        "reset": sim.reset,
        "pause": sim.pause,
        "resume": sim.resume,

        # State management
        "save_state": sim.save_state,
        "load_state": sim.load_state,
        "export_state": sim.export_state,

        # Information retrieval
        "get_actions": sim.get_actions,
        "get_metadata": sim.get_metadata,
        "get_parameters": sim.get_parameters,
        "get_observation": sim.get_observation,

        # Logging and Debugging
        "enable_logging": sim.enable_logging,
        "disable_logging": sim.disable_logging,
        "get_logs": sim.get_logs,

        # Configuration and Customization
        "set_parameters": sim.set_parameters,
        "load_config": sim.load_config,
        "save_config": sim.save_config,

        # Visualization and Export
        "render": sim.render,
        "export_results": sim.export_results,
    }

    for call in tool_calls:
        tool = call['tool']
        params = call['parameters']

        result = None
        try:
            if tool in tool_mapping:
                func = tool_mapping[tool]
                result = func(**params)
            elif tool == "answer":
                result = {"answer_provided": params["answer"]}
            else:
                raise ValueError(f"Unknown tool '{tool}'")
        except Exception as e:
            print(f"[Error] Exception during '{tool}': {str(e)}")
            result = {"error": str(e)}

        aggregated_results.append({
            "tool": tool,
            "parameters": params,
            "result": result,
            "sim_time": sim.time  # logging current timestep (from tool execution)
        })

    return aggregated_results

# JSON extraction helper from your desired LLM output format
def extract_json_response(llm_output: str) -> str:
    try:
        json_start = llm_output.index("{")
        json_part = llm_output[json_start:]
        json_obj = json.loads(json_part)
        return json.dumps(json_obj["response"])
    except Exception as e:
        raise ValueError(f"Invalid response format: {e}")

# Experimental Class
class Experimental:
    def __init__(self, scene_id, max_iterations=5):
        """
        Initialize the Experimental class with a Scene ID.

        Parameters:
        - scene_id: str, Identifier for the scene to load in the simulator.
        - max_iterations: int, Maximum number of iterations to run the experiment.
        """
        api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
        self.max_iterations = max_iterations
        # Create the Scene and Simulator objects
        self.scene = Scene(scene_id)  # Scene takes scene_id as input
        self.simulator = Simulator(self.scene)  # Pass Scene object into Simulator constructor
        self.agent = OpenAIAgent(api_key)

    import json
from dotenv import load_dotenv
import os
from simulator import Simulator  # Import the Simulator class
from scene import Scene  # Import the Scene class
from openai_agent import OpenAIAgent  # Import the OpenAI Agent class
from typing import Any, Dict, List

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# Answer checking tool
def answer_tool(answer: Any, correct_answer: Any):
    if answer == correct_answer:
        return("[Answer Tool] ✅ Correct answer!")
    else:
        return(f"[Answer Tool] ❌ Incorrect answer! Got {answer}, expected {correct_answer}")

# Generalized execution of tool calls with try-except and timestep logging
def execute_tool_calls(sim: Simulator, tool_calls_json: str) -> List[Dict[str, Any]]:
    tool_calls = json.loads(tool_calls_json)
    aggregated_results = []

    tool_mapping = {
        # Core functionalities
        "get_state": sim.get_state,
        "set_state": sim.set_state,
        "step": sim.step,
        "run_steps": sim.run_steps,

        # Simulation control
        "reset": sim.reset,
        "pause": sim.pause,
        "resume": sim.resume,

        # State management
        "save_state": sim.save_state,
        "load_state": sim.load_state,
        "export_state": sim.export_state,

        # Information retrieval
        "get_actions": sim.get_actions,
        "get_metadata": sim.get_metadata,
        "get_parameters": sim.get_parameters,
        "get_observation": sim.get_observation,

        # Logging and Debugging
        "enable_logging": sim.enable_logging,
        "disable_logging": sim.disable_logging,
        "get_logs": sim.get_logs,

        # Configuration and Customization
        "set_parameters": sim.set_parameters,
        "load_config": sim.load_config,
        "save_config": sim.save_config,

        # Visualization and Export
        "render": sim.render,
        "export_results": sim.export_results,
    }

    for call in tool_calls:
        tool = call['tool']
        params = call['parameters']

        result = None
        try:
            if tool in tool_mapping:
                func = tool_mapping[tool]
                result = func(**params)
            elif tool == "answer":
                result = {"answer_provided": params["answer"]}
            else:
                raise ValueError(f"Unknown tool '{tool}'")
        except Exception as e:
            print(f"[Error] Exception during '{tool}': {str(e)}")
            result = {"error": str(e)}

        aggregated_results.append({
            "tool": tool,
            "parameters": params,
            "result": result,
            "sim_time": sim.time  # logging current timestep (from tool execution)
        })

    return aggregated_results

# JSON extraction helper from your desired LLM output format
def extract_json_response(llm_output: str) -> str:
    try:
        json_start = llm_output.index("{")
        json_part = llm_output[json_start:]
        json_obj = json.loads(json_part)
        return json.dumps(json_obj["response"])
    except Exception as e:
        raise ValueError(f"Invalid response format: {e}")

# Experimental Class
class Experimental:
    def __init__(self, scene_id, max_iterations=5):
        """
        Initialize the Experimental class with a Scene ID.

        Parameters:
        - scene_id: str, Identifier for the scene to load in the simulator.
        - max_iterations: int, Maximum number of iterations to run the experiment.
        """
        api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
        self.max_iterations = max_iterations
        # Create the Scene and Simulator objects
        self.scene = Scene(scene_id)  # Scene takes scene_id as input
        self.simulator = Simulator(self.scene)  # Pass Scene object into Simulator constructor
        self.agent = OpenAIAgent(api_key)

    def run_experiment(self):
        """Run the experiment using the simulator and AI agent."""
        self.simulator.reset()  # Reset the simulation
        correct_answer_found = False
        iterations_count = 0
        timeout_occurred = False

        # Get prompt from Scene (per Dylan's feedback)
        scene_prompt = self.scene.get_prompt()

        # Describe all tools (per Dylan's feedback)
        tool_descriptions = "\n".join([f"- {tool}: {func.__doc__}" for tool, func in execute_tool_calls.__globals__["tool_mapping"].items()])
        full_prompt = f"{scene_prompt}\n\nAvailable Tools:\n{tool_descriptions}"

        # Run the experiment for a specified number of iterations
        for itr in range(self.max_iterations):
            state = self.simulator.get_state()  # Get current state of the simulator

            # Use aggregated results after first iteration
            if itr == 0:
                user_input = f"{full_prompt}\nCurrent state: {state}. What should I do next?"
            else:
                user_input = f"Previous Results: {json.dumps(results, indent=2)}\nWhat should I do next?"

            llm_response = self.agent.interact(user_input)  # Ask AI agent for action

            # Extract the tool calls from the LLM's response
            tool_calls_json = extract_json_response(llm_response)
            tool_calls = json.loads(tool_calls_json)

            # Check if answer is present before executing any tool calls
            for call in tool_calls:
                if call['tool'] == 'answer':
                    correct_answer_found = True
                    break

            if correct_answer_found:
                break

            # Execute tool calls and aggregate the results
            print("\n=== Executing Tool Calls ===")
            results = execute_tool_calls(self.simulator, tool_calls_json)

            iterations_count += 1

        if not correct_answer_found:
            timeout_occurred = True

        # Return the results of the experiment
        experiment_results = {
            'correct': correct_answer_found,
            'iterations': iterations_count,
            'timeout': timeout_occurred,
        }
        return experiment_results


# Example Usage
if __name__ == "__main__":
    scene_id = "scene_01"  # Replace with your scene ID
    experiment = Experimental(scene_id)
    results = experiment.run_experiment()

    # Output the experiment results
    print("\n=== Experiment Results ===")
    print(results)
    if results['correct']:
        print("[Answer Tool] ✅ Correct answer!")
    else:
        print("[Answer Tool] ❌ Incorrect answer.")

# Example Usage
if __name__ == "__main__":
    scene_id = "scene_01"  # Replace with your scene ID
    experiment = Experimental(scene_id)
    results = experiment.run_experiment()

    # Output the experiment results
    print("\n=== Experiment Results ===")
    print(results)
    if results['correct']:
        print("[Answer Tool] ✅ Correct answer!")
    else:
        print("[Answer Tool] ❌ Incorrect answer.")
