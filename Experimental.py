import json
from dotenv import load_dotenv
import os
from simulator import Simulator  # Import the Simulator class
from openai_agent import OpenAIAgent  # Import the OpenAI Agent class
from typing import Any, Dict, List

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')


# Answer checking tool
def answer_tool(answer: Any, correct_answer: Any):
    if answer == correct_answer:
        print("[Answer Tool] ✅ Correct answer!")
    else:
        print(f"[Answer Tool] ❌ Incorrect answer! Got {answer}, expected {correct_answer}")


# Generalized execution of tool calls with try-except and timestep logging
def execute_tool_calls(sim: Simulator, tool_calls_json: str) -> List[Dict[str, Any]]:
    tool_calls = json.loads(tool_calls_json)
    aggregated_results = []

    tool_mapping = {
        "get_state": sim.get_state,
        "set_state": sim.set_state,
        "step": sim.step,
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


# Example Usage
if __name__ == "__main__":
    simulator = Simulator()

    # Example simulated LLM response with chain-of-thought reasoning
    llm_response = """
    I think, after setting the ball to rolling and advancing 3 seconds, 
    the ball will still be rolling.

    my answer is:
    {
        "response": [
            {"tool": "set_state", "parameters": {"object": "ball", "state": "rolling"}},
            {"tool": "step", "parameters": {"seconds": 3}},
            {"tool": "get_state", "parameters": {"object": "ball"}},
            {"tool": "answer", "parameters": {"answer": "rolling"}}
        ]
    }
    """

    # Extract and execute JSON tool calls
    tool_calls_json = extract_json_response(llm_response)
    print("\n=== Executing Tool Calls ===")
    results = execute_tool_calls(simulator, tool_calls_json)

    # Aggregated results with simulator timestep logged
    print("\n=== Aggregated Results ===")
    print(json.dumps(results, indent=2))

    # Evaluate final answer correctness
    final_answer = results[-1]['parameters']['answer']
    correct_answer = "rolling"
    answer_tool(final_answer, correct_answer)


# Experimental Class
class Experimental:
    def __init__(self, scene_id):
        """
        Initialize the Experimental class with a Scene ID.

        Parameters:
        - scene_id: str, Identifier for the scene to load in the simulator.
        """
        api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
        self.simulator = Simulator()  # Initialize the simulator
        self.agent = OpenAIAgent(api_key)  # Initialize OpenAI agent with API key
        self.scene_id = scene_id

        # Load the scene based on the scene_id
        self.load_scene(self.scene_id)

    def load_scene(self, scene_id):
        """Load a specific scene into the simulator based on the scene ID."""
        # Assuming the simulator has a method to load a scene by its ID
        print(f"Loading scene with ID: {scene_id}")
        self.simulator.load_scene(scene_id)  # This assumes the simulator class has this method

    def run_experiment(self):
        """Run the experiment using the simulator and AI agent."""
        self.simulator.reset()  # Reset the simulation

        # Run the experiment for an unspecified amount of time based on the tool-calling convention loop
        while True:  # This is assuming you have a loop function in the tool-calling convention that handles time
            state = self.simulator.get_state('ball')  # Get state of 'ball' from the simulator
            user_input = f"Current state: {state}. What should I do next?"
            llm_response = self.agent.interact(user_input)  # Ask AI agent for action
            
            # Extract the tool calls from the LLM's response
            tool_calls_json = extract_json_response(llm_response)
            
            # Execute tool calls and aggregate the results
            print("\n=== Executing Tool Calls ===")
            results = execute_tool_calls(self.simulator, tool_calls_json)
            
            # Evaluate final answer correctness
            final_answer = results[-1]['parameters']['answer']
            correct_answer = "rolling"  # Assuming correct answer is "rolling"
            print("\n=== Aggregated Results ===")
            print(json.dumps(results, indent=2))
            answer_tool(final_answer, correct_answer)

            # Stop the loop when the agent provides an answer
            if final_answer:
                break

    def close(self):
        """Close the simulator."""
        self.simulator.close()  # Ensure the simulator is properly closed at the end of the experiment
