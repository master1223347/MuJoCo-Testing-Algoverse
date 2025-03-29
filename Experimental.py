import json
from dotenv import load_dotenv
import os
from simulator import Simulator  # Import the Simulator class
from openai_agent import OpenAIAgent  # Import the OpenAI Agent class
from typing import Any, Dict, List

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

#[DYLAN] BOTH OF THESE SHOULD BE METHODS AND THEREFORE NOT NEED TO TAKE IN SIMULATOR ???

# Answer checking tool 
#[DYLAN] PRINTING DOES NOTHING FOR US NEEDS TO RETURN. done
def answer_tool(answer: Any, correct_answer: Any):
    if answer == correct_answer:
        return("[Answer Tool] ✅ Correct answer!")
    else:
        return(f"[Answer Tool] ❌ Incorrect answer! Got {answer}, expected {correct_answer}")


# Generalized execution of tool calls with try-except and timestep logging
def execute_tool_calls(sim: Simulator, tool_calls_json: str) -> List[Dict[str, Any]]:
    tool_calls = json.loads(tool_calls_json)
    aggregated_results = []

    # [DYLAN] ADD MORE TOOLS THAN THIS done
    tool_mapping = {
    # Core functionalities
        "get_state": sim.get_state,  # Retrieves the current simulation state
        "set_state": sim.set_state,  # Sets a new simulation state
        "step": sim.step,  # Advances the simulation by one step
        "run_steps": sim.run_steps,  # Runs multiple steps in sequence

    # Simulation control
        "reset": sim.reset,  # Resets simulation to initial state
        "pause": sim.pause,  # Pauses the simulation
        "resume": sim.resume,  # Resumes the simulation

    # State management
        "save_state": sim.save_state,  # Saves the current state
        "load_state": sim.load_state,  # Loads a previously saved state
        "export_state": sim.export_state,  # Exports the state snapshot

    # Information retrieval
        "get_actions": sim.get_actions,  # Retrieves available actions
        "get_metadata": sim.get_metadata,  # Retrieves simulation metadata
        "get_parameters": sim.get_parameters,  # Gets current simulation parameters
        "get_observation": sim.get_observation,  # Retrieves observation data (if applicable)

    # Logging and Debugging
        "enable_logging": sim.enable_logging,  # Turns on logging for debugging
        "disable_logging": sim.disable_logging,  # Turns off logging
        "get_logs": sim.get_logs,  # Retrieves simulation logs

    # Configuration and Customization
        "set_parameters": sim.set_parameters,  # Sets new simulation parameters
        "load_config": sim.load_config,  # Loads a configuration file
        "save_config": sim.save_config,  # Saves the current configuration

    # Visualization and Export
        "render": sim.render,  # Renders the simulation state
        "export_results": sim.export_results,  # Exports data/results for analysis
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
            #[DYLAN] HAVE TO THINK ABOUT THIS ONE MORE, BUT I THINK IF A TOOL RAISES AN ERROR THEN WE SHOULD NOT EXECUTE THE REMAINING TOOLS
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
    def __init__(self, scene_id, timeout=30): #[DYLAN] WE TALKED ABOUT THIS, IT SHOULDNT BE 30 SECONDS IT SHOULD BE LIKE 5 ITERATIONS, ALSO YOU DONT DO ANYTHING W THIS VARIABLE
        """
        Initialize the Experimental class with a Scene ID.

        Parameters:
        - scene_id: str, Identifier for the scene to load in the simulator.
        """
        api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
        self.simulator = Simulator()  # Initialize the simulator
        self.agent = OpenAIAgent(api_key)  # Initialize OpenAI agent with API key
        self.scene_id = scene_id

        #[DYLAN] WHERE IS THE SCENE CLASS? WE TALKED ABOUT THIS IN DETAIL
        
        # Load the scene based on the scene_id
        self.load_scene(self.scene_id) #[DYLAN] THIS SHOULD NOT BE HERE, ALL LOADING OF SIMULATOR SHOLD BE IN SIMULATOR CONSTRUCTOR

    def load_scene(self, scene_id):
        """Load a specific scene into the simulator based on the scene ID."""
        # Assuming the simulator has a method to load a scene by its ID
        print(f"Loading scene with ID: {scene_id}")
        self.simulator.load_scene(scene_id)  # This assumes the simulator class has this method

    def run_experiment(self):
        """Run the experiment using the simulator and AI agent."""
        self.simulator.reset()  # Reset the simulation

        #[DYLAN] OK THIS PART NEEDS A LOT OF WORK WE TALKED ABOUT GETTING THE PROMPT FROM SELF.SCENE.GET_PROMPT
        #ALSO THIS FILE IS MISSING THE TOOL DESCRIPTION PROMPT WE TALKED ABOUT (VERY IMPORTANT TO DESCRIBE ALL THE TOOLS
        #AFTER THE FIRST PROMPT THE INPUT TO THE MODEL SHOULD BE THE AGGREGATED RESULT
        
        # Run the experiment for an unspecified amount of time based on the tool-calling convention loop
        while True:  # [DYLAN] THIS SHOULD BE FOR ITR IN RANGE(MAX_ITR)
            state = self.simulator.get_state('ball')  # Get state of 'ball' from the simulator WHY ARE YOU HARDCODING BALL
            user_input = f"Current state: {state}. What should I do next?" 
            llm_response = self.agent.interact(user_input)  # Ask AI agent for action
            
            # Extract the tool calls from the LLM's response
            tool_calls_json = extract_json_response(llm_response)
            
            # Execute tool calls and aggregate the results
            print("\n=== Executing Tool Calls ===")
            results = execute_tool_calls(self.simulator, tool_calls_json)

            #[DYLAN] THIS LOGIC IS ALL MESSED UP IT SHOULD ONLY DO THE ANSWER LOGIC AND BREAK IF ANSWER IS PRESENT IN TOOL_CALLS_JSON
            # Evaluate final answer correctness
            final_answer = results[-1]['parameters']['answer']
            correct_answer = "rolling"  # Assuming correct answer is "rolling"
            print("\n=== Aggregated Results ===")
            print(json.dumps(results, indent=2))
            answer_tool(final_answer, correct_answer)

            # Stop the loop when the agent provides an answer
            if final_answer:
                break

    #[DYLAN] SHOULD ULTIMATELY RETURN SOME DATA, LIKE CORRECT: BOOL, NUMBER OF ENV_ITRS: INT, TIMEOUT: BOOL, NUM TOOL CALLS: INT

    def close(self): #[DYLAN] NOT NEEDED 
        """Close the simulator."""
        self.simulator.close()  # Ensure the simulator is properly closed at the end of the experiment
