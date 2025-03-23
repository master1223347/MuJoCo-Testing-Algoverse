import mujoco
import mujoco.viewer
import numpy as np

class ExperimentalMuJoCoScene:
    def __init__(self, model_path, sim_time=5.0, render=True):
        """
        Initialize the MuJoCo environment for an experimental scene.

        Parameters:
        - model_path: str, path to the XML model file defining the MuJoCo scene
        - sim_time: float, duration of the simulation (in seconds)
        - render: bool, whether to render the scene during the simulation
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_time = sim_time
        self.time_step = self.model.opt.timestep  # Time step of the simulation

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        """Reset the simulation to its initial state."""
        mujoco.mj_resetData(self.model, self.data)

    def step(self):
        """Advance the simulation by one time step."""
        mujoco.mj_step(self.model, self.data)

    def render(self):
        """Render the current state of the scene."""
        if self.viewer:
            self.viewer.sync()

    def run(self):
        """Run the simulation for the given time."""
        total_steps = int(self.sim_time / self.time_step)
        for _ in range(total_steps):
            self.step()
            self.render()

    def get_state(self):
        """Return the current state of the simulation."""
        return self.data.qpos.copy(), self.data.qvel.copy()

    def apply_action(self, action):
        """Apply an action to the scene (e.g., control joints or tendons)."""
        self.data.ctrl[:] = action

    def close(self):
        """Close the viewer if it's being used."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # --- LLM INTERACTION FUNCTIONS ---
    
    def get_env_info(self):
        """Return a structured summary of the environment for an LLM."""
        env_info = {
            "num_bodies": self.model.nbody,
            "num_joints": self.model.njnt,
            "num_actuators": self.model.nu,
            "timestep": self.model.opt.timestep,
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "control": self.data.ctrl.tolist()
        }
        return env_info

    def execute_command(self, command, args=None):
        """
        Execute a command from an LLM.

        Parameters:
        - command: str, the command name (e.g., "reset", "apply_action").
        - args: dict, additional parameters if needed.

        Returns:
        - str, a response describing the result of the execution.
        """
        if command == "reset":
            self.reset()
            return "Simulation reset."

        elif command == "step":
            self.step()
            return "Simulation stepped forward."

        elif command == "apply_action" and args:
            action = np.array(args.get("action", []))
            if len(action) != self.model.nu:
                return f"Error: Expected action size {self.model.nu}, got {len(action)}."
            self.apply_action(action)
            return f"Action {action.tolist()} applied."

        elif command == "get_state":
            state = self.get_state()
            return {"qpos": state[0].tolist(), "qvel": state[1].tolist()}

        elif command == "observe":
            return self.observe()

        else:
            return f"Unknown command: {command}"

    def observe(self):
        """
        Generate a textual description of the scene.

        This is useful for an LLM to understand the environment state.
        """
        description = (
            f"The environment contains {self.model.nbody} bodies, "
            f"{self.model.njnt} joints, and {self.model.nu} actuators. "
            f"The current joint positions are {self.data.qpos.tolist()}, "
            f"and velocities are {self.data.qvel.tolist()}."
        )
        return description


# Running the simulation
if __name__ == "__main__":
    model_path = "your_model.xml"  # Replace with actual XML model path

    try:
        scene = ExperimentalMuJoCoScene(model_path)

        # Example LLM interaction
        print(scene.execute_command("reset"))
        print(scene.execute_command("observe"))
        print(scene.execute_command("apply_action", {"action": [0.1, -0.2]}))  # Example control input
        print(scene.execute_command("get_state"))

        scene.run()
    finally:
        scene.close()
