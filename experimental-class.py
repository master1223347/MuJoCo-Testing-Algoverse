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

# Running the simulation
if __name__ == "__main__":
    model_path = "your_model.xml"  # Replace with actual XML model path

    try:
        scene = ExperimentalMuJoCoScene(model_path)
        scene.run()
    finally:
        scene.close()
