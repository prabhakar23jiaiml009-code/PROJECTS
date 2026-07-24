# task--1
import gymnasium as gym
import numpy as np

from comm_manager import CommunicationManager


class DroneSwarmEnv(gym.Env):

    def __init__(self, num_drones=5, comm_range=150.0):
        super().__init__()

        self.num_drones = num_drones
        
        self.current_step = 0
        # bhai logic higest priority
        self.comm_manager = CommunicationManager(comm_range=comm_range)

        # UAV States
        self.positions = {
            i: np.random.rand(3) * 100
            for i in range(num_drones)
        }

        self.targets = {
            i: np.random.rand(3) * 100
            for i in range(num_drones)
        }

        self.batteries = {
            i: 100.0
            for i in range(num_drones)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.positions = {
            i: np.random.rand(3) * 100
            for i in range(self.num_drones)
        }

        self.targets = {
            i: np.random.rand(3) * 100
            for i in range(self.num_drones)
        }

        self.batteries = {
            i: 100.0
            for i in range(self.num_drones)
        }

        uav_states = {}

        for i in range(self.num_drones):
            uav_states[i] = {
                "position": self.positions[i],
                "target": self.targets[i],
                "battery": self.batteries[i],
                "status": "ACTIVE",
                "seq_num": 0
            }

        observations = self._get_observations(
            uav_states,
            {i: [] for i in range(self.num_drones)}
        )

        return observations, {}

    def step(self, actions):

        self.current_step += 1

        for i in range(self.num_drones):

            move = np.random.uniform(-1, 1, 3)

            self.positions[i] += move

            self.positions[i] = np.clip(
                self.positions[i],
                0,
                100
            )

            self.batteries[i] = max(
                0.0,
                self.batteries[i] - 0.5
            )

        uav_states = {}

        for i in range(self.num_drones):

            uav_states[i] = {

                "position": tuple(self.positions[i]),

                "target": tuple(self.targets[i]),

                "battery": float(self.batteries[i]),

                "status": "ACTIVE" if self.batteries[i] > 0 else "DEAD",

                "seq_num": self.current_step
            }

        # Communication Layer
        # Har env.step() par UAV apni state broadcast karta hai
        # har uav messages receive krta hai 
        messages_received = self.comm_manager.process_step(

            uav_states=uav_states,

            current_time=float(self.current_step)

        )
        # Rainforcement ka lafda 
        
        observations = self._get_observations(

            uav_states,

            messages_received

        )
        reward = 0.0

        terminated = False

        truncated = False
        info = {

            "comm_metrics":
            self.comm_manager.get_communication_statistics()

        }

        return observations, reward, terminated, truncated, info

    def _get_observations(self, states, comm_messages):

        observations = {}

        for uav_id in states:

            observations[uav_id] = {

                "state": states[uav_id],

                "received_messages": comm_messages[uav_id]

            }

        return observations