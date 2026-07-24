from dronswarm_env import DroneSwarmEnv


def main():

    print("\n--- Starting UAV Swarm RL Simulation ---\n")

    env = DroneSwarmEnv(
        num_drones=5,
        comm_range=150.0
    )

    env.reset()

    for step in range(1, 6):

        observations, reward, terminated, truncated, info = env.step(
            actions=None
        )

        print(f"-- Step {step} --")

        comm_stats = info["comm_metrics"]

        print(f"Total Messages Sent: {comm_stats['total_messages_sent']}")

        print(f"Total Messages Received: {comm_stats['total_messages_received']}")

        print(
            f"Communication Success Rate: "
            f"{comm_stats['communication_success_rate_percent']}%"
        )

        print(
            f"Current Neighbors Connected: "
            f"{comm_stats['current_neighbors_connected']}"
        )

        print(
            f"Average Active Links Per Step: "
            f"{comm_stats['average_active_links_per_step']}"
        )

        print(
            f"Average Communication Distance: "
            f"{comm_stats['average_communication_distance']} meters"
        )


        if terminated or truncated:
            print("\nMission Completed.")
            break


if __name__ == "__main__":
    main()