import numpy as np
from typing import Dict, List, Tuple
from uav_message import UAVMessage

class CommunicationManager:

    def __init__(self, comm_range: float = 100.0):

        self.comm_range = comm_range

        # Communication Statistics
        self.total_sent = 0
        self.total_received = 0
        self.total_distance_sum = 0.0
        self.total_steps = 0

        self.total_active_links_accumulated = 0
        self.current_active_links = 0

    def calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float]
    ) -> float:

        p1 = np.array(pos1, dtype=float)
        p2 = np.array(pos2, dtype=float)

        return float(np.linalg.norm(p1 - p2))

    def process_step(
        self,
        uav_states: Dict[int, dict],
        current_time: float
    ) -> Dict[int, List[UAVMessage]]:

        received_messages = {
            uav_id: []
            for uav_id in uav_states.keys()
        }

        uav_ids = list(uav_states.keys())

        if len(uav_ids) <= 1:
            return received_messages

        possible_transmissions = len(uav_ids) * (len(uav_ids) - 1)

        received_this_step = 0
        active_links = 0
        distance_sum = 0.0

        for sender_id in uav_ids:

            sender = uav_states[sender_id]

            for receiver_id in uav_ids:

                if sender_id == receiver_id:
                    continue

                distance = self.calculate_distance(
                    sender["position"],
                    uav_states[receiver_id]["position"]
                )

                if distance > self.comm_range:
                    continue

                active_links += 1
                received_this_step += 1
                distance_sum += distance

                message = UAVMessage(

                    message_id=f"MSG_{sender_id}_{receiver_id}_{self.total_sent+1}",

                    sender_id=sender_id,

                    receiver_id=receiver_id,

                    is_broadcast=True,

                    timestamp=current_time,

                    sequence_number=sender.get("seq_num", 1),

                    position=tuple(sender["position"]),

                    target=tuple(sender["target"]),

                    battery=float(sender["battery"]),

                    mission_status=str(sender["status"]),

                    # Future Research
                    trust_score=None,
                    encrypted_payload=None,
                    digital_signature=None
                )

                received_messages[receiver_id].append(message)

        self.total_steps += 1
        self.current_active_links = active_links

        self.total_sent += possible_transmissions
        self.total_received += received_this_step

        self.total_distance_sum += distance_sum
        self.total_active_links_accumulated += active_links

        return received_messages

    def get_communication_statistics(self) -> dict:

        success_rate = 0.0

        if self.total_sent > 0:
            success_rate = (
                self.total_received /
                self.total_sent
            ) * 100

        avg_distance = 0.0

        if self.total_received > 0:
            avg_distance = (
                self.total_distance_sum /
                self.total_received
            )

        avg_active_links = 0.0

        if self.total_steps > 0:
            avg_active_links = (
                self.total_active_links_accumulated /
                self.total_steps
            )

        return {

            "total_messages_sent":
                self.total_sent,

            "total_messages_received":
                self.total_received,

            "communication_success_rate_percent":
                round(success_rate, 2),

            "current_neighbors_connected":
                self.current_active_links,

            "average_active_links_per_step":
                round(avg_active_links, 2),

            "average_communication_distance":
                round(avg_distance, 2)
        }