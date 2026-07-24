#Task--2
from dataclasses import dataclass
from typing import Optional, Tuple
@dataclass
class UAVMessage:
    """
    Standardized UAV Communication Packet
    Future-Ready Design
    """
    message_id: str
    sender_id: int
    receiver_id: int
    is_broadcast: bool
    timestamp: float
    sequence_number: int
    position: Tuple[float, float, float]
    target: Tuple[float, float, float]

    battery: float
    mission_status: str
    trust_score: Optional[float] = None

    encrypted_payload: Optional[bytes] = None

    digital_signature: Optional[str] = None
    def to_dict(self):
        """
        Convert message into dictionary.
        Useful for logging or network transmission.
        """

        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "is_broadcast": self.is_broadcast,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "position": self.position,
            "target": self.target,
            "battery": self.battery,
            "mission_status": self.mission_status,
            "trust_score": self.trust_score,
            "encrypted_payload": self.encrypted_payload,
            "digital_signature": self.digital_signature
        }

    def __str__(self):
        return (
            f"UAVMessage("
            f"ID={self.message_id}, "
            f"Sender={self.sender_id}, "
            f"Receiver={self.receiver_id}, "
            f"Broadcast={self.is_broadcast}, "
            f"Time={self.timestamp}, "
            f"Seq={self.sequence_number}, "
            f"Battery={self.battery:.2f}, "
            f"Status={self.mission_status})"
        )