"""
Room model - represents a room in a household with electricity consumption
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class Room:
    """Represents a room in a household with electrical devices"""
    name: str
    base_consumption: float  # kWh per interval
    is_active: bool = True
    device_type: str = "general"  # living_room, bedroom, kitchen, bathroom
    
    def get_base_consumption(self) -> float:
        """
        Get base consumption for this room.
        Returns 0 if room is inactive.
        """
        if not self.is_active:
            return 0.0
        return self.base_consumption
    
    def turn_on(self):
        """Turn on the room (enable consumption)"""
        self.is_active = True
    
    def turn_off(self):
        """Turn off the room (disable consumption)"""
        self.is_active = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "base_consumption": self.base_consumption,
            "is_active": self.is_active,
            "device_type": self.device_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        return cls(
            name=data["name"],
            base_consumption=data["base_consumption"],
            is_active=data.get("is_active", True),
            device_type=data.get("device_type", "general")
        )
