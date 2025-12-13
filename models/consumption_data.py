"""
Data models for electricity consumption
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ConsumptionData:
    """Represents electricity consumption data from a household"""
    household_id: str
    consumption: float  # Total kWh
    timestamp: datetime
    room_details: Optional[Dict[str, float]] = None  # Consumption per room
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "household_id": self.household_id,
            "consumption": self.consumption,
            "timestamp": self.timestamp.isoformat(),
            "room_details": self.room_details or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        return cls(
            household_id=data["household_id"],
            consumption=data["consumption"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            room_details=data.get("room_details")
        )
