"""
Consumption Simulator - generates realistic electricity consumption data
"""
from datetime import datetime
from typing import List, Dict
from models.room import Room
import random
import logging

logger = logging.getLogger(__name__)


class ConsumptionSimulator:
    """Simulates electricity consumption for a household with multiple rooms"""
    
    def __init__(self, household_id: str):
        self.household_id = household_id
        self.rooms = []
        
    def initialize_rooms(self) -> List[Room]:
        """
        Initialize rooms with randomized base consumption.
        Each household gets unique consumption patterns.
        """
        # Randomize base consumption for each household (±30% variation)
        living_room_base = random.uniform(0.25, 0.40)
        kitchen_base = random.uniform(0.40, 0.65)
        bedroom_base = random.uniform(0.10, 0.20)
        bathroom_base = random.uniform(0.15, 0.30)
        
        # Random number of bedrooms (1-3)
        num_bedrooms = random.randint(1, 3)
        
        self.rooms = [
            Room("Living Room", living_room_base, True, "living_room"),
            Room("Kitchen", kitchen_base, True, "kitchen"),
            Room("Bathroom", bathroom_base, True, "bathroom"),
        ]
        
        # Add random number of bedrooms
        for i in range(num_bedrooms):
            bedroom_consumption = random.uniform(0.10, 0.20)
            self.rooms.append(Room(f"Bedroom {i+1}", bedroom_consumption, True, "bedroom"))
        
        logger.info(
            f"{self.household_id} initialized with {len(self.rooms)} rooms "
            f"(Living: {living_room_base:.2f}, Kitchen: {kitchen_base:.2f}, "
            f"Bedrooms: {num_bedrooms})"
        )
        return self.rooms
    
    def get_total_consumption(self, hour: int = None) -> tuple[float, Dict[str, float]]:
        """
        Calculate total consumption and per-room breakdown with realistic simulation.
        
        Returns:
            tuple: (total_consumption, room_details_dict)
        """
        if hour is None:
            hour = datetime.now().hour
        
        room_consumptions = {}
        total_consumption = 0.0
        
        for room in self.rooms:
            # Get base consumption (0 if inactive)
            base = room.get_base_consumption()
            
            if base == 0:
                room_consumptions[room.name] = 0.0
                continue
            
            # Apply time-based patterns based on room type
            consumption = self._simulate_room_consumption(base, room.device_type, hour)
            room_consumptions[room.name] = consumption
            total_consumption += consumption
        
        return total_consumption, room_consumptions
    
    def _simulate_room_consumption(self, base_consumption: float, device_type: str, hour: int) -> float:
        """
        Simulate realistic consumption for a room based on type and time.
        All randomization logic is here.
        """
        consumption = base_consumption
        
        # Adjust based on room type and time
        if device_type == "kitchen":
            # Higher consumption during meal times
            if hour in [7, 8, 12, 13, 18, 19, 20]:
                consumption *= random.uniform(2.0, 3.0)
            else:
                consumption *= random.uniform(0.3, 0.7)
                
        elif device_type == "living_room":
            # Higher in evening
            if 18 <= hour <= 23:
                consumption *= random.uniform(1.5, 2.5)
            else:
                consumption *= random.uniform(0.2, 0.8)
                
        elif device_type == "bedroom":
            # Higher at night
            if hour < 7 or hour > 22:
                consumption *= random.uniform(0.5, 1.0)
            else:
                consumption *= random.uniform(0.1, 0.3)
                
        elif device_type == "bathroom":
            # Morning and evening peaks
            if hour in [6, 7, 8, 21, 22, 23]:
                consumption *= random.uniform(1.0, 2.0)
            else:
                consumption *= random.uniform(0.2, 0.5)
        
        # Add some random noise
        consumption += random.uniform(-0.05, 0.05)
        
        return max(0.0, consumption)
    
    def get_room(self, room_name: str) -> Room:
        """Get a specific room by name"""
        for room in self.rooms:
            if room.name == room_name:
                return room
        return None
    
    def toggle_room(self, room_name: str) -> bool:
        """
        Toggle a room on/off.
        
        Returns:
            bool: True if room was found and toggled, False otherwise
        """
        room = self.get_room(room_name)
        if room:
            if room.is_active:
                room.turn_off()
                logger.info(f"{self.household_id} - Turned OFF {room_name}")
            else:
                room.turn_on()
                logger.info(f"{self.household_id} - Turned ON {room_name}")
            return True
        
        logger.warning(f"{self.household_id} - Room '{room_name}' not found")
        return False
    
    def get_room_status(self) -> List[Dict]:
        """Get status of all rooms"""
        return [room.to_dict() for room in self.rooms]
    
    def get_active_room_count(self) -> int:
        """Get number of active rooms"""
        return sum(1 for room in self.rooms if room.is_active)
    
    def set_rooms(self, rooms: List[Room]):
        """Set custom room configuration"""
        self.rooms = rooms
    
    def turn_off_room(self, room_name: str) -> bool:
        """
        Turn off a specific room (used by central actor for control).
        
        Returns:
            bool: True if room was found and turned off, False otherwise
        """
        room = self.get_room(room_name)
        if room:
            room.turn_off()
            logger.warning(f"🔴 {self.household_id} - {room_name} FORCED OFF by central control")
            return True
        
        logger.error(f"{self.household_id} - Cannot turn off '{room_name}' - room not found")
        return False
    
    def reduce_consumption(self, target_reduction: float = 0.2) -> List[str]:
        """
        Reduce consumption by turning off some rooms.
        Prioritizes rooms with higher consumption.
        
        Args:
            target_reduction: Fraction of current consumption to reduce (0.0-1.0)
        
        Returns:
            List of room names that were turned off
        """
        # Calculate current total consumption
        current_total, room_consumptions = self.get_total_consumption()
        target_amount = current_total * target_reduction
        
        # Get active rooms sorted by consumption (highest first)
        active_rooms = [(name, consumption) for name, consumption in room_consumptions.items() 
                       if consumption > 0]
        active_rooms.sort(key=lambda x: x[1], reverse=True)
        
        rooms_turned_off = []
        reduction_achieved = 0.0
        
        # Turn off rooms until we reach target reduction
        for room_name, consumption in active_rooms:
            if reduction_achieved >= target_amount:
                break
            
            room = self.get_room(room_name)
            if room and room.is_active:
                room.turn_off()
                rooms_turned_off.append(room_name)
                reduction_achieved += consumption
                logger.info(
                    f"📉 {self.household_id} - Turned off {room_name} "
                    f"({consumption:.2f} kWh saved)"
                )
        
        if rooms_turned_off:
            logger.warning(
                f"📉 {self.household_id} - Reduced consumption by {reduction_achieved:.2f} kWh "
                f"(target: {target_amount:.2f} kWh)"
            )
        
        return rooms_turned_off
