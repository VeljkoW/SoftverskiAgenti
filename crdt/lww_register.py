"""
Last-Write-Wins Register CRDT implementation
"""
from datetime import datetime
from typing import Any, Dict


class LWWRegister:
    """
    Last-Write-Wins Register CRDT for conflict-free data synchronization.
    Resolves conflicts by keeping the value with the latest timestamp.
    """
    
    def __init__(self):
        self.data: Dict[str, tuple[Any, datetime]] = {}
        
    def set(self, key: str, value: Any, timestamp: datetime = None):
        """
        Set a value with timestamp.
        Only updates if the new timestamp is newer than the existing one.
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # LWW: only update if timestamp is newer
        if key not in self.data or timestamp > self.data[key][1]:
            self.data[key] = (value, timestamp)
            
    def get(self, key: str) -> Any:
        """Get the current value for a key"""
        if key in self.data:
            return self.data[key][0]
        return None
    
    def get_state(self) -> Dict:
        """Get the full CRDT state for synchronization"""
        return {
            key: {"value": value, "timestamp": timestamp.isoformat()}
            for key, (value, timestamp) in self.data.items()
        }
    
    def merge(self, remote_state: Dict):
        """
        Merge with remote CRDT state.
        For each key, keeps the value with the latest timestamp.
        """
        for key, item in remote_state.items():
            remote_value = item["value"]
            remote_timestamp = datetime.fromisoformat(item["timestamp"])
            self.set(key, remote_value, remote_timestamp)
    
    def get_all_keys(self):
        """Get all keys in the register"""
        return list(self.data.keys())
    
    def __repr__(self):
        return f"LWWRegister(keys={len(self.data)})"
