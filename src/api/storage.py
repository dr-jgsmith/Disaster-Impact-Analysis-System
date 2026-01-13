"""
In-memory storage for sp_events.

This module provides simple in-memory storage for development and testing.
In production, this could be replaced with Redis, PostgreSQL, or other backends.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from src.core.base.sp_event import SpatialEvent


class EventStorage:
    """
    In-memory storage for spatial sp_events.
    
    Stores sp_event instances by ID for the duration of the application.
    Thread-safe for single-process deployment.
    """
    
    def __init__(self):
        """Initialize storage."""
        self._sp_events: Dict[str, Dict] = {}
        self._lock = None  # Could add threading.Lock() for thread safety
    
    def create(
        self,
        sp_event: SpatialEvent,
        event_type: str,
    ) -> str:
        """
        Store a new sp_event.
        
        Args:
            sp_event: Event instance
            event_type: Type identifier
        
        Returns:
            Generated sp_event ID
        """
        # Generate unique ID
        event_id = f"{event_type}_{uuid.uuid4().hex[:8]}"
        
        # Store sp_event with metadata
        self._sp_events[event_id] = {
            "sp_event": sp_event,
            "event_type": event_type,
            "created_at": datetime.utcnow(),
            "status": "ready",
        }
        
        return event_id
    
    def get(self, event_id: str) -> Optional[Dict]:
        """
        Retrieve sp_event by ID.
        
        Args:
            event_id: Event identifier
        
        Returns:
            Event data dict or None if not found
        """
        return self._sp_events.get(event_id)
    
    def get_sp_event(self, event_id: str) -> Optional[SpatialEvent]:
        """
        Get just the sp_event instance.
        
        Args:
            event_id: Event identifier
        
        Returns:
            SpatialEvent instance or None
        """
        data = self.get(event_id)
        return data["sp_event"] if data else None
    
    def update_status(self, event_id: str, status: str) -> bool:
        """
        Update sp_event status.
        
        Args:
            event_id: Event identifier
            status: New status
        
        Returns:
            True if updated, False if not found
        """
        if event_id in self._sp_events:
            self._sp_events[event_id]["status"] = status
            return True
        return False
    
    def delete(self, event_id: str) -> bool:
        """
        Delete sp_event.
        
        Args:
            event_id: Event identifier
        
        Returns:
            True if deleted, False if not found
        """
        if event_id in self._sp_events:
            del self._sp_events[event_id]
            return True
        return False
    
    def list_all(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """
        List all sp_events with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of sp_event data dicts
        """
        all_sp_events = list(self._sp_events.items())
        paginated = all_sp_events[skip : skip + limit]
        
        return [
            {
                "id": event_id,
                **data,
            }
            for event_id, data in paginated
        ]
    
    def count(self) -> int:
        """
        Get total count of sp_events.
        
        Returns:
            Number of stored sp_events
        """
        return len(self._sp_events)
    
    def clear(self) -> None:
        """Clear all sp_events (for testing)."""
        self._sp_events.clear()

