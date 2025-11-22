"""
In-memory storage for phenomena.

This module provides simple in-memory storage for development and testing.
In production, this could be replaced with Redis, PostgreSQL, or other backends.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from src.core.base.phenomenon import SpatialPhenomenon


class PhenomenonStorage:
    """
    In-memory storage for spatial phenomena.
    
    Stores phenomenon instances by ID for the duration of the application.
    Thread-safe for single-process deployment.
    """
    
    def __init__(self):
        """Initialize storage."""
        self._phenomena: Dict[str, Dict] = {}
        self._lock = None  # Could add threading.Lock() for thread safety
    
    def create(
        self,
        phenomenon: SpatialPhenomenon,
        phenomenon_type: str,
    ) -> str:
        """
        Store a new phenomenon.
        
        Args:
            phenomenon: Phenomenon instance
            phenomenon_type: Type identifier
        
        Returns:
            Generated phenomenon ID
        """
        # Generate unique ID
        phenomenon_id = f"{phenomenon_type}_{uuid.uuid4().hex[:8]}"
        
        # Store phenomenon with metadata
        self._phenomena[phenomenon_id] = {
            "phenomenon": phenomenon,
            "phenomenon_type": phenomenon_type,
            "created_at": datetime.utcnow(),
            "status": "ready",
        }
        
        return phenomenon_id
    
    def get(self, phenomenon_id: str) -> Optional[Dict]:
        """
        Retrieve phenomenon by ID.
        
        Args:
            phenomenon_id: Phenomenon identifier
        
        Returns:
            Phenomenon data dict or None if not found
        """
        return self._phenomena.get(phenomenon_id)
    
    def get_phenomenon(self, phenomenon_id: str) -> Optional[SpatialPhenomenon]:
        """
        Get just the phenomenon instance.
        
        Args:
            phenomenon_id: Phenomenon identifier
        
        Returns:
            SpatialPhenomenon instance or None
        """
        data = self.get(phenomenon_id)
        return data["phenomenon"] if data else None
    
    def update_status(self, phenomenon_id: str, status: str) -> bool:
        """
        Update phenomenon status.
        
        Args:
            phenomenon_id: Phenomenon identifier
            status: New status
        
        Returns:
            True if updated, False if not found
        """
        if phenomenon_id in self._phenomena:
            self._phenomena[phenomenon_id]["status"] = status
            return True
        return False
    
    def delete(self, phenomenon_id: str) -> bool:
        """
        Delete phenomenon.
        
        Args:
            phenomenon_id: Phenomenon identifier
        
        Returns:
            True if deleted, False if not found
        """
        if phenomenon_id in self._phenomena:
            del self._phenomena[phenomenon_id]
            return True
        return False
    
    def list_all(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """
        List all phenomena with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of phenomenon data dicts
        """
        all_phenomena = list(self._phenomena.items())
        paginated = all_phenomena[skip : skip + limit]
        
        return [
            {
                "id": phenom_id,
                **data,
            }
            for phenom_id, data in paginated
        ]
    
    def count(self) -> int:
        """
        Get total count of phenomena.
        
        Returns:
            Number of stored phenomena
        """
        return len(self._phenomena)
    
    def clear(self) -> None:
        """Clear all phenomena (for testing)."""
        self._phenomena.clear()

