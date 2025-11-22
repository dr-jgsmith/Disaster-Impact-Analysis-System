#!/bin/bash
# Script to rename phenomenon/phenomena to sp_event/sp_events throughout codebase

echo "Renaming phenomenon to sp_event throughout codebase..."

# Find all Python files and perform replacements
find src tests -name "*.py" -type f -exec sed -i '' \
  -e 's/SpatialPhenomenon/SpatialEvent/g' \
  -e 's/FloodPhenomenon/FloodEvent/g' \
  -e 's/ContagionPhenomenon/ContagionEvent/g' \
  -e 's/SupplyChainPhenomenon/SupplyChainEvent/g' \
  -e 's/phenomenon_type/event_type/g' \
  -e 's/get_phenomenon_type/get_event_type/g' \
  -e 's/PhenomenonType/EventType/g' \
  -e 's/PhenomenonStatus/EventStatus/g' \
  -e 's/PhenomenonLinks/EventLinks/g' \
  -e 's/PhenomenonInfo/EventInfo/g' \
  -e 's/PhenomenonSummary/EventSummary/g' \
  -e 's/PhenomenonList/EventList/g' \
  -e 's/PhenomenonStorage/EventStorage/g' \
  -e 's/CreatePhenomenonRequest/CreateEventRequest/g' \
  -e 's/CreatePhenomenonResponse/CreateEventResponse/g' \
  -e 's/phenomenon_to_geojson/sp_event_to_geojson/g' \
  -e 's/phenomenon_id/event_id/g' \
  -e 's/phenomenonId/eventId/g' \
  -e 's/phenom_id/event_id/g' \
  -e 's/phenomenon/sp_event/g' \
  -e 's/phenomena/sp_events/g' \
  -e 's/Phenomenon/Event/g' \
  -e 's/src\.core\.base\.phenomenon/src.core.base.sp_event/g' \
  -e 's/src\.core\.phenomena/src.core.sp_events/g' \
  -e 's/from src.core.phenomena/from src.core.sp_events/g' \
  -e 's/import phenomena/import sp_events/g' \
  {} \;

# Update HTML files  
find frontend -name "*.html" -type f -exec sed -i '' \
  -e 's/phenomenon_type/event_type/g' \
  -e 's/phenomenonId/eventId/g' \
  -e 's/phenomenon/sp_event/g' \
  -e 's/phenomena/sp_events/g' \
  {} \;

# Update markdown files
find . -name "*.md" -maxdepth 1 -type f -exec sed -i '' \
  -e 's/SpatialPhenomenon/SpatialEvent/g' \
  -e 's/FloodPhenomenon/FloodEvent/g' \
  -e 's/phenomenon_type/event_type/g' \
  -e 's/Multi-phenomenon/Multi-event/g' \
  -e 's/multi-phenomenon/multi-event/g' \
  -e 's/phenomenon/event/g' \
  -e 's/phenomena/events/g' \
  {} \;

echo "Renaming complete!"
echo "Files have been updated. Please review changes with 'git diff'"

