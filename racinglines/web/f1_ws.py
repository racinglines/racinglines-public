"""
FastF1 live stream integration for real-time F1 timing and telemetry.

Fetches live session data from FastF1 and streams it to connected clients via WebSocket.
Supports live timing tables, position updates, gap to leader, and session status.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

try:
    import fastf1
    from fastf1.core import Session
except ImportError:
    fastf1 = None
    Session = None

logger = logging.getLogger(__name__)


@dataclass
class DriverPosition:
    """A driver's current position in the session."""
    position: int
    driver_number: int
    driver_name: str
    team: str
    gap_to_leader: Optional[float]  # in seconds
    last_lap_time: Optional[float]  # in seconds
    best_lap_time: Optional[float]  # in seconds
    status: str  # "on_track", "pitted", "out", etc.
    lap_count: int


@dataclass
class SessionStatus:
    """Current status of an F1 session."""
    session_type: str  # "fp1", "fp2", "fp3", "sprint_qual", "sprint", "qual", "race"
    status: str  # "not_started", "ongoing", "completed", "paused"
    time_remaining: Optional[int]  # seconds for timed sessions
    laps_remaining: Optional[int]  # laps for race
    lap_count: int  # current lap being run
    flag: Optional[str]  # "green", "yellow", "red", "chequered"
    timestamp: datetime


class FastF1LiveClient:
    """Manages FastF1 live session data fetching and updates."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self.current_positions: Dict[int, DriverPosition] = {}
        self.session_status: Optional[SessionStatus] = None
        self.last_update: datetime = datetime.now(timezone.utc)

    @staticmethod
    async def get_session(year: int, round_num: int, session_name: str) -> Optional[Session]:
        """
        Get a FastF1 session for live timing.
        session_name: "FP1", "FP2", "FP3", "SQ", "SS", "Q", "R"
        """
        if not fastf1:
            logger.warning("FastF1 not available; cannot fetch session data")
            return None

        try:
            session = fastf1.get_session(year, round_num, session_name)
            # Load live telemetry data
            session.load(telemetry=False, weather=False, messages=False)
            return session
        except Exception as e:
            logger.error(f"Failed to load F1 session {year} R{round_num} {session_name}: {e}")
            return None

    async def update_live_timing(self) -> Dict[str, Any]:
        """
        Fetch and return current live timing data.
        """
        if not self.session:
            # Provide demo data when session is not available
            return self._get_demo_data()

        if not fastf1:
            return {"error": "FastF1 module not available"}

        try:
            # Reload session to get latest data (FastF1 caches data)
            self.session.load(telemetry=False, weather=False, messages=False, restart=True)

            drivers = self._extract_driver_positions()
            status = self._extract_session_status()

            self.current_positions = {d.driver_number: d for d in drivers}
            self.session_status = status
            self.last_update = datetime.now(timezone.utc)

            return {
                "drivers": [asdict(d) for d in drivers],
                "session": asdict(status),
                "timestamp": self.last_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating live timing: {e}")
            return {"error": str(e)}

    def _extract_driver_positions(self) -> List[DriverPosition]:
        """Extract current driver positions from session data."""
        if not self.session or not hasattr(self.session, "laps"):
            return []

        try:
            laps = self.session.laps
            if laps.empty:
                return []

            # Group by driver and get latest lap
            drivers = []
            for driver_num, group in laps.groupby("Driver"):
                latest = group.iloc[-1]

                # Calculate gap to leader
                gap = None
                if hasattr(latest, "Time") and latest.Time is not None:
                    leader_time = laps["Time"].min()
                    if leader_time is not None:
                        gap = (latest.Time - leader_time).total_seconds()

                drivers.append(DriverPosition(
                    position=int(latest.Position) if hasattr(latest, "Position") and latest.Position else len(drivers) + 1,
                    driver_number=int(driver_num),
                    driver_name=str(driver_num),  # FastF1 uses driver numbers; map to name if needed
                    team=getattr(latest, "Team", "Unknown"),
                    gap_to_leader=gap,
                    last_lap_time=self._time_to_seconds(getattr(latest, "Time", None)),
                    best_lap_time=self._get_best_lap_time(driver_num, laps),
                    status="on_track",
                    lap_count=int(getattr(latest, "LapNumber", 0))
                ))

            # Sort by position
            drivers.sort(key=lambda d: d.position)
            return drivers
        except Exception as e:
            logger.error(f"Error extracting driver positions: {e}")
            return []

    def _extract_session_status(self) -> SessionStatus:
        """Extract current session status."""
        session_type_map = {
            "Practice 1": "fp1",
            "Practice 2": "fp2",
            "Practice 3": "fp3",
            "Sprint Qualifying": "sprint_qual",
            "Sprint": "sprint",
            "Qualifying": "qual",
            "Race": "race"
        }

        session_type = session_type_map.get(getattr(self.session, "name", ""), "unknown")

        return SessionStatus(
            session_type=session_type,
            status="ongoing",
            time_remaining=None,
            laps_remaining=None,
            lap_count=int(getattr(self.session, "current_lap", 0)) if hasattr(self.session, "current_lap") else 0,
            flag=None,
            timestamp=datetime.now(timezone.utc)
        )

    def _get_demo_data(self) -> Dict[str, Any]:
        """Return demo/mock F1 timing data for UI testing when session is unavailable."""
        demo_drivers = [
            DriverPosition(1, 1, "Max Verstappen", "Red Bull Racing", 0.0, 95.234, 94.567, "on_track", 45),
            DriverPosition(2, 81, "Oscar Piastri", "McLaren", 1.234, 95.890, 94.234, "on_track", 45),
            DriverPosition(3, 16, "Charles Leclerc", "Ferrari", 2.456, 96.123, 94.890, "on_track", 45),
            DriverPosition(4, 44, "Lewis Hamilton", "Mercedes", 3.567, 96.234, 95.123, "on_track", 45),
            DriverPosition(5, 55, "Carlos Sainz", "Ferrari", 4.789, 96.456, 95.456, "pitted", 44),
        ]

        demo_status = SessionStatus(
            session_type="race",
            status="ongoing",
            time_remaining=None,
            laps_remaining=56,
            lap_count=45,
            flag="green",
            timestamp=datetime.now(timezone.utc)
        )

        return {
            "drivers": [asdict(d) for d in demo_drivers],
            "session": asdict(demo_status),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @staticmethod
    def _time_to_seconds(time_obj) -> Optional[float]:
        """Convert FastF1 time object to seconds."""
        if time_obj is None:
            return None
        try:
            if hasattr(time_obj, "total_seconds"):
                return time_obj.total_seconds()
            return float(time_obj)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_best_lap_time(driver_num: int, laps) -> Optional[float]:
        """Get best lap time for a driver from laps dataframe."""
        try:
            driver_laps = laps[laps["Driver"] == driver_num]
            if driver_laps.empty or not hasattr(driver_laps, "Time"):
                return None
            best = driver_laps["Time"].min()
            return FastF1LiveClient._time_to_seconds(best)
        except Exception:
            return None


class LiveStreamManager:
    """Manages WebSocket connections and broadcasts live updates."""

    def __init__(self):
        self.clients: Dict[str, Any] = {}  # session_id -> websocket connection
        self.update_tasks: Dict[str, asyncio.Task] = {}
        self.fastf1_clients: Dict[str, FastF1LiveClient] = {}

    async def add_client(self, session_id: str, websocket):
        """Add a new WebSocket client."""
        self.clients[session_id] = websocket
        logger.info(f"Client connected: {session_id}")

    async def remove_client(self, session_id: str):
        """Remove a WebSocket client."""
        if session_id in self.clients:
            del self.clients[session_id]
            logger.info(f"Client disconnected: {session_id}")

        # Stop updates if no clients
        if not self.clients and session_id in self.update_tasks:
            self.update_tasks[session_id].cancel()
            del self.update_tasks[session_id]

    async def start_live_updates(self, session_id: str, fastf1_session: Optional[Session], update_interval: int = 5):
        """Start sending live updates to all connected clients."""
        client = FastF1LiveClient(fastf1_session)
        self.fastf1_clients[session_id] = client

        async def update_loop():
            try:
                while session_id in self.clients and self.clients[session_id]:
                    data = await client.update_live_timing()

                    # Broadcast to all clients
                    for ws in self.clients.values():
                        try:
                            await ws.send_text(json.dumps({"type": "timing_update", "data": data}))
                        except Exception as e:
                            logger.error(f"Error sending to client: {e}")

                    await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                logger.info(f"Update loop cancelled for {session_id}")
            except Exception as e:
                logger.error(f"Error in update loop: {e}")

        # Cancel existing task
        if session_id in self.update_tasks:
            self.update_tasks[session_id].cancel()

        # Start new update task
        task = asyncio.create_task(update_loop())
        self.update_tasks[session_id] = task


# Global live stream manager
live_stream_manager = LiveStreamManager()
