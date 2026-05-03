"""
FastAPI Routing Service
Microservice for route calculation and optimization
This service runs on a separate port (8010) and provides routing endpoints

Run with: python routing_service.py
Or: uvicorn routing_service:app --host 0.0.0.0 --port 8010
"""
import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# Import Google Drive downloader
try:
    from google_drive import download_all_graphml_files
    GDRIVE_AVAILABLE = True
except Exception as e:
    GDRIVE_AVAILABLE = False
    logging.warning(f"Google Drive download not available: {e}")

# Import graphml routing functions for real road routing
try:
    from graphml import get_route_on_roads, find_regions_for_points, build_region_index
    GRAPHML_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("✓ GraphML routing module loaded - using real roads")
except Exception as e:
    GRAPHML_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"✗ GraphML module not available - using fallback only: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== PYDANTIC MODELS =====================

class Point(BaseModel):
    lat: float
    lng: float


class RouteRequest(BaseModel):
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    driver_lat: Optional[float] = None
    driver_lng: Optional[float] = None
    driver_region: Optional[str] = None  # Region of operation for the driver
    alternatives: Optional[int] = 3


class Route(BaseModel):
    coordinates: List[List[float]]  # [[lng, lat], ...]
    distance_m: float  # Distance in meters
    distance_km: float  # Distance in kilometers
    duration_seconds: int  # Duration in seconds
    duration_minutes: int  # Duration in minutes


class RouteResponse(BaseModel):
    primary: Route
    alternatives: Optional[List[Route]] = None


class GeocodeRequest(BaseModel):
    address: str
    country_codes: Optional[List[str]] = None


class GeocodeResponse(BaseModel):
    address: str
    lat: float
    lng: float


# ===================== FASTAPI APP =====================

app = FastAPI(
    title="Delivery Routing Service",
    description="Microservice for route calculation and optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== CACHE =====================

class SimpleCache:
    """Simple in-memory cache with TTL"""
    def __init__(self, max_size=5000, ttl_seconds=3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str):
        if key in self.cache:
            if datetime.utcnow() - self.timestamps[key] < timedelta(seconds=self.ttl_seconds):
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = datetime.utcnow()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()


route_cache = SimpleCache(max_size=5000, ttl_seconds=3600)

# ===================== HELPER FUNCTIONS =====================

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate distance between two points using Haversine formula
    Returns distance in meters
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = radians(lat1)
    lng1_rad = radians(lng1)
    lat2_rad = radians(lat2)
    lng2_rad = radians(lng2)
    
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlng/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def estimate_duration(distance_m: float, avg_speed_kmh: float = 40) -> int:
    """
    Estimate duration based on distance and average speed
    Returns duration in seconds
    """
    distance_km = distance_m / 1000
    hours = distance_km / avg_speed_kmh
    return int(hours * 3600)


def generate_route_coordinates_from_graphml(lat1: float, lng1: float, lat2: float, lng2: float, region: Optional[str] = None) -> Optional[List[List[float]]]:
    """
    Generate REAL road coordinates using GraphML data
    Returns None if routing not available, falls back to interpolation
    """
    try:
        import networkx as nx
        import os
        
        # Determine which region's GraphML to use
        if region:
            graph_name = region
        else:
            # Find region by bounding box of coordinates
            # Using rough Ghana region mapping
            if lat1 > 10 and lat1 < 11.5:
                if lng1 > 0 and lng1 < 3:
                    graph_name = "Northern"
                elif lng1 > -1 and lng1 < 1:
                    graph_name = "North_East"
                elif lng1 > -3.5 and lng1 < -1:
                    graph_name = "Upper_West"
                else:
                    graph_name = "Northern"
            elif lat1 > 9 and lat1 < 10:
                graph_name = "Savannah"
            else:
                return None  # Unknown region
        
        # Load the graph
        graphml_path = os.path.join(os.path.dirname(__file__), 'data', f'{graph_name}_Region_Ghana.graphml')
        if not os.path.exists(graphml_path):
            logger.warning(f"GraphML not found: {graphml_path}")
            return None
        
        logger.info(f"Loading graph: {graphml_path}")
        G = nx.read_graphml(graphml_path)
        logger.info(f"Loaded graph with {G.number_of_nodes()} nodes")
        
        # Convert edge lengths from string to float (GraphML stores as strings)
        for u, v, key, data in G.edges(keys=True, data=True):
            if 'length' in data:
                try:
                    data['length'] = float(data['length'])
                except (ValueError, TypeError):
                    # If conversion fails, use default value
                    data['length'] = 1.0
        
        # Find nearest nodes to pickup and dropoff
        min_pickup_dist = float('inf')
        nearest_pickup_node = None
        min_dropoff_dist = float('inf')
        nearest_dropoff_node = None
        
        for node_id, node_data in G.nodes(data=True):
            try:
                node_lat = float(node_data.get('y', 0))
                node_lng = float(node_data.get('x', 0))
            except (ValueError, TypeError):
                continue
            
            # Distance to pickup
            pickup_dist = calculate_distance(lat1, lng1, node_lat, node_lng)
            if pickup_dist < min_pickup_dist:
                min_pickup_dist = pickup_dist
                nearest_pickup_node = node_id
            
            # Distance to dropoff
            dropoff_dist = calculate_distance(lat2, lng2, node_lat, node_lng)
            if dropoff_dist < min_dropoff_dist:
                min_dropoff_dist = dropoff_dist
                nearest_dropoff_node = node_id
        
        # If no nodes found or nodes too far away, return None
        if not nearest_pickup_node or not nearest_dropoff_node:
            logger.warning("Could not find nodes near pickup/dropoff")
            return None
        
        if min_pickup_dist > 5000 or min_dropoff_dist > 5000:  # 5km threshold
            logger.warning(f"Nearest nodes too far: pickup={min_pickup_dist:.0f}m, dropoff={min_dropoff_dist:.0f}m")
            return None
        
        # Find shortest path
        try:
            path = nx.shortest_path(G, nearest_pickup_node, nearest_dropoff_node, weight='length')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning("No path found between nodes")
            return None
        
        # Extract coordinates along path
        coords = []
        for i, node_id in enumerate(path):
            node_data = G.nodes[node_id]
            try:
                lat = float(node_data.get('y', 0))
                lng = float(node_data.get('x', 0))
                if lat != 0 and lng != 0:  # Skip invalid coordinates
                    coords.append([lng, lat])
            except (ValueError, TypeError):
                continue
        
        if len(coords) < 2:
            logger.warning("Insufficient coordinates in path")
            return None
        
        logger.info(f"Generated GraphML route with {len(coords)} waypoints for region {graph_name}")
        return coords
        
    except Exception as e:
        logger.error(f"Error generating GraphML route: {e}", exc_info=True)
        return None


def generate_route_coordinates(lat1: float, lng1: float, lat2: float, lng2: float, region: Optional[str] = None) -> List[List[float]]:
    """
    Generate REAL road coordinates using OSM data via GraphML
    NO FALLBACK - Only returns real road routes or raises error
    Routes are cached for fast retrieval
    """
    # Create cache key for this route
    cache_key = f"route_{lat1:.6f}_{lng1:.6f}_{lat2:.6f}_{lng2:.6f}_{region or 'auto'}"
    
    # Check cache first
    cached = route_cache.get(cache_key)
    if cached:
        logger.info(f"[ROUTE] Cache hit for route (saved real calculation)")
        return cached
    
    if not GRAPHML_AVAILABLE:
        raise ValueError("GraphML routing module not available - real roads data cannot be loaded")
    
    try:
        logger.info(f"[ROUTE] Calculating real roads route from ({lat1:.4f}, {lng1:.4f}) to ({lat2:.4f}, {lng2:.4f})")
        
        # Use the graphml module's get_route_on_roads function
        result = get_route_on_roads(
            pickup={'lat': lat1, 'lng': lng1},
            dropoff={'lat': lat2, 'lng': lng2},
            num_alternatives=1,
            detail_level='medium',
            region=region
        )
        
        if result and 'primary' in result and 'coordinates' in result['primary']:
            coords = result['primary']['coordinates']
            if len(coords) >= 2:
                logger.info(f"[ROUTE] ✓ Real roads route calculated: {len(coords)} waypoints")
                # Cache the route for future requests
                route_cache.set(cache_key, coords)
                return coords
        
        # If no valid route returned from GraphML
        raise ValueError(f"GraphML returned invalid route data")
        
    except Exception as e:
        logger.error(f"[ROUTE] FATAL: Real roads routing failed: {e}")
        raise ValueError(f"Route calculation failed - real roads data unavailable: {str(e)}")

# NOTE: Fallback routes REMOVED - only real road routes are used
# If GraphML routing fails, an error is raised instead of showing fake routes


def generate_alternative_routes(lat1: float, lng1: float, lat2: float, lng2: float, count: int = 2) -> List[Route]:
    """Generate alternative route options"""
    alternatives = []
    
    # Generate slightly offset routes
    offsets = [0.0005, -0.0005]
    for offset in offsets[:count]:
        lat_offset = lat1 + offset
        lng_offset = lng1 + offset
        
        coords = generate_route_coordinates(lat_offset, lng_offset, lat2, lng2)
        distance_m = calculate_distance(lat_offset, lng_offset, lat2, lng2)
        duration_s = estimate_duration(distance_m)
        
        alternatives.append(Route(
            coordinates=coords,
            distance_m=distance_m,
            distance_km=round(distance_m / 1000, 2),
            duration_seconds=duration_s,
            duration_minutes=round(duration_s / 60, 0)
        ))
    
    return alternatives


# ===================== ENDPOINTS =====================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "routing-service",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/route", response_model=RouteResponse)
async def get_route(req: RouteRequest):
    """
    Calculate route between two points
    
    Query params:
    - pickup_lat, pickup_lng: Start point
    - dropoff_lat, dropoff_lng: End point
    - driver_lat, driver_lng (optional): Current driver location
    - alternatives: Number of alternative routes (default 3)
    """
    try:
        # Create cache key
        cache_key = f"route_{req.pickup_lat}_{req.pickup_lng}_{req.dropoff_lat}_{req.dropoff_lng}"
        
        # Check cache
        cached = route_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        # Calculate primary route with timeout protection
        distance_m = calculate_distance(
            req.pickup_lat, req.pickup_lng,
            req.dropoff_lat, req.dropoff_lng
        )
        duration_s = estimate_duration(distance_m)
        
        # Validate region if provided
        if req.driver_region:
            logger.info(f"Using region-specific routing for {req.driver_region}")
        
        # Get real road routing - no fallback
        coords = generate_route_coordinates(
            req.pickup_lat, req.pickup_lng,
            req.dropoff_lat, req.dropoff_lng,
            region=req.driver_region
        )
        
        primary = Route(
            coordinates=coords,
            distance_m=distance_m,
            distance_km=round(distance_m / 1000, 2),
            duration_seconds=duration_s,
            duration_minutes=round(duration_s / 60, 0)
        )
        
        # Generate alternatives if requested
        alternatives = None
        if req.alternatives and req.alternatives > 0:
            alternatives = generate_alternative_routes(
                req.pickup_lat, req.pickup_lng,
                req.dropoff_lat, req.dropoff_lng,
                req.alternatives
            )
        
        response = RouteResponse(primary=primary, alternatives=alternatives)
        
        # Cache the result
        route_cache.set(cache_key, response)
        
        logger.info(f"Route calculated: {distance_m:.0f}m, {duration_s}s")
        return response
        
    except Exception as e:
        logger.error(f"Error calculating route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/geocode", response_model=GeocodeResponse)
async def geocode(req: GeocodeRequest):
    """
    Geocode an address to coordinates
    In production, this would call Google Maps or OSM Nominatim
    """
    try:
        # Placeholder implementation
        # In production, integrate with Google Maps API or OSM
        logger.info(f"Geocoding: {req.address}")
        
        # For demo purposes, return mock coordinates
        # Real implementation would call external API
        return GeocodeResponse(
            address=req.address,
            lat=0.0,
            lng=0.0
        )
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "size": len(route_cache.cache),
        "max_size": route_cache.max_size,
        "ttl_seconds": route_cache.ttl_seconds
    }


@app.delete("/cache")
async def clear_cache():
    """Clear route cache"""
    route_cache.clear()
    return {"message": "Cache cleared"}


@app.get("/distance")
async def get_distance(
    lat1: float = Query(...),
    lng1: float = Query(...),
    lat2: float = Query(...),
    lng2: float = Query(...),
):
    """Calculate straight-line distance between two points"""
    try:
        distance_m = calculate_distance(lat1, lng1, lat2, lng2)
        return {
            "distance_meters": round(distance_m, 2),
            "distance_km": round(distance_m / 1000, 2),
            "distance_miles": round(distance_m / 1609.34, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===================== WEBSOCKET ENDPOINT =====================

@app.websocket("/ws/route")
async def websocket_route(websocket: WebSocket):
    """
    WebSocket endpoint for real-time route calculation
    Client sends: {"pickup_lat": float, "pickup_lng": float, "dropoff_lat": float, "dropoff_lng": float}
    Server responds with: {"primary": {"coordinates": [...], "distance_km": ..., "duration_minutes": ...}}
    """
    await websocket.accept()
    logger.info("[WS] New route calculation client connected")
    
    try:
        while True:
            # Receive route request from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            logger.info(f"[WS] Route request received: {request_data}")
            
            try:
                # Extract coordinates
                pickup_lat = request_data.get("pickup_lat")
                pickup_lng = request_data.get("pickup_lng")
                dropoff_lat = request_data.get("dropoff_lat")
                dropoff_lng = request_data.get("dropoff_lng")
                region = request_data.get("driver_region")
                
                if not all([pickup_lat, pickup_lng, dropoff_lat, dropoff_lng]):
                    await websocket.send_json({
                        "error": "Missing required coordinates"
                    })
                    continue
                
                # Calculate route
                distance_m = calculate_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
                duration_s = estimate_duration(distance_m)
                
                # Get real roads route or fallback
                coords = generate_route_coordinates(
                    pickup_lat, pickup_lng,
                    dropoff_lat, dropoff_lng,
                    region=region
                )
                
                # Send route response via WebSocket
                response = {
                    "primary": {
                        "coordinates": coords,
                        "distance_m": distance_m,
                        "distance_km": round(distance_m / 1000, 2),
                        "duration_seconds": duration_s,
                        "duration_minutes": round(duration_s / 60, 1)
                    }
                }
                
                logger.info(f"[WS] Sending route with {len(coords)} waypoints")
                await websocket.send_json(response)
                
            except Exception as e:
                logger.error(f"[WS] Route calculation error: {e}")
                await websocket.send_json({
                    "error": f"Route calculation failed: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("[WS] Route client disconnected")
    except Exception as e:
        logger.error(f"[WS] WebSocket error: {e}")


# ===================== ERROR HANDLERS =====================

@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found"}
    )


@app.exception_handler(500)
async def server_error(request, exc):
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ===================== STARTUP/SHUTDOWN =====================

@app.on_event("startup")
async def startup():
    logger.info("Routing service starting up")
    
    # Download GraphML files from Google Drive
    if GDRIVE_AVAILABLE:
        logger.info("Downloading GraphML files from Google Drive...")
        try:
            download_all_graphml_files()
            logger.info("GraphML files download complete")
        except Exception as e:
            logger.error(f"Failed to download GraphML files: {e}")
    else:
        logger.warning("Google Drive download not available, assuming files exist locally")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Routing service shutting down")


# ===================== RUN =====================

if __name__ == "__main__":
    import uvicorn
    # Use PORT for Railway, fallback to ROUTING_PORT or 8010 for local
    port = int(os.environ.get("PORT") or os.environ.get("ROUTING_PORT", 8010))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
