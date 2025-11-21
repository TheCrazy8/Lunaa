"""Geolocation data integration"""
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    _GEOPY_AVAILABLE = True
except ImportError:
    _GEOPY_AVAILABLE = False

class GeolocationEngine:
    def __init__(self):
        if _GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent="lunaa_ai")
        else:
            self.geolocator = None
        self.current_location = None
    
    def geocode(self, address: str):
        """Convert address to coordinates"""
        if not _GEOPY_AVAILABLE:
            return "Geopy not installed"
        
        try:
            location = self.geolocator.geocode(address)
            if location:
                return {
                    'address': location.address,
                    'latitude': location.latitude,
                    'longitude': location.longitude
                }
            return "Location not found"
        except Exception as e:
            return f"Error geocoding: {e}"
    
    def reverse_geocode(self, latitude: float, longitude: float):
        """Convert coordinates to address"""
        if not _GEOPY_AVAILABLE:
            return "Geopy not installed"
        
        try:
            location = self.geolocator.reverse(f"{latitude}, {longitude}")
            if location:
                return location.address
            return "Address not found"
        except Exception as e:
            return f"Error reverse geocoding: {e}"
    
    def calculate_distance(self, coord1: tuple, coord2: tuple):
        """Calculate distance between two coordinates"""
        if not _GEOPY_AVAILABLE:
            return "Geopy not installed"
        
        try:
            distance = geodesic(coord1, coord2).kilometers
            return f"{distance:.2f} km"
        except Exception as e:
            return f"Error calculating distance: {e}"
    
    def set_current_location(self, address: str):
        """Set current location"""
        result = self.geocode(address)
        if isinstance(result, dict):
            self.current_location = (result['latitude'], result['longitude'])
            return f"Location set to {result['address']}"
        return result
