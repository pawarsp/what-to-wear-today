from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder


def get_coords_from_location_name(location: str = "Berlin, Germany"):
    """
    Translates place name like "Berlin, Germany" into (latitude, longitude)
    """
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(location)
    if location is None:
        print("Location could not be resolved. Using 'Berlin, Germany' instead.")
        location = geolocator.geocode("Berlin, Germany")
    return (location.latitude, location.longitude)


def get_timezone_from_coords(latitude, longitude):
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    return tz_name
