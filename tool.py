import googlemaps
import os
import json


def get_current_position() -> dict:
    """
    Retrieves the current geographic position using the Google Maps API.

    This function uses a geolocation request to determine the latitude and longitude 
    of the device and then performs a reverse geocoding operation to obtain the 
    formatted address.

    Returns:
        dict: A dictionary containing the following keys:
            - "lat" (float): The latitude of the current position.
            - "lng" (float): The longitude of the current position.
            - "address" (str): The formatted address corresponding to the location.
    """
    cwd = os.path.dirname(__file__)
    cred_path = os.path.join(cwd, "credentials", "credentials_google.json")
    with open(cred_path, "r") as f:
        creds = json.load(f)
        KEY_API_GOOGLE_MAPS = creds["KEY_API_GOOGLE_MAPS"]

    gmaps = googlemaps.Client(key=KEY_API_GOOGLE_MAPS)
    response = gmaps.geolocate()
    latitude = response["location"]["lat"]
    longitude = response["location"]["lng"]

    result = gmaps.reverse_geocode((latitude, longitude))
    position = {
        "lat": latitude,
        "lng": longitude,
        "address": result[0]["formatted_address"],
    }
    return position





def get_supermarkets_nearby(latitude, longitude, radius=2000):
    """

    Retrieves a list of supermarkets within a specified radius from a given latitude and longitude 
    using the Google Maps Places API.

    Args:
        latitude (float): The latitude of the reference location.
        longitude (float): The longitude of the reference location.
        radius (int, optional): The search radius in meters. Default is 2000 meters (2 km).

    Returns:
        list: A list of dictionaries, each containing information about a supermarket. 
              Each dictionary includes:
              - 'name' (str): The name of the supermarket.
              - 'lat' (float): The latitude of the supermarket.
              - 'lon' (float): The longitude of the supermarket.
              - 'distance_m' (str): The vicinity (address or description) of the supermarket.
    
    """

    cwd = os.path.dirname(__file__)
    cred_path = os.path.join(cwd, "credentials", "credentials_google.json")
    with open(cred_path, "r") as f:
        creds = json.load(f)
        KEY_API_GOOGLE_MAPS = creds["KEY_API_GOOGLE_MAPS"]
    gmaps = googlemaps.Client(key=KEY_API_GOOGLE_MAPS)

    # Request supermarkets (place type: supermarket) within a 2 km radius
    places_result = gmaps.places_nearby((latitude, longitude), radius=radius, type="supermarket")

    supermarkets = []
    for place in places_result.get('results', []):
        supermarkets.append({
            'name': place.get('name', 'Unknown'),
            'lat': place['geometry']['location']['lat'],
            'lon': place['geometry']['location']['lng'],
            'distance_m': place.get('vicinity', 'Unknown')
        })
    
    return supermarkets