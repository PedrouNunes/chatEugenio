import math
from typing import Optional, Tuple

from smolagents import tool


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point.
        destination_coords: Tuple of (latitude, longitude) for the destination.
        cruising_speed_kmh: Optional cruising speed in km/h.

    Returns:
        Estimated travel time in hours.
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    earth_radius_km = 6371.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = earth_radius_km * c

    actual_distance = distance * 1.1
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    return round(flight_time, 2)


if __name__ == "__main__":
    result = calculate_cargo_travel_time(
        (41.8781, -87.6298),
        (-33.8688, 151.2093),
    )
    print(result)