import * as Location from 'expo-location';

export async function requestLocationPermission() {
  const { status } = await Location.requestForegroundPermissionsAsync();
  return status === 'granted';
}

export async function getCurrentLocation() {
  const location = await Location.getCurrentPositionAsync({
    accuracy: Location.Accuracy.Balanced,
  });
  return {
    lat: location.coords.latitude,
    lon: location.coords.longitude,
    accuracy: location.coords.accuracy,
  };
}

/**
 * Watch position and call onLocation with {lat, lon} updates.
 * Returns a cleanup function to stop watching.
 */
export async function watchLocation(onLocation, onError) {
  const sub = await Location.watchPositionAsync(
    {
      accuracy: Location.Accuracy.Balanced,
      distanceInterval: 20, // update every 20m of movement
      timeInterval: 5000,
    },
    (loc) => onLocation({
      lat: loc.coords.latitude,
      lon: loc.coords.longitude,
    })
  );
  return () => sub.remove();
}

/**
 * Check if a point is within `radiusM` meters of a landmark.
 * Used for geofence triggers in the live tour.
 */
export function isWithinRadius(userLat, userLon, targetLat, targetLon, radiusM) {
  const R = 6371000; // Earth radius in meters
  const φ1 = (userLat * Math.PI) / 180;
  const φ2 = (targetLat * Math.PI) / 180;
  const Δφ = ((targetLat - userLat) * Math.PI) / 180;
  const Δλ = ((targetLon - userLon) * Math.PI) / 180;

  const a =
    Math.sin(Δφ / 2) ** 2 +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c <= radiusM;
}
