import { useState, useEffect } from 'react';
import { api } from '../services/api';

export function useNearbyLandmarks(location, radius = 5) {
  const [landmarks, setLandmarks] = useState([]);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  useEffect(() => {
    if (!location) return;

    let cancelled = false;
    setLoading(true);

    api.landmarks.nearby(location.lat, location.lon, radius)
      .then(({ landmarks }) => {
        if (!cancelled) setLandmarks(landmarks);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [location?.lat, location?.lon, radius]);

  return { landmarks, loading, error };
}
