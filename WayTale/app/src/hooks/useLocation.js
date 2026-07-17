import { useState, useEffect, useRef } from 'react';
import { requestLocationPermission, watchLocation } from '../services/location';

export function useLocation() {
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);
  const [permitted, setPermitted] = useState(false);
  const cleanupRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    (async () => {
      const granted = await requestLocationPermission();
      if (!mounted) return;
      setPermitted(granted);

      if (!granted) {
        setError('Location permission denied');
        return;
      }

      cleanupRef.current = await watchLocation(
        (loc) => { if (mounted) setLocation(loc); },
        (err) => { if (mounted) setError(err.message); }
      );
    })();

    return () => {
      mounted = false;
      cleanupRef.current?.();
    };
  }, []);

  return { location, error, permitted };
}
