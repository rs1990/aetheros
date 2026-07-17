/**
 * Route scoring endpoint.
 * Phase 2: takes candidate route options and scores them by:
 *   (a) travel time
 *   (b) landmark count/quality along route
 *   (c) crowd penalty heuristic (time-of-day + day-of-week)
 *
 * For now: stub route provider (returns mock geometry).
 * Swap in Google Directions / Mapbox via MAPS_PROVIDER env var.
 */

import { Router } from 'express';
import { db } from '../db/client.js';

const router = Router();

/**
 * POST /route/score
 * Body: { origin: {lat, lon}, destination: {lat, lon}, departureTime?: ISO }
 */
router.post('/score', async (req, res, next) => {
  try {
    const { origin, destination, departureTime } = req.body;

    if (!origin?.lat || !origin?.lon || !destination?.lat || !destination?.lon) {
      return res.status(400).json({ error: 'origin and destination {lat, lon} required' });
    }

    const departure = departureTime ? new Date(departureTime) : new Date();

    // Stub route options — replace with real Directions API call
    const routeOptions = generateStubRoutes(origin, destination);

    // Score each route
    const scored = await Promise.all(routeOptions.map(async (route) => {
      // Find landmarks along this route
      const wkt = `LINESTRING(${route.points.map(([lat, lon]) => `${lon} ${lat}`).join(',')})`;
      const { rows: landmarks } = await db.query(`
        SELECT l.id, l.name, l.latitude, l.longitude,
          ST_Distance(l.location, ST_SetSRID(ST_GeomFromText($1), 4326)::geography) AS dist_m
        FROM landmarks l
        WHERE ST_DWithin(l.location, ST_SetSRID(ST_GeomFromText($1), 4326)::geography, 2000)
        ORDER BY dist_m
        LIMIT 15
      `, [wkt]);

      const crowdPenalty = computeCrowdPenalty(departure, landmarks.length);
      const landmarkScore = Math.min(landmarks.length * 10, 50);
      const timeScore = Math.max(0, 50 - route.duration_minutes);
      const totalScore = timeScore + landmarkScore - crowdPenalty;

      return {
        ...route,
        landmarks: landmarks.slice(0, 5),
        landmark_count: landmarks.length,
        crowd_penalty: crowdPenalty,
        score: Math.round(totalScore),
        label: null, // assigned after sorting
      };
    }));

    // Sort and label
    scored.sort((a, b) => b.score - a.score);
    const labels = ['Best balance', 'Most landmarks', 'Fastest'];
    scored.forEach((r, i) => { r.label = labels[i] ?? `Option ${i+1}`; });

    res.json({ routes: scored, departure: departure.toISOString() });
  } catch (err) {
    next(err);
  }
});

function generateStubRoutes(origin, destination) {
  // Stubs 3 route variants with interpolated midpoints
  const midLat = (origin.lat + destination.lat) / 2;
  const midLon = (origin.lon + destination.lon) / 2;

  return [
    {
      id: 'direct',
      label: 'Fastest',
      duration_minutes: 25,
      distance_km: 12,
      points: [
        [origin.lat, origin.lon],
        [midLat, midLon],
        [destination.lat, destination.lon],
      ],
    },
    {
      id: 'scenic_north',
      label: 'Scenic North',
      duration_minutes: 38,
      distance_km: 18,
      points: [
        [origin.lat, origin.lon],
        [midLat + 0.02, midLon - 0.01],
        [midLat + 0.01, midLon + 0.01],
        [destination.lat, destination.lon],
      ],
    },
    {
      id: 'scenic_south',
      label: 'Scenic South',
      duration_minutes: 42,
      distance_km: 20,
      points: [
        [origin.lat, origin.lon],
        [midLat - 0.02, midLon + 0.01],
        [midLat - 0.01, midLon - 0.01],
        [destination.lat, destination.lon],
      ],
    },
  ];
}

/**
 * Time-of-day + day-of-week heuristic crowd penalty.
 * Returns 0–20. Higher = more crowded = more penalty.
 * No live data needed — this is the universal fallback per spec.
 */
function computeCrowdPenalty(departure, landmarkCount) {
  const hour = departure.getHours();
  const dow  = departure.getDay(); // 0=Sun, 6=Sat
  const isWeekend = dow === 0 || dow === 6;
  const isPeakHour = hour >= 10 && hour <= 16;

  let penalty = 0;
  if (isWeekend) penalty += 8;
  if (isPeakHour) penalty += 7;
  if (landmarkCount > 5) penalty += 5; // more landmarks = more congestion
  return Math.min(penalty, 20);
}

export default router;
