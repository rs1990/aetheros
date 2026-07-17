import { Router } from 'express';
import { db } from '../db/client.js';

const router = Router();

/**
 * GET /landmarks/nearby?lat=37.7&lon=-122.4&radius=5
 * Returns landmarks within `radius` km of a point.
 * Reads from the pre-generated cache — no live generation.
 */
router.get('/nearby', async (req, res, next) => {
  try {
    const lat    = parseFloat(req.query.lat);
    const lon    = parseFloat(req.query.lon);
    const radius = parseFloat(req.query.radius ?? 5);
    const limit  = parseInt(req.query.limit ?? 20, 10);

    if (isNaN(lat) || isNaN(lon)) {
      return res.status(400).json({ error: 'lat and lon are required' });
    }

    const { rows } = await db.query(`
      SELECT
        l.id, l.wikidata_id, l.name, l.description,
        l.latitude, l.longitude, l.category, l.image_url,
        ST_Distance(l.location, ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography) AS distance_m,
        (
          SELECT COUNT(*) FROM landmark_content lc
          WHERE lc.landmark_id = l.id AND lc.status IN ('reviewed', 'published')
        ) AS content_count
      FROM landmarks l
      WHERE ST_DWithin(
        l.location,
        ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
        $3 * 1000
      )
      ORDER BY distance_m
      LIMIT $4
    `, [lat, lon, radius, limit]);

    res.json({ landmarks: rows });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /landmarks/:id
 * Returns full landmark detail.
 */
router.get('/:id', async (req, res, next) => {
  try {
    const { rows: [landmark] } = await db.query(
      `SELECT * FROM landmarks WHERE id = $1 OR wikidata_id = $1`,
      [req.params.id]
    );
    if (!landmark) return res.status(404).json({ error: 'Landmark not found' });
    res.json({ landmark });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /landmarks/:id/content?type=ambient&variant=history
 * Returns pre-generated narration content for a landmark.
 * Status must be 'reviewed' or 'published' — pending content is never served to app.
 */
router.get('/:id/content', async (req, res, next) => {
  try {
    const { type, variant, length } = req.query;

    let sql = `
      SELECT lc.* FROM landmark_content lc
      JOIN landmarks l ON l.id = lc.landmark_id
      WHERE (l.id::text = $1 OR l.wikidata_id = $1)
        AND lc.status IN ('reviewed', 'published')
    `;
    const params = [req.params.id];
    let idx = 2;

    if (type)    { sql += ` AND lc.content_type = $${idx++}`;  params.push(type);    }
    if (variant) { sql += ` AND lc.variant = $${idx++}`;       params.push(variant); }
    if (length)  { sql += ` AND lc.length = $${idx++}`;        params.push(length);  }

    sql += ' ORDER BY lc.content_type, lc.variant';

    const { rows } = await db.query(sql, params);
    res.json({ content: rows });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /landmarks/along-route?points=lat,lon|lat,lon|...&radius=2
 * Returns landmarks within radius km of a polyline (route points).
 * Points encoded as "lat,lon" pairs separated by |.
 */
router.get('/along-route', async (req, res, next) => {
  try {
    const { points, radius = 2 } = req.query;
    if (!points) return res.status(400).json({ error: 'points required' });

    const coords = points.split('|').map(p => {
      const [lat, lon] = p.split(',').map(Number);
      return [lat, lon];
    });

    if (coords.length < 2) return res.status(400).json({ error: 'at least 2 points required' });

    // Build a WKT LineString for the route
    const wkt = `LINESTRING(${coords.map(([lat, lon]) => `${lon} ${lat}`).join(',')})`;

    const { rows } = await db.query(`
      SELECT
        l.id, l.wikidata_id, l.name, l.description,
        l.latitude, l.longitude, l.image_url, l.category,
        ST_Distance(
          l.location,
          ST_SetSRID(ST_GeomFromText($1), 4326)::geography
        ) AS distance_to_route_m
      FROM landmarks l
      WHERE ST_DWithin(
        l.location,
        ST_SetSRID(ST_GeomFromText($1), 4326)::geography,
        $2 * 1000
      )
      ORDER BY distance_to_route_m
      LIMIT 30
    `, [wkt, radius]);

    res.json({ landmarks: rows });
  } catch (err) {
    next(err);
  }
});

export default router;
