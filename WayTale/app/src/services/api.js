/**
 * WayTale backend API client.
 * All routes read from the pre-generated content cache — no live LLM calls for landmarks.
 */

const BASE_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3001';

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? `API error ${res.status}`);
  }
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? `API error ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => get('/health'),

  landmarks: {
    nearby: (lat, lon, radius = 5) =>
      get(`/landmarks/nearby?lat=${lat}&lon=${lon}&radius=${radius}`),
    get: (id) =>
      get(`/landmarks/${id}`),
    content: (id, { type, variant, length } = {}) => {
      const params = new URLSearchParams();
      if (type)    params.set('type', type);
      if (variant) params.set('variant', variant);
      if (length)  params.set('length', length);
      return get(`/landmarks/${id}/content?${params}`);
    },
    alongRoute: (points, radius = 2) => {
      const encoded = points.map(([lat, lon]) => `${lat},${lon}`).join('|');
      return get(`/landmarks/along-route?points=${encodeURIComponent(encoded)}&radius=${radius}`);
    },
  },

  routes: {
    score: (origin, destination, departureTime) =>
      post('/route/score', { origin, destination, departureTime }),
  },
};
