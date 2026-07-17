import cors from 'cors';

const ALLOWED_ORIGINS = [
  'http://localhost:3000',
  'http://localhost:19006',  // Expo web
  'http://localhost:8081',   // Expo Metro bundler
];

export default cors({
  origin: (origin, cb) => {
    if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
    cb(new Error(`CORS blocked: ${origin}`));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
});
