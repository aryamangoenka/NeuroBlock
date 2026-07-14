// API base URL configuration.
//
// Development: Vite dev server on :5173, backend runs separately on :8080
//   (override with VITE_BACKEND_URL in frontend/.env).
// Production: single container — Flask serves this built app AND the API,
//   so everything is same-origin. VITE_BACKEND_URL at build time can still
//   override for a split deployment.

const isDevelopment = import.meta.env.DEV;

const DEV_API_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8080';

const API_BASE_URL: string = isDevelopment
  ? DEV_API_URL
  : import.meta.env.VITE_BACKEND_URL ?? window.location.origin;

// Socket.IO accepts the same http(s) origin URL
export function getSocketUrl(): string {
  return API_BASE_URL;
}

export default API_BASE_URL;
