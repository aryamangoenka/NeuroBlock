// Get the current environment
const isDevelopment = import.meta.env.DEV;

// Define API URLs for different environments
const DEV_API_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8080';
const PROD_API_URL = 'https://dnd-neural-backend-76136455379.us-central1.run.app';
const CUSTOM_DOMAIN_API_URL = 'https://api.neuroblock.co';

// Export the appropriate API URL based on environment
const API_BASE_URL = isDevelopment 
  ? DEV_API_URL 
  : window.location.hostname.includes('neuroblock.co')
    ? CUSTOM_DOMAIN_API_URL
    : PROD_API_URL;

// Add a function to get the WebSocket URL
export function getSocketUrl(): string {
  const baseUrl = API_BASE_URL;
  // Convert http:// to ws:// and https:// to wss://
  return baseUrl.replace(/^http/, 'ws');
}

export default API_BASE_URL;
