# Frontend Setup

This frontend provides a chat interface for the eme bot, matching the styling of the backend landing page.

## Setup

1. Install dependencies:

```bash
npm install
```

2. Create a `.env` file in the frontend directory with:

```
VITE_API_URL=http://localhost:8000
```

3. Start the development server:

```bash
npm run dev
```

## Features

- Chat interface with streaming responses
- Dark theme matching the backend styling
- Geist Mono font for consistency
- Responsive design
- Real-time message streaming

## API Integration

The frontend communicates with the backend via:

- `POST /chat` - Send messages and receive streaming responses
- `GET /health` - Health check endpoint

Make sure your backend server is running on the configured URL before starting the frontend.
