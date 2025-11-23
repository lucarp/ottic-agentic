# Ottic Agentic Frontend

Modern React + TypeScript frontend built with Vite, Tailwind CSS, and shadcn/ui components.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS v3** - Utility-first CSS framework
- **shadcn/ui** - High-quality React components built on Radix UI
- **Chart.js** - Data visualization
- **marked** - Markdown parsing
- **highlight.js** - Syntax highlighting

## Development

### Prerequisites
- Node.js 20.19+ or 22.12+ (recommended)
- npm 10+

### Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start dev server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

4. **Preview production build:**
   ```bash
   npm run preview
   ```

### Environment Variables

The WebSocket URL is currently hardcoded to `ws://localhost:8000/ws`. To change it, edit `src/hooks/useWebSocket.ts` or use environment variables.

## Features

### Split-Screen Layout
- **Terminal (60%)**: Real-time chat interface with the agent
- **Artifacts (40%)**: Tabbed view of created artifacts

### Artifact Types Supported
- CSV tables
- Charts (bar, line, pie, doughnut, radar)
- Code with syntax highlighting
- Markdown documents
- HTML content
- Payment links
- Fetched web content
- SEO domain overviews
- Competitor analysis
- Keyword research

## Documentation

For complete documentation, see the full README with detailed information about:
- Project structure
- Component architecture
- Adding new artifact types
- Customization and theming
- Performance optimization
- Troubleshooting

## License

MIT
