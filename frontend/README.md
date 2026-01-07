# Stock Prediction Dashboard - Frontend

React + TypeScript + Vite dashboard for stock price predictions using LSTM model.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:5001`

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`

## 📦 Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **shadcn/ui** - Component library
- **Recharts** - Charts
- **Axios** - HTTP client
- **Tailwind CSS** - Styling

## 🏗️ Project Structure

```
src/
├── components/
│   ├── ui/              # shadcn components
│   ├── StockSelector.tsx
│   ├── PredictionChart.tsx
│   ├── PredictionCard.tsx
│   ├── LoadingSpinner.tsx
│   └── ErrorMessage.tsx
├── services/
│   └── api.ts           # API client
├── types/
│   └── index.ts         # TypeScript types
├── hooks/
│   └── usePrediction.ts # Custom hook
├── lib/
│   └── utils.ts         # Utilities
├── App.tsx              # Main dashboard
└── main.tsx             # Entry point
```

## 🔌 API Integration

The app connects to the backend API with these endpoints:

- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Stock prediction

Configure API URL in `.env`:
```env
VITE_API_URL=http://localhost:5001
```

## 🎨 Features

- 📊 Interactive stock price predictions
- 📈 Real-time charts with Recharts
- 🎯 10+ popular stock tickers (US + Brazil)
- ⚡ Loading states with skeletons
- 🚨 Error handling with user-friendly messages
- 📱 Fully responsive design
- 🌙 Dark mode support (via shadcn)
- ♿ Accessible components

## 🧪 Available Scripts

```bash
# Development
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type check
tsc --noEmit
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:5001` |

## 🎯 Stock Tickers Available

**US Stocks:**
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- TSLA (Tesla)
- NVDA (NVIDIA)
- META (Meta)

**Brazil (B3):**
- PETR4.SA (Petrobras)
- VALE3.SA (Vale)
- ITUB4.SA (Itaú)

## 🔧 Development

### Adding New Components

```bash
# Add shadcn component
npx shadcn@latest add [component-name]
```

### Code Style

- TypeScript strict mode enabled
- ESLint + Prettier configured
- Path aliases: `@/` → `src/`

## 📄 License

MIT
