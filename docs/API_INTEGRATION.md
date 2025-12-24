# API Integration Guide

## Overview

The real-time satisfaction detector supports automatic integration with a Session Summaries API. When configured, the detector will automatically send session data to your API when the session ends.

## Features

### Automatic API Integration
- Automatically sends session summaries to the API when the detector session ends
- Includes all session statistics: duration, predictions, confidence levels, etc.
- Graceful error handling with fallback to local display

### Command Line Arguments

```bash
--api-url URL          # API base URL (default: https://laravel-api.fly.dev)
--session-id ID        # Optional session ID for tracking
```

### API Connection Testing
- Automatically tests API connectivity on startup
- Provides clear feedback about API status

## Usage Examples

### Basic Usage with API Integration

```bash
# Run with default API integration
python realtime_satisfaction_detector.py

# Run with custom session ID
python realtime_satisfaction_detector.py --session-id "store_001_morning"

# Run with custom API URL
python realtime_satisfaction_detector.py --api-url "https://your-api.com"

# Run with all options
python realtime_satisfaction_detector.py \
    --session-id "demo_session" \
    --camera 0 \
    --min-face-size 80
```

### Local Mode (No API)

```bash
# Run without API (local display only)
python realtime_satisfaction_detector.py --api-url ""
```

## API Data Format

The detector sends data to the `/api/session-summaries` endpoint:

```json
{
    "session_duration": 15.5,
    "total_predictions": 10,
    "satisfied_count": 7,
    "neutral_count": 2,
    "unsatisfied_count": 1,
    "average_confidence": 0.85,
    "most_common_prediction": "satisfied",
    "session_id": "optional-session-id"
}
```

## Requirements

### Python Dependencies

```bash
pip install requests
```

### API Server

The detector will:
1. Test connectivity to `/api/session-summaries/statistics` on startup
2. Send session data to `/api/session-summaries` when the session ends

## Error Handling

The detector includes robust error handling:
- **Connection Errors**: If API is unreachable, continues normally with local display
- **API Errors**: Logs errors and continues
- **Timeout Protection**: API calls have 20-second timeout

## Session Data Mapping

| Detector Data | API Field | Description |
|---------------|-----------|-------------|
| Session duration | `session_duration` | Total session time in seconds |
| Total predictions | `total_predictions` | Number of predictions made |
| Satisfied count | `satisfied_count` | Number of "Satisfied" predictions |
| Neutral count | `neutral_count` | Number of "Neutral" predictions |
| Unsatisfied count | `unsatisfied_count` | Number of "Unsatisfied" predictions |
| Average confidence | `average_confidence` | Mean confidence (0.0-1.0) |
| Most common | `most_common_prediction` | Most frequent prediction type |
| Session ID | `session_id` | Optional session identifier |

## Troubleshooting

### API Connection Issues

1. Check API server is running
2. Verify the API URL is correct
3. Ensure network connectivity
4. Check firewall settings

### Common Error Messages

- `❌ API connection test failed`: API server not reachable
- `⚠️ API returned status code: XXX`: API server error
- `❌ Error sending data to API`: Network/connection error
