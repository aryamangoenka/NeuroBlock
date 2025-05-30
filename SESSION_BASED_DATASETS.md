# Session-Based Dataset Management

## Overview

This implementation solves the issue where custom datasets were globally shared between all users. Now each user session has its own isolated dataset storage, ensuring privacy and preventing dataset list pollution.

## Key Changes

### 1. Session Management (`backend/main.py`)

- Added Flask session support with `session` import
- Enabled CORS with credentials support: `CORS(app, supports_credentials=True)`
- Added session initialization middleware that creates unique session IDs
- Started automatic cleanup thread for old sessions

### 2. Session Manager (`backend/utils/session_manager.py`)

- **`get_session_id()`**: Creates/retrieves unique session ID
- **`get_session_datasets_dir()`**: Returns session-specific dataset directory
- **`cleanup_old_sessions()`**: Removes old session directories (default: 24 hours)
- **`start_cleanup_thread()`**: Background thread for automatic cleanup
- **Access tracking**: Updates `.last_access` file for cleanup decisions

### 3. Dataset Routes (`backend/api/dataset_routes.py`)

- Updated all endpoints to use `get_session_datasets_dir()` instead of global directory
- **`/create`**: Saves datasets to session-specific directory
- **`/custom`**: Lists only current session's datasets
- **`/available`**: Returns built-in + session-specific custom datasets
- **`/delete/<name>`**: Deletes datasets from current session only

### 4. Dataset Registry (`backend/dataset_loader.py`)

- Removed global `custom_datasets` storage
- Updated methods to work with session-specific storage:
  - **`register_custom_dataset()`**: Now accepts optional `session_id`
  - **`get_custom_datasets()`**: Returns session-specific datasets
  - **`_load_custom_dataset_data()`**: Loads from session directory
  - **`_load_dataset_from_any_session()`**: Fallback for training without session context

### 5. Frontend API (`frontend/src/utils/customDatasetApi.ts`)

- Added `credentials: 'include'` to all API requests for session cookie support
- Updated delete endpoint to use correct route: `/delete/{name}`

## Directory Structure

```
datasets/
├── custom/                    # Old global storage (still exists for backward compatibility)
│   ├── dataset1.npz
│   └── dataset1_metadata.json
└── sessions/                  # New session-based storage
    ├── {session-uuid-1}/
    │   ├── .last_access       # Timestamp for cleanup
    │   ├── user_dataset.npz
    │   └── user_dataset_metadata.json
    └── {session-uuid-2}/
        ├── .last_access
        ├── another_dataset.npz
        └── another_dataset_metadata.json
```

## Session Lifecycle

1. **Session Creation**: Automatic UUID generation on first request
2. **Dataset Storage**: All custom datasets saved to session directory
3. **Access Tracking**: `.last_access` file updated on each directory access
4. **Automatic Cleanup**: Background thread removes sessions older than 24 hours
5. **Training Support**: Fallback mechanism searches all sessions during training

## Benefits

### ✅ Privacy

- Each user's datasets are completely isolated
- No cross-contamination between users

### ✅ Scalability

- No unlimited dataset accumulation
- Automatic cleanup prevents disk space issues

### ✅ User Experience

- Clean dataset list for each user
- No confusion from other users' datasets

### ✅ Backward Compatibility

- Built-in datasets still work normally
- Training pipeline unchanged for built-in datasets

## Configuration

### Cleanup Settings (in `backend/main.py`)

```python
start_cleanup_thread(
    cleanup_interval_hours=6,  # How often to run cleanup
    max_age_hours=24          # Maximum session age before cleanup
)
```

### Session Directory

- Location: `{PROJECT_ROOT}/datasets/sessions/{session_id}/`
- Access tracking: `.last_access` file with Unix timestamp
- Automatic creation on first use

## API Endpoints

All dataset endpoints now work with session-specific storage:

- **GET** `/api/datasets/available` - Returns built-in + session datasets
- **GET** `/api/datasets/custom` - Lists current session's datasets
- **POST** `/api/datasets/create` - Creates dataset in current session
- **DELETE** `/api/datasets/delete/{name}` - Deletes from current session

## Testing

The implementation has been tested and verified:

1. ✅ Server starts successfully with session support
2. ✅ Session directories are created automatically
3. ✅ Each session gets unique UUID-based directory
4. ✅ Access tracking files are created and updated
5. ✅ API endpoints return session-specific data
6. ✅ Built-in datasets remain available to all sessions

## Migration

No migration is required. The system:

- Continues to work with existing built-in datasets
- Creates new session directories as needed
- Maintains backward compatibility
- Automatically cleans up old sessions

## Security Considerations

- Session IDs are UUIDs (cryptographically secure)
- No session data is exposed in API responses
- Automatic cleanup prevents data accumulation
- CORS configured for credential support only from allowed origins
