# Custom Dataset Management API

This document describes the custom dataset management endpoints for the NeuroBlock backend.

## Overview

The custom dataset management API allows users to upload, preview, validate, and create custom datasets for training neural networks. The API supports CSV and Excel files with comprehensive data analysis and preprocessing capabilities.

## Endpoints

### 1. POST `/api/datasets/preview`

Upload a file and get a comprehensive data preview.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File upload with key `file`

**Supported File Formats:**

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

**File Constraints:**

- Maximum file size: 50MB
- Maximum rows: 100,000

**Response:**

```json
{
  "success": true,
  "message": "Dataset preview generated successfully",
  "data": {
    "filename": "example.csv",
    "shape": [1000, 5],
    "columns": ["feature1", "feature2", "feature3", "feature4", "target"],
    "data_types": {
      "feature1": "integer",
      "feature2": "float",
      "feature3": "categorical",
      "feature4": "text",
      "target": "integer"
    },
    "missing_values": {
      "feature1": 0,
      "feature2": 5,
      "feature3": 0,
      "feature4": 10,
      "target": 0
    },
    "missing_percentages": {
      "feature1": 0.0,
      "feature2": 0.5,
      "feature3": 0.0,
      "feature4": 1.0,
      "target": 0.0
    },
    "sample_data": [
      {
        "feature1": 1,
        "feature2": 0.1,
        "feature3": "A",
        "feature4": "text1",
        "target": 0
      },
      {
        "feature1": 2,
        "feature2": 0.2,
        "feature3": "B",
        "feature4": "text2",
        "target": 1
      }
    ],
    "numeric_statistics": {
      "feature1": {
        "mean": 500.5,
        "std": 288.67,
        "min": 1.0,
        "max": 1000.0,
        "median": 500.5
      }
    },
    "categorical_info": {
      "feature3": {
        "unique_count": 2,
        "top_values": { "A": 500, "B": 500 }
      }
    },
    "row_limit_exceeded": false,
    "analysis_timestamp": "2024-01-15T10:30:00"
  }
}
```

### 2. POST `/api/datasets/validate`

Validate preprocessing configuration against dataset information.

**Request:**

- Method: `POST`
- Content-Type: `application/json`
- Body:

```json
{
  "preprocessing_config": {
    "target_column": "target",
    "feature_columns": ["feature1", "feature2", "feature3"],
    "task_type": "classification"
  },
  "file_info": {
    "columns": ["feature1", "feature2", "feature3", "target"],
    "data_types": {
      "feature1": "integer",
      "feature2": "float",
      "feature3": "categorical",
      "target": "integer"
    },
    "missing_values": {
      "feature1": 0,
      "feature2": 5,
      "feature3": 0,
      "target": 0
    },
    "missing_percentages": {
      "feature1": 0.0,
      "feature2": 0.5,
      "feature3": 0.0,
      "target": 0.0
    },
    "shape": [1000, 4]
  }
}
```

**Response:**

```json
{
  "success": true,
  "validation_results": {
    "valid": true,
    "warnings": ["Target column \"target\" has 5 missing values"],
    "errors": [],
    "recommendations": ["Categorical features may need encoding: feature3"]
  }
}
```

### 3. POST `/api/datasets/create`

Create and save a custom dataset with preprocessing configuration.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Dataset file
  - `config`: JSON string with configuration

**Configuration Format:**

```json
{
  "dataset_name": "my_custom_dataset",
  "target_column": "target",
  "feature_columns": ["feature1", "feature2", "feature3"],
  "task_type": "classification",
  "missing_value_strategy": "drop"
}
```

**Missing Value Strategies:**

- `drop`: Remove rows with missing values
- `fill_mean`: Fill numeric columns with mean values
- `fill_mode`: Fill columns with mode (most frequent) values

**Response:**

```json
{
  "success": true,
  "message": "Custom dataset created successfully",
  "dataset": {
    "name": "my_custom_dataset",
    "shape": [950, 3],
    "task_type": "classification",
    "feature_count": 3,
    "class_labels": ["class_0", "class_1"]
  }
}
```

### 4. GET `/api/datasets/custom`

List all created custom datasets.

**Request:**

- Method: `GET`
- No body required

**Response:**

```json
{
  "success": true,
  "datasets": [
    {
      "name": "my_custom_dataset",
      "task_type": "classification",
      "shape": [950, 3],
      "feature_count": 3,
      "target_column": "target",
      "created_at": "2024-01-15T10:30:00",
      "original_filename": "data.csv",
      "class_labels": ["class_0", "class_1"]
    }
  ]
}
```

### 5. GET `/api/datasets/available`

Get all available datasets (built-in and custom) for frontend dataset selection.

**Request:**

- Method: `GET`
- No body required

**Response:**

```json
{
  "success": true,
  "datasets": {
    "built_in": [
      {
        "name": "Iris",
        "type": "built_in",
        "description": "Built-in Iris dataset"
      },
      {
        "name": "MNIST",
        "type": "built_in",
        "description": "Built-in MNIST dataset"
      }
    ],
    "custom": [
      {
        "name": "my_custom_dataset",
        "task_type": "classification",
        "shape": [950, 3],
        "feature_count": 3,
        "target_column": "target",
        "created_at": "2024-01-15T10:30:00",
        "original_filename": "data.csv",
        "class_labels": ["class_0", "class_1"]
      }
    ],
    "all_names": [
      "Iris",
      "MNIST",
      "CIFAR-10",
      "California Housing",
      "Breast Cancer",
      "my_custom_dataset"
    ]
  }
}
```

## Data Types

The API automatically categorizes columns into the following data types:

- **integer**: Integer numeric values
- **float**: Floating-point numeric values
- **categorical**: Object type with low unique value ratio (< 10% and < 50 unique values)
- **text**: Object type with high unique value ratio
- **boolean**: Boolean values
- **other**: Any other data type

## Task Types

Supported machine learning task types:

- **classification**: For categorical target variables
- **regression**: For continuous numeric target variables

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "Descriptive error message"
}
```

Common HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `413`: File too large
- `500`: Internal server error

## File Storage

Custom datasets are stored in:

- Data files: `backend/datasets/custom/{dataset_name}.npz`
- Metadata: `backend/datasets/custom/{dataset_name}_metadata.json`

## Testing

Use the provided test script to verify endpoint functionality:

```bash
cd backend
python test_dataset_endpoints.py
```

Make sure the Flask server is running on `localhost:5000` before running tests.

## Integration

To integrate these endpoints with your frontend:

1. Use the `/preview` endpoint to show users dataset information
2. Use the `/validate` endpoint to check configuration before creation
3. Use the `/create` endpoint to save the final dataset
4. Use the `/custom` endpoint to list available custom datasets

The created datasets can then be used in your neural network training pipeline by loading them from the saved `.npz` files.

## DatasetRegistry Integration

Custom datasets are automatically integrated with the existing training pipeline through the extended `DatasetRegistry` class in `backend/dataset_loader.py`.

### Key Features

- **Automatic Discovery**: Custom datasets are automatically discovered and registered when the registry is accessed
- **Seamless Integration**: Custom datasets appear alongside built-in datasets in the training system
- **Standard Format**: All datasets (built-in and custom) return the same `((x_train, y_train), (x_test, y_test))` format
- **Preprocessing**: Automatic feature scaling, label encoding, and train/test splitting

### Usage in Training Pipeline

```python
from backend.dataset_loader import load_dataset, get_available_datasets

# List all available datasets (built-in + custom)
available_datasets = get_available_datasets()
print(f"Available datasets: {available_datasets}")

# Load any dataset (built-in or custom) using the same interface
(x_train, y_train), (x_test, y_test) = load_dataset("my_custom_dataset")
```

### DatasetRegistry Methods

The extended `DatasetRegistry` class provides these methods:

- `register_custom_dataset(dataset_config)` - Register a custom dataset
- `load_custom_dataset(dataset_id)` - Load a specific custom dataset
- `get_custom_datasets()` - List all custom datasets with metadata
- `get_available_datasets()` - List all datasets (built-in + custom)

### Automatic Registration

When you create a custom dataset via the `/create` endpoint, it's automatically registered with the `DatasetRegistry`, making it immediately available for training without any additional steps.

### Data Processing

Custom datasets undergo the same preprocessing as built-in datasets:

1. **Feature Scaling**: StandardScaler normalization
2. **Label Encoding**: Automatic encoding for classification tasks
3. **Train/Test Split**: 80/20 split with stratification for classification
4. **TensorFlow Conversion**: Automatic conversion to TensorFlow tensors

This ensures compatibility with existing neural network architectures and training code.
