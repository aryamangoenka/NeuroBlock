# Custom Dataset API Service

This directory contains the TypeScript API service for managing custom datasets in the NeuroBlock frontend application.

## Files

- `customDatasetApi.ts` - Main API service with TypeScript interfaces and functions
- `customDatasetApi.test.ts` - Comprehensive examples and usage patterns
- `apiConfig.ts` - API configuration and base URL
- `README.md` - This documentation file

## Quick Start

```typescript
import {
  previewDataset,
  createCustomDataset,
  getCustomDatasets,
  CustomDatasetApiError,
  type DatasetConfig,
} from "../utils/customDatasetApi";

// Preview a dataset file
const preview = await previewDataset(file);

// Create a custom dataset
const config: DatasetConfig = {
  dataset_name: "my_dataset",
  target_column: "target",
  feature_columns: ["feature1", "feature2"],
  task_type: "classification",
};
const result = await createCustomDataset(file, config);
```

## API Functions

### Core Functions

#### `previewDataset(file: File): Promise<DatasetPreview>`

Upload a file and get comprehensive data analysis including:

- Column information and data types
- Missing value analysis
- Statistical summaries
- Sample data preview
- Shape and metadata

**Example:**

```typescript
try {
  const preview = await previewDataset(file);
  console.log("Dataset shape:", preview.shape);
  console.log("Columns:", preview.columns);
  console.log("Data types:", preview.data_types);
} catch (error) {
  if (error instanceof CustomDatasetApiError) {
    console.error("API Error:", error.message);
  }
}
```

#### `validateDatasetConfig(config: DatasetConfig, fileInfo: Partial<DatasetPreview>): Promise<ValidationResult>`

Validate preprocessing configuration against dataset information.

**Example:**

```typescript
const validation = await validateDatasetConfig(config, preview);
if (validation.valid) {
  console.log("Configuration is valid!");
} else {
  console.log("Errors:", validation.errors);
}
```

#### `createCustomDataset(file: File, config: DatasetConfig): Promise<CreateResult>`

Create and save a custom dataset with preprocessing.

**Example:**

```typescript
const result = await createCustomDataset(file, {
  dataset_name: "my_custom_dataset",
  target_column: "target",
  feature_columns: ["feature1", "feature2", "feature3"],
  task_type: "classification",
  missing_value_strategy: "drop",
});
```

#### `getCustomDatasets(): Promise<CustomDataset[]>`

Get a list of all created custom datasets.

**Example:**

```typescript
const datasets = await getCustomDatasets();
datasets.forEach((dataset) => {
  console.log(`${dataset.name}: ${dataset.task_type}, shape: ${dataset.shape}`);
});
```

### Additional Functions

#### `getAvailableDatasets(): Promise<AvailableDatasets>`

Get all available datasets (built-in and custom) for frontend selection.

#### `deleteCustomDataset(datasetName: string): Promise<{success: boolean, message: string}>`

Delete a custom dataset by name.

### Utility Functions

#### `formatFileSize(bytes: number): string`

Format file size for display (e.g., "1.5 MB").

#### `formatDate(dateString: string): string`

Format ISO date string for display.

#### `getDataTypeColor(dataType: string): string`

Get color hex codes for different data types for UI styling.

#### `validateConfigLocally(config: Partial<DatasetConfig>): string[]`

Validate dataset configuration locally before server submission.

## TypeScript Interfaces

### `DatasetPreview`

```typescript
interface DatasetPreview {
  columns: string[];
  rows: any[][];
  shape: number[];
  data_types: Record<string, string>;
  missing_values: Record<string, number>;
  missing_percentages: Record<string, number>;
  numeric_statistics: Record<string, StatsSummary>;
  categorical_info: Record<string, CategoryInfo>;
  filename: string;
  row_limit_exceeded: boolean;
  analysis_timestamp: string;
}
```

### `DatasetConfig`

```typescript
interface DatasetConfig {
  dataset_name: string;
  target_column: string;
  feature_columns: string[];
  task_type: "classification" | "regression";
  missing_value_strategy?: "drop" | "fill_mean" | "fill_mode";
}
```

### `ValidationResult`

```typescript
interface ValidationResult {
  valid: boolean;
  warnings: string[];
  errors: string[];
  recommendations: string[];
}
```

### `CreateResult`

```typescript
interface CreateResult {
  success: boolean;
  dataset: {
    name: string;
    shape: number[];
    task_type: string;
    feature_count: number;
    class_labels: string[] | null;
  };
  message: string;
}
```

### `CustomDataset`

```typescript
interface CustomDataset {
  name: string;
  task_type: string;
  shape: number[];
  feature_count: number;
  target_column: string;
  created_at: string;
  original_filename: string;
  class_labels: string[] | null;
}
```

## Error Handling

The API service uses a custom `CustomDatasetApiError` class that extends the standard `Error` class:

```typescript
try {
  const result = await previewDataset(file);
} catch (error) {
  if (error instanceof CustomDatasetApiError) {
    console.error("API Error:", error.message);
    console.error("Status Code:", error.status);
    console.error("Response Data:", error.response);
  } else {
    console.error("Unexpected error:", error);
  }
}
```

### Common Error Scenarios

- **400 Bad Request**: Invalid file format, missing parameters, validation errors
- **413 Payload Too Large**: File size exceeds 50MB limit
- **500 Internal Server Error**: Server-side processing errors
- **Network Errors**: Connection issues, timeout errors

## React Integration Examples

### File Upload Component

```typescript
import React, { useState, useCallback } from "react";
import {
  previewDataset,
  CustomDatasetApiError,
} from "../utils/customDatasetApi";

export function FileUpload() {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);

    try {
      const previewData = await previewDataset(file);
      setPreview(previewData);
    } catch (err) {
      if (err instanceof CustomDatasetApiError) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Component JSX...
}
```

### Dataset Management Hook

```typescript
import { useState, useEffect, useCallback } from "react";
import {
  getCustomDatasets,
  deleteCustomDataset,
} from "../utils/customDatasetApi";

export function useCustomDatasets() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);

  const refreshDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getCustomDatasets();
      setDatasets(data);
    } catch (error) {
      console.error("Failed to fetch datasets:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteDataset = useCallback(
    async (name: string) => {
      try {
        await deleteCustomDataset(name);
        await refreshDatasets(); // Refresh the list
      } catch (error) {
        console.error("Failed to delete dataset:", error);
      }
    },
    [refreshDatasets]
  );

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  return { datasets, loading, refreshDatasets, deleteDataset };
}
```

## File Constraints

- **Supported formats**: CSV (`.csv`), Excel (`.xlsx`, `.xls`)
- **Maximum file size**: 50MB
- **Maximum rows**: 100,000 rows
- **Encoding support**: UTF-8, Latin-1, CP1252, ISO-8859-1 for CSV files

## Configuration Options

### Missing Value Strategies

- `drop`: Remove rows with missing values
- `fill_mean`: Fill numeric columns with mean values
- `fill_mode`: Fill columns with mode (most frequent) values

### Task Types

- `classification`: For categorical target variables
- `regression`: For continuous numeric target variables

## Best Practices

1. **Always handle errors**: Use try-catch blocks and check for `CustomDatasetApiError`
2. **Validate locally first**: Use `validateConfigLocally()` for immediate feedback
3. **Show loading states**: API calls can take time, especially for large files
4. **Provide user feedback**: Display warnings and recommendations from validation
5. **Refresh data after modifications**: Update lists after creating/deleting datasets

## API Base URL Configuration

The API base URL is configured in `apiConfig.ts`:

```typescript
const API_BASE_URL =
  import.meta.env.VITE_API_URL || "https://api.neuroblock.co";
```

For local development, set the `VITE_API_URL` environment variable:

```bash
VITE_API_URL=http://localhost:5000
```

## Testing

See `customDatasetApi.test.ts` for comprehensive usage examples and test patterns that can be adapted for your components.

## Integration with Neural Network Training

Custom datasets created through this API are automatically available in the training pipeline. They appear alongside built-in datasets in the dataset selection interface and can be used immediately for model training without additional configuration.

## Support

For issues or questions about the custom dataset API:

1. Check the error message and status code
2. Refer to the backend API documentation
3. Review the example usage patterns in the test file
4. Ensure your API base URL and network connectivity are correct
