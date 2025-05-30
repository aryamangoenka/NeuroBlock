import API_BASE_URL from './apiConfig';

// Add type definitions for process.env
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production';
    }
  }
}

// Add a function to get the correct socket URL
export function getSocketUrl(): string {
  return API_BASE_URL;
}

// TypeScript interfaces for custom dataset management
export interface DatasetPreview {
  columns: string[];
  sample_data: any[];  // Changed from 'rows' to 'sample_data' to match backend response
  shape: number[];
  data_types: Record<string, string>;
  missing_values: Record<string, number>;
  missing_percentages: Record<string, number>;
  numeric_statistics: Record<string, {
    mean: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
    median: number | null;
  }>;
  categorical_info: Record<string, {
    unique_count: number;
    top_values: Record<string, number>;
  }>;
  filename: string;
  row_limit_exceeded: boolean;
  analysis_timestamp: string;
}

export interface ValidationResult {
  valid: boolean;
  warnings: string[];
  errors: string[];
  recommendations: string[];
}

export interface CreateResult {
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

export interface CustomDataset {
  name: string;
  task_type: string;
  shape: number[];
  feature_count: number;
  target_column: string;
  created_at: string;
  original_filename: string;
  class_labels: string[] | null;
}

export interface DatasetConfig {
  dataset_name: string;
  target_column: string;
  feature_columns: string[];
  task_type: 'classification' | 'regression';
  missing_value_strategy?: 'drop' | 'fill_mean' | 'fill_mode';
}

export interface AvailableDatasets {
  built_in: Array<{
    name: string;
    type: string;
    description: string;
  }>;
  custom: CustomDataset[];
  all_names: string[];
}

// API error handling
export class CustomDatasetApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'CustomDatasetApiError';
  }
}

// Helper function for making API requests
async function makeApiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  // Use the development URL for local development
  const baseUrl = getSocketUrl();
    
  const url = `${baseUrl}/api/datasets${endpoint}`;
  
  console.log(`🌐 Making API request to: ${url}`);
  console.log(`📋 Request options:`, {
    method: options.method || 'GET',
    credentials: 'include',
    headers: options.headers
  });
  
  try {
    const response = await fetch(url, {
      ...options,
      credentials: 'include',
      headers: {
        ...options.headers,
      },
    });

    console.log(`📡 Response status: ${response.status} ${response.statusText}`);
    console.log(`🍪 Response headers:`, Object.fromEntries(response.headers.entries()));

    // Handle non-2xx responses
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      let errorData = null;

      try {
        errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
        console.log(`❌ Error response data:`, errorData);
      } catch (parseError) {
        // If response is not JSON, use the status text
        console.log(`⚠️ Could not parse error response as JSON`);
      }

      throw new CustomDatasetApiError(errorMessage, response.status, errorData);
    }

    // Parse JSON response
    const data = await response.json();
    console.log(`✅ Response data:`, data);
    return data;
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error(`🚨 API Error:`, error);
      throw error;
    }

    // Handle network errors or other issues
    console.error(`🌐 Network Error:`, error);
    throw new CustomDatasetApiError(
      `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      0,
      null
    );
  }
}

/**
 * Upload a file and get a comprehensive data preview
 * @param file - The file to preview (CSV or Excel)
 * @returns Promise with dataset preview information
 */
export async function previewDataset(file: File): Promise<DatasetPreview> {
  // Validate file type
  const allowedTypes = [
    'text/csv',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel'
  ];
  
  const allowedExtensions = ['.csv', '.xlsx', '.xls'];
  const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
  
  if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
    throw new CustomDatasetApiError(
      'Unsupported file format. Please upload a CSV or Excel file.',
      400
    );
  }

  // Check file size (50MB limit)
  const maxSize = 50 * 1024 * 1024; // 50MB
  if (file.size > maxSize) {
    throw new CustomDatasetApiError(
      `File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds maximum allowed size (50MB)`,
      413
    );
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await makeApiRequest<{ success: boolean; data: DatasetPreview; message: string }>('/preview', {
    method: 'POST',
    body: formData,
  });

  if (!response.success) {
    throw new CustomDatasetApiError('Failed to preview dataset', 500, response);
  }

  console.log('🔍 Raw preview response data:', response.data);
  
  // The backend returns the data nested under 'data', so we extract it
  return response.data;
}

/**
 * Validate preprocessing configuration against dataset information
 * @param config - The preprocessing configuration
 * @param fileInfo - Information about the uploaded file
 * @returns Promise with validation results
 */
export async function validateDatasetConfig(
  config: DatasetConfig,
  fileInfo: Partial<DatasetPreview>
): Promise<ValidationResult> {
  const requestBody = {
    preprocessing_config: config,
    file_info: fileInfo,
  };

  const response = await makeApiRequest<{
    success: boolean;
    validation_results: ValidationResult;
  }>('/validate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.success) {
    throw new CustomDatasetApiError('Failed to validate dataset configuration', 500, response);
  }

  return response.validation_results;
}

/**
 * Create and save a custom dataset with preprocessing configuration
 * @param file - The dataset file
 * @param config - The dataset configuration
 * @returns Promise with creation results
 */
export async function createCustomDataset(
  file: File,
  config: DatasetConfig
): Promise<CreateResult> {
  // Validate inputs
  if (!file) {
    throw new CustomDatasetApiError('No file provided', 400);
  }

  if (!config.dataset_name?.trim()) {
    throw new CustomDatasetApiError('Dataset name is required', 400);
  }

  if (!config.target_column?.trim()) {
    throw new CustomDatasetApiError('Target column is required', 400);
  }

  if (!config.feature_columns || config.feature_columns.length === 0) {
    throw new CustomDatasetApiError('At least one feature column is required', 400);
  }

  if (!['classification', 'regression'].includes(config.task_type)) {
    throw new CustomDatasetApiError('Task type must be either "classification" or "regression"', 400);
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('config', JSON.stringify(config));

  const response = await makeApiRequest<CreateResult>('/create', {
    method: 'POST',
    body: formData,
  });

  if (!response.success) {
    throw new CustomDatasetApiError('Failed to create custom dataset', 500, response);
  }

  return response;
}

/**
 * Get a list of all custom datasets
 * @returns Promise with array of custom datasets
 */
export async function getCustomDatasets(): Promise<CustomDataset[]> {
  console.log('🔍 Fetching custom datasets...');
  
  const response = await makeApiRequest<{success: boolean, datasets: CustomDataset[]}>('/custom');
  
  console.log('📦 getCustomDatasets raw response:', response);
  
  // Extract the datasets array from the response
  const datasets = response.datasets || [];
  console.log('⭐ Extracted custom datasets:', datasets);
  
  return datasets;
}

/**
 * Get all available datasets (built-in and custom) for frontend selection
 * @returns Promise with categorized datasets
 */
export async function getAvailableDatasets(): Promise<AvailableDatasets> {
  console.log('🔍 Fetching available datasets...');
  
  const response = await makeApiRequest<{success: boolean, datasets: AvailableDatasets}>('/available');
  
  console.log('📦 getAvailableDatasets raw response:', response);
  
  // Extract the datasets object from the response
  const datasets = response.datasets || { built_in: [], custom: [], all_names: [] };
  console.log('⭐ Extracted available datasets:', datasets);
  
  return datasets;
}

/**
 * Delete a custom dataset (if endpoint exists)
 * @param datasetName - Name of the dataset to delete
 * @returns Promise with deletion result
 */
export async function deleteCustomDataset(datasetName: string): Promise<{ success: boolean; message: string }> {
  if (!datasetName?.trim()) {
    throw new CustomDatasetApiError('Dataset name is required', 400);
  }

  const response = await makeApiRequest<{ success: boolean; message: string }>(`/delete/${encodeURIComponent(datasetName)}`, {
    method: 'DELETE',
  });

  if (!response.success) {
    throw new CustomDatasetApiError('Failed to delete custom dataset', 500, response);
  }

  return response;
}

// Utility functions for working with dataset data

/**
 * Format file size for display
 * @param bytes - File size in bytes
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format date string for display
 * @param dateString - ISO date string
 * @returns Formatted date string
 */
export function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  } catch (error) {
    return dateString; // Return original if parsing fails
  }
}

/**
 * Get data type color for UI display
 * @param dataType - The data type string
 * @returns CSS color class or hex color
 */
export function getDataTypeColor(dataType: string): string {
  const colorMap: Record<string, string> = {
    integer: '#3B82F6', // blue
    float: '#10B981',   // green
    categorical: '#F59E0B', // amber
    text: '#8B5CF6',    // purple
    boolean: '#EF4444', // red
    other: '#6B7280',   // gray
  };
  
  return colorMap[dataType] || colorMap.other;
}

/**
 * Validate dataset configuration before submission
 * @param config - The dataset configuration to validate
 * @returns Array of validation error messages
 */
export function validateConfigLocally(config: Partial<DatasetConfig>): string[] {
  const errors: string[] = [];

  if (!config.dataset_name?.trim()) {
    errors.push('Dataset name is required');
  } else if (config.dataset_name.length > 50) {
    errors.push('Dataset name must be 50 characters or less');
  } else if (!/^[a-zA-Z0-9_-]+$/.test(config.dataset_name)) {
    errors.push('Dataset name can only contain letters, numbers, underscores, and hyphens');
  }

  if (!config.target_column?.trim()) {
    errors.push('Target column is required');
  }

  if (!config.feature_columns || config.feature_columns.length === 0) {
    errors.push('At least one feature column must be selected');
  }

  if (!config.task_type) {
    errors.push('Task type is required');
  } else if (!['classification', 'regression'].includes(config.task_type)) {
    errors.push('Task type must be either "classification" or "regression"');
  }

  // Check for duplicate columns
  if (config.target_column && config.feature_columns) {
    if (config.feature_columns.includes(config.target_column)) {
      errors.push('Target column cannot be included in feature columns');
    }
  }

  return errors;
}

// Export all functions and types
export default {
  previewDataset,
  validateDatasetConfig,
  createCustomDataset,
  getCustomDatasets,
  getAvailableDatasets,
  deleteCustomDataset,
  formatFileSize,
  formatDate,
  getDataTypeColor,
  validateConfigLocally,
  CustomDatasetApiError,
}; 