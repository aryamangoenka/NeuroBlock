/**
 * Test file demonstrating usage of the Custom Dataset API service
 * This file shows examples of how to use all the API functions
 */

import {
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
  type DatasetPreview,
  type ValidationResult,
  type CreateResult,
  type CustomDataset,
  type DatasetConfig,
  type AvailableDatasets,
} from './customDatasetApi';

// Example usage functions that can be called from React components

/**
 * Example: Preview a dataset file
 */
export async function examplePreviewDataset(file: File): Promise<void> {
  try {
    console.log('📊 Previewing dataset:', file.name);
    console.log('📏 File size:', formatFileSize(file.size));
    
    const preview: DatasetPreview = await previewDataset(file);
    
    console.log('✅ Dataset Preview Results:');
    console.log('  📁 Filename:', preview.filename);
    console.log('  📐 Shape:', preview.shape);
    console.log('  📋 Columns:', preview.columns);
    console.log('  🏷️  Data Types:', preview.data_types);
    console.log('  ❓ Missing Values:', preview.missing_values);
    console.log('  📊 Sample Data (first 5 rows):', preview.rows);
    
    if (preview.row_limit_exceeded) {
      console.log('⚠️  Row limit exceeded - showing first 100,000 rows only');
    }
    
    // Display numeric statistics
    if (Object.keys(preview.numeric_statistics).length > 0) {
      console.log('  📈 Numeric Statistics:');
      Object.entries(preview.numeric_statistics).forEach(([col, stats]) => {
        console.log(`    ${col}:`, stats);
      });
    }
    
    // Display categorical information
    if (Object.keys(preview.categorical_info).length > 0) {
      console.log('  🏷️  Categorical Info:');
      Object.entries(preview.categorical_info).forEach(([col, info]) => {
        console.log(`    ${col}: ${info.unique_count} unique values`);
        console.log(`      Top values:`, info.top_values);
      });
    }
    
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error('❌ API Error:', error.message);
      console.error('   Status:', error.status);
    } else {
      console.error('❌ Unexpected error:', error);
    }
  }
}

/**
 * Example: Validate dataset configuration
 */
export async function exampleValidateConfig(
  config: DatasetConfig,
  preview: DatasetPreview
): Promise<ValidationResult | null> {
  try {
    console.log('🔍 Validating dataset configuration...');
    
    // First, validate locally for immediate feedback
    const localErrors = validateConfigLocally(config);
    if (localErrors.length > 0) {
      console.log('❌ Local validation errors:', localErrors);
      return null;
    }
    
    console.log('✅ Local validation passed');
    
    // Then validate with the server
    const validation: ValidationResult = await validateDatasetConfig(config, preview);
    
    console.log('🔍 Server Validation Results:');
    console.log('  ✅ Valid:', validation.valid);
    
    if (validation.warnings.length > 0) {
      console.log('  ⚠️  Warnings:', validation.warnings);
    }
    
    if (validation.errors.length > 0) {
      console.log('  ❌ Errors:', validation.errors);
    }
    
    if (validation.recommendations.length > 0) {
      console.log('  💡 Recommendations:', validation.recommendations);
    }
    
    return validation;
    
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error('❌ Validation API Error:', error.message);
    } else {
      console.error('❌ Unexpected validation error:', error);
    }
    return null;
  }
}

/**
 * Example: Create a custom dataset
 */
export async function exampleCreateDataset(
  file: File,
  config: DatasetConfig
): Promise<CreateResult | null> {
  try {
    console.log('🚀 Creating custom dataset:', config.dataset_name);
    
    const result: CreateResult = await createCustomDataset(file, config);
    
    console.log('✅ Dataset Created Successfully!');
    console.log('  📊 Name:', result.dataset.name);
    console.log('  📐 Shape:', result.dataset.shape);
    console.log('  🎯 Task Type:', result.dataset.task_type);
    console.log('  🔢 Feature Count:', result.dataset.feature_count);
    
    if (result.dataset.class_labels) {
      console.log('  🏷️  Class Labels:', result.dataset.class_labels);
    }
    
    console.log('  💬 Message:', result.message);
    
    return result;
    
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error('❌ Dataset Creation Error:', error.message);
      if (error.status === 400) {
        console.error('   This is likely a validation error. Check your configuration.');
      }
    } else {
      console.error('❌ Unexpected creation error:', error);
    }
    return null;
  }
}

/**
 * Example: Get all custom datasets
 */
export async function exampleGetCustomDatasets(): Promise<CustomDataset[]> {
  try {
    console.log('📋 Fetching custom datasets...');
    
    const datasets: CustomDataset[] = await getCustomDatasets();
    
    console.log(`✅ Found ${datasets.length} custom datasets:`);
    
    datasets.forEach((dataset, index) => {
      console.log(`  ${index + 1}. ${dataset.name}`);
      console.log(`     🎯 Task: ${dataset.task_type}`);
      console.log(`     📐 Shape: ${dataset.shape}`);
      console.log(`     📅 Created: ${formatDate(dataset.created_at)}`);
      console.log(`     📁 Original File: ${dataset.original_filename}`);
      console.log(`     🎯 Target: ${dataset.target_column}`);
      
      if (dataset.class_labels) {
        console.log(`     🏷️  Classes: ${dataset.class_labels.join(', ')}`);
      }
    });
    
    return datasets;
    
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error('❌ Error fetching custom datasets:', error.message);
    } else {
      console.error('❌ Unexpected error:', error);
    }
    return [];
  }
}

/**
 * Example: Get all available datasets (built-in + custom)
 */
export async function exampleGetAvailableDatasets(): Promise<AvailableDatasets | null> {
  try {
    console.log('📋 Fetching all available datasets...');
    
    const datasets: AvailableDatasets = await getAvailableDatasets();
    
    console.log('✅ Available Datasets:');
    console.log(`  🔧 Built-in datasets: ${datasets.built_in.length}`);
    datasets.built_in.forEach((dataset, index) => {
      console.log(`    ${index + 1}. ${dataset.name} - ${dataset.description}`);
    });
    
    console.log(`  📊 Custom datasets: ${datasets.custom.length}`);
    datasets.custom.forEach((dataset, index) => {
      console.log(`    ${index + 1}. ${dataset.name} (${dataset.task_type})`);
    });
    
    console.log(`  📝 All dataset names: ${datasets.all_names.join(', ')}`);
    
    return datasets;
    
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error('❌ Error fetching available datasets:', error.message);
    } else {
      console.error('❌ Unexpected error:', error);
    }
    return null;
  }
}

/**
 * Example: Delete a custom dataset
 */
export async function exampleDeleteDataset(datasetName: string): Promise<boolean> {
  try {
    console.log('🗑️  Deleting dataset:', datasetName);
    
    const result = await deleteCustomDataset(datasetName);
    
    console.log('✅ Dataset deleted successfully!');
    console.log('  💬 Message:', result.message);
    
    return true;
    
  } catch (error) {
    if (error instanceof CustomDatasetApiError) {
      console.error('❌ Error deleting dataset:', error.message);
      if (error.status === 404) {
        console.error('   Dataset not found');
      }
    } else {
      console.error('❌ Unexpected deletion error:', error);
    }
    return false;
  }
}

/**
 * Example: Complete workflow - from file upload to dataset creation
 */
export async function exampleCompleteWorkflow(file: File): Promise<void> {
  try {
    console.log('🔄 Starting complete dataset creation workflow...');
    
    // Step 1: Preview the dataset
    console.log('\n📊 Step 1: Previewing dataset...');
    const preview = await previewDataset(file);
    
    // Step 2: Create configuration based on preview
    console.log('\n⚙️  Step 2: Creating configuration...');
    const config: DatasetConfig = {
      dataset_name: `dataset_${Date.now()}`,
      target_column: preview.columns[preview.columns.length - 1], // Use last column as target
      feature_columns: preview.columns.slice(0, -1), // Use all but last column as features
      task_type: 'classification', // Assume classification for this example
      missing_value_strategy: 'drop'
    };
    
    console.log('  📝 Configuration:', config);
    
    // Step 3: Validate configuration
    console.log('\n🔍 Step 3: Validating configuration...');
    const validation = await validateDatasetConfig(config, preview);
    
    if (!validation.valid) {
      console.log('❌ Configuration is not valid. Errors:', validation.errors);
      return;
    }
    
    if (validation.warnings.length > 0) {
      console.log('⚠️  Warnings:', validation.warnings);
    }
    
    // Step 4: Create the dataset
    console.log('\n🚀 Step 4: Creating dataset...');
    const result = await createCustomDataset(file, config);
    
    if (result) {
      console.log('✅ Workflow completed successfully!');
      console.log('  📊 Dataset Name:', result.dataset.name);
      console.log('  📐 Final Shape:', result.dataset.shape);
    }
    
  } catch (error) {
    console.error('❌ Workflow failed:', error);
  }
}

/**
 * Example: Utility functions demonstration
 */
export function exampleUtilityFunctions(): void {
  console.log('🛠️  Utility Functions Demo:');
  
  // File size formatting
  console.log('📏 File Size Examples:');
  console.log('  1024 bytes =', formatFileSize(1024));
  console.log('  1048576 bytes =', formatFileSize(1048576));
  console.log('  52428800 bytes =', formatFileSize(52428800));
  
  // Date formatting
  console.log('📅 Date Formatting Examples:');
  const now = new Date().toISOString();
  console.log('  ISO string:', now);
  console.log('  Formatted:', formatDate(now));
  
  // Data type colors
  console.log('🎨 Data Type Colors:');
  const dataTypes = ['integer', 'float', 'categorical', 'text', 'boolean', 'other'];
  dataTypes.forEach(type => {
    console.log(`  ${type}: ${getDataTypeColor(type)}`);
  });
  
  // Local validation
  console.log('✅ Local Validation Examples:');
  
  const validConfig: Partial<DatasetConfig> = {
    dataset_name: 'test_dataset',
    target_column: 'target',
    feature_columns: ['feature1', 'feature2'],
    task_type: 'classification'
  };
  
  const invalidConfig: Partial<DatasetConfig> = {
    dataset_name: '', // Invalid: empty name
    target_column: 'target',
    feature_columns: ['target'], // Invalid: target in features
    task_type: 'classification'
  };
  
  console.log('  Valid config errors:', validateConfigLocally(validConfig));
  console.log('  Invalid config errors:', validateConfigLocally(invalidConfig));
}

// Example React component usage patterns

/**
 * Example: How to use in a React component with state management
 */
export const exampleReactUsage = `
import React, { useState, useCallback } from 'react';
import {
  previewDataset,
  createCustomDataset,
  getCustomDatasets,
  CustomDatasetApiError,
  type DatasetPreview,
  type DatasetConfig,
  type CustomDataset
} from '../utils/customDatasetApi';

export function CustomDatasetManager() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);

  const handleFileChange = useCallback(async (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setLoading(true);

    try {
      const previewData = await previewDataset(selectedFile);
      setPreview(previewData);
    } catch (err) {
      if (err instanceof CustomDatasetApiError) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleCreateDataset = useCallback(async (config: DatasetConfig) => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      await createCustomDataset(file, config);
      // Refresh the datasets list
      const updatedDatasets = await getCustomDatasets();
      setDatasets(updatedDatasets);
    } catch (err) {
      if (err instanceof CustomDatasetApiError) {
        setError(err.message);
      } else {
        setError('Failed to create dataset');
      }
    } finally {
      setLoading(false);
    }
  }, [file]);

  // Component JSX would go here...
  return null; // Placeholder
}
`;

console.log('📖 Custom Dataset API Test Examples Ready!');
console.log('💡 Import and use the example functions in your React components.');
console.log('🚀 Example React usage pattern:');
console.log(exampleReactUsage); 