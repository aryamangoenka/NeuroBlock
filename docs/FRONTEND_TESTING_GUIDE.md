# Frontend Testing Guide for Session-Based Datasets

## Prerequisites

1. **Backend Running**: Ensure the backend server is running on `http://localhost:5000`
2. **Frontend Running**: Start your frontend application

## Browser-Based Testing

### Test 1: Basic Session Isolation

1. **Open Browser Tab 1**:

   - Navigate to your application
   - Go to the dataset upload/management section
   - Upload a custom dataset (name it "Dataset_Tab1")
   - Verify it appears in your dataset list

2. **Open Browser Tab 2** (same browser):

   - Navigate to your application in a new tab
   - Go to the dataset upload/management section
   - Check if "Dataset_Tab1" is visible
   - **Expected**: Should NOT see "Dataset_Tab1" (new session)

3. **Upload in Tab 2**:

   - Upload a different dataset (name it "Dataset_Tab2")
   - Verify only "Dataset_Tab2" appears in Tab 2

4. **Verify Isolation**:
   - Switch back to Tab 1
   - Refresh the page
   - **Expected**: Should still only see "Dataset_Tab1"

### Test 2: Incognito/Private Mode Testing

1. **Regular Browser Window**:

   - Upload a dataset named "Regular_Dataset"
   - Note the dataset list

2. **Open Incognito/Private Window**:

   - Navigate to the same application
   - Check dataset list
   - **Expected**: Should be empty (completely isolated session)

3. **Upload in Incognito**:
   - Upload "Incognito_Dataset"
   - Verify it's isolated from regular window

### Test 3: Session Persistence

1. **Upload Dataset**:

   - Upload a dataset in a browser tab
   - Note the dataset name

2. **Refresh Page**:

   - Refresh the browser tab
   - **Expected**: Dataset should still be there (same session)

3. **Navigate Away and Back**:
   - Navigate to a different page in the app
   - Come back to dataset section
   - **Expected**: Dataset should persist

### Test 4: Multiple Users Simulation

1. **User 1 (Chrome)**:

   - Open application in Chrome
   - Upload "Chrome_Dataset"

2. **User 2 (Firefox)**:

   - Open application in Firefox
   - Upload "Firefox_Dataset"

3. **User 3 (Safari)**:

   - Open application in Safari
   - Upload "Safari_Dataset"

4. **Verification**:
   - Each browser should only see its own dataset
   - No cross-contamination between browsers

## Developer Tools Testing

### Test 5: Session Cookie Inspection

1. **Open Developer Tools** (F12)
2. **Go to Application/Storage Tab**
3. **Check Cookies**:
   - Look for session cookie
   - Note the session ID format (should be UUID)
   - Verify it's HttpOnly and Secure

### Test 6: Network Request Inspection

1. **Open Network Tab** in Developer Tools
2. **Upload a Dataset**:

   - Watch the network requests
   - Verify `credentials: 'include'` is working
   - Check that session cookies are sent with requests

3. **API Response Verification**:
   - Check `/api/datasets/available` response
   - Verify only session-specific datasets are returned
   - Confirm built-in datasets are always available

## API Testing with Browser Console

### Test 7: Direct API Calls

Open browser console and run:

```javascript
// Test 1: Check available datasets
fetch("/api/datasets/available", {
  credentials: "include",
})
  .then((r) => r.json())
  .then((data) => console.log("Available datasets:", data));

// Test 2: Check custom datasets
fetch("/api/datasets/custom", {
  credentials: "include",
})
  .then((r) => r.json())
  .then((data) => console.log("Custom datasets:", data));

// Test 3: Create a test dataset
const formData = new FormData();
const csvContent = "feature1,feature2,target\n1,2,A\n3,4,B\n5,6,A";
const blob = new Blob([csvContent], { type: "text/csv" });
formData.append("file", blob, "test.csv");
formData.append(
  "config",
  JSON.stringify({
    dataset_name: "console_test",
    target_column: "target",
    feature_columns: ["feature1", "feature2"],
    task_type: "classification",
  })
);

fetch("/api/datasets/create", {
  method: "POST",
  body: formData,
  credentials: "include",
})
  .then((r) => r.json())
  .then((data) => console.log("Dataset created:", data));
```

## Error Scenarios Testing

### Test 8: Session Timeout/Cleanup

1. **Create Dataset**: Upload a dataset
2. **Wait**: Wait for session cleanup (24 hours by default)
3. **Or Simulate**: Manually delete session directory
4. **Refresh**: Refresh the application
5. **Expected**: Dataset should be gone, new session created

### Test 9: CORS Testing

1. **Different Origin**: Try accessing from different domain
2. **Expected**: Should handle CORS properly with credentials

## Automated Frontend Testing

### Test 10: Selenium/Playwright Script

```javascript
// Example Playwright test
const { test, expect } = require("@playwright/test");

test("session isolation test", async ({ browser }) => {
  // Create two browser contexts (simulate different users)
  const context1 = await browser.newContext();
  const context2 = await browser.newContext();

  const page1 = await context1.newPage();
  const page2 = await context2.newPage();

  // Navigate both to the app
  await page1.goto("http://localhost:3000");
  await page2.goto("http://localhost:3000");

  // Upload dataset in page1
  await page1.locator('[data-testid="upload-dataset"]').click();
  // ... upload logic

  // Check that page2 doesn't see the dataset
  await page2.locator('[data-testid="dataset-list"]').waitFor();
  const datasets = await page2.locator('[data-testid="dataset-item"]').count();
  expect(datasets).toBe(0);

  await context1.close();
  await context2.close();
});
```

## Expected Results Summary

✅ **What Should Work**:

- Each browser tab/window gets isolated datasets
- Session persistence within same tab
- Built-in datasets available to all sessions
- Proper session cookie handling
- Clean dataset lists for new sessions

❌ **What Should NOT Happen**:

- Datasets appearing in wrong sessions
- Cross-contamination between users
- Session data leaking between tabs
- Missing session cookies in requests
- Global dataset pollution

## Troubleshooting

### Common Issues:

1. **Datasets Appearing Everywhere**:

   - Check if session middleware is working
   - Verify CORS credentials configuration
   - Check frontend API calls include credentials

2. **Session Not Persisting**:

   - Check cookie settings (HttpOnly, Secure, SameSite)
   - Verify session secret is configured
   - Check browser cookie settings

3. **CORS Errors**:

   - Verify `supports_credentials=True` in backend
   - Check `credentials: 'include'` in frontend
   - Verify origin configuration

4. **Empty Dataset Lists**:
   - Check if backend server is running
   - Verify API endpoints are accessible
   - Check browser console for errors
