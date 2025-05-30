# Testing Session-Based Dataset Implementation

## Prerequisites

1. **Activate Virtual Environment**

   ```bash
   cd DND-Neural-Network
   source venv/bin/activate
   ```

2. **Start the Backend Server**
   ```bash
   python -m backend.main
   ```
   The server should start on `http://localhost:5000`

## Test Categories

### 1. Basic Functionality Tests

#### Test 1: Server Health Check

```bash
curl -s http://localhost:5000/health
```

**Expected Output:**

```json
{ "status": "healthy" }
```

#### Test 2: Session Creation and Isolation

```bash
# Create first session
curl -s -c session1.txt http://localhost:5000/api/datasets/available

# Create second session
curl -s -c session2.txt http://localhost:5000/api/datasets/available

# Verify different session IDs were created
echo "Session 1 cookies:"
cat session1.txt
echo "Session 2 cookies:"
cat session2.txt
```

### 2. Dataset Isolation Tests

#### Test 3: Empty Custom Datasets for New Sessions

```bash
# Test session 1
curl -s -b session1.txt http://localhost:5000/api/datasets/custom
# Expected: {"datasets":[],"success":true}

# Test session 2
curl -s -b session2.txt http://localhost:5000/api/datasets/custom
# Expected: {"datasets":[],"success":true}
```

#### Test 4: Built-in Datasets Available to All Sessions

```bash
# Test session 1
curl -s -b session1.txt http://localhost:5000/api/datasets/available | jq '.datasets.built_in[].name'

# Test session 2
curl -s -b session2.txt http://localhost:5000/api/datasets/available | jq '.datasets.built_in[].name'
# Both should return the same built-in datasets
```

### 3. File Upload and Dataset Creation Tests

#### Test 5: Create Sample CSV File

```bash
cat > test_dataset.csv << 'EOF'
feature1,feature2,feature3,target
1.0,2.0,3.0,A
2.0,3.0,4.0,B
3.0,4.0,5.0,A
4.0,5.0,6.0,B
5.0,6.0,7.0,A
EOF
```

#### Test 6: Upload Dataset to Session 1

```bash
curl -s -b session1.txt -X POST \
  -F "file=@test_dataset.csv" \
  -F 'config={"dataset_name":"test_session1","target_column":"target","feature_columns":["feature1","feature2","feature3"],"task_type":"classification"}' \
  http://localhost:5000/api/datasets/create
```

#### Test 7: Upload Different Dataset to Session 2

```bash
# Create different dataset for session 2
cat > test_dataset2.csv << 'EOF'
x,y,z,label
10,20,30,1
20,30,40,0
30,40,50,1
40,50,60,0
EOF

curl -s -b session2.txt -X POST \
  -F "file=@test_dataset2.csv" \
  -F 'config={"dataset_name":"test_session2","target_column":"label","feature_columns":["x","y","z"],"task_type":"classification"}' \
  http://localhost:5000/api/datasets/create
```

### 4. Session Isolation Verification

#### Test 8: Verify Dataset Isolation

```bash
echo "Session 1 datasets:"
curl -s -b session1.txt http://localhost:5000/api/datasets/custom | jq '.datasets[].name'

echo "Session 2 datasets:"
curl -s -b session2.txt http://localhost:5000/api/datasets/custom | jq '.datasets[].name'

# Should show different datasets for each session
```

#### Test 9: Verify Available Datasets Include Session-Specific Ones

```bash
echo "Session 1 available datasets:"
curl -s -b session1.txt http://localhost:5000/api/datasets/available | jq '.datasets.all_names'

echo "Session 2 available datasets:"
curl -s -b session2.txt http://localhost:5000/api/datasets/available | jq '.datasets.all_names'
```

### 5. Directory Structure Tests

#### Test 10: Check Session Directory Creation

```bash
echo "Session directories created:"
ls -la datasets/sessions/

echo "Session 1 directory contents:"
find datasets/sessions/ -name "*test_session1*" -exec ls -la {} \;

echo "Session 2 directory contents:"
find datasets/sessions/ -name "*test_session2*" -exec ls -la {} \;
```

#### Test 11: Verify Access Tracking Files

```bash
echo "Access tracking files:"
find datasets/sessions/ -name ".last_access" -exec ls -la {} \;

echo "Recent access times:"
find datasets/sessions/ -name ".last_access" -exec cat {} \;
```

### 6. Dataset Deletion Tests

#### Test 12: Delete Dataset from Session 1

```bash
curl -s -b session1.txt -X DELETE \
  http://localhost:5000/api/datasets/delete/test_session1

# Verify deletion
echo "Session 1 datasets after deletion:"
curl -s -b session1.txt http://localhost:5000/api/datasets/custom
```

#### Test 13: Verify Other Session Unaffected

```bash
echo "Session 2 datasets (should be unchanged):"
curl -s -b session2.txt http://localhost:5000/api/datasets/custom
```

### 7. Frontend Integration Tests

#### Test 14: Frontend API with Credentials

```bash
# Test with credentials (simulating frontend)
curl -s -c frontend_session.txt -b frontend_session.txt \
  -H "Content-Type: application/json" \
  http://localhost:5000/api/datasets/available
```

### 8. Error Handling Tests

#### Test 15: Invalid Dataset Name

```bash
curl -s -b session1.txt -X DELETE \
  http://localhost:5000/api/datasets/delete/nonexistent_dataset
# Should return 404 error
```

#### Test 16: Missing Session Context

```bash
# Test without session cookies
curl -s http://localhost:5000/api/datasets/custom
# Should still work but return empty results
```

## Automated Test Script

Create a comprehensive test script:

```bash
cat > test_session_datasets.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing Session-Based Dataset Implementation"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"

    echo -e "\n${YELLOW}Testing: $test_name${NC}"

    result=$(eval "$test_command" 2>&1)

    if echo "$result" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "Expected pattern: $expected_pattern"
        echo "Actual result: $result"
        ((TESTS_FAILED++))
    fi
}

# Test 1: Health Check
run_test "Server Health Check" \
    "curl -s http://localhost:5000/health" \
    "healthy"

# Test 2: Session Creation
run_test "Session 1 Creation" \
    "curl -s -c session1.txt http://localhost:5000/api/datasets/available" \
    "success"

run_test "Session 2 Creation" \
    "curl -s -c session2.txt http://localhost:5000/api/datasets/available" \
    "success"

# Test 3: Empty Custom Datasets
run_test "Empty Custom Datasets Session 1" \
    "curl -s -b session1.txt http://localhost:5000/api/datasets/custom" \
    '"datasets":\[\]'

run_test "Empty Custom Datasets Session 2" \
    "curl -s -b session2.txt http://localhost:5000/api/datasets/custom" \
    '"datasets":\[\]'

# Test 4: Built-in Datasets
run_test "Built-in Datasets Available" \
    "curl -s -b session1.txt http://localhost:5000/api/datasets/available" \
    "Iris.*MNIST"

# Create test datasets
echo -e "\n${YELLOW}Creating test datasets...${NC}"
cat > test_dataset1.csv << 'EOD'
feature1,feature2,target
1.0,2.0,A
2.0,3.0,B
3.0,4.0,A
EOD

cat > test_dataset2.csv << 'EOD'
x,y,label
10,20,1
20,30,0
30,40,1
EOD

# Test 5: Dataset Upload Session 1
run_test "Upload Dataset to Session 1" \
    "curl -s -b session1.txt -X POST -F 'file=@test_dataset1.csv' -F 'config={\"dataset_name\":\"test1\",\"target_column\":\"target\",\"feature_columns\":[\"feature1\",\"feature2\"],\"task_type\":\"classification\"}' http://localhost:5000/api/datasets/create" \
    "success"

# Test 6: Dataset Upload Session 2
run_test "Upload Dataset to Session 2" \
    "curl -s -b session2.txt -X POST -F 'file=@test_dataset2.csv' -F 'config={\"dataset_name\":\"test2\",\"target_column\":\"label\",\"feature_columns\":[\"x\",\"y\"],\"task_type\":\"classification\"}' http://localhost:5000/api/datasets/create" \
    "success"

# Test 7: Dataset Isolation
run_test "Session 1 Has Only Its Dataset" \
    "curl -s -b session1.txt http://localhost:5000/api/datasets/custom" \
    "test1"

run_test "Session 2 Has Only Its Dataset" \
    "curl -s -b session2.txt http://localhost:5000/api/datasets/custom" \
    "test2"

# Test 8: Directory Structure
run_test "Session Directories Created" \
    "ls datasets/sessions/ | wc -l" \
    "2"

# Test 9: Access Files Created
run_test "Access Tracking Files Created" \
    "find datasets/sessions/ -name '.last_access' | wc -l" \
    "2"

# Summary
echo -e "\n${YELLOW}Test Summary${NC}"
echo "============"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! Session-based datasets are working correctly.${NC}"
else
    echo -e "\n${RED}❌ Some tests failed. Please check the implementation.${NC}"
fi

# Cleanup
rm -f session1.txt session2.txt test_dataset1.csv test_dataset2.csv
EOF

chmod +x test_session_datasets.sh
```

## Manual Testing Steps

### Step 1: Start the Server

```bash
cd DND-Neural-Network
source venv/bin/activate
python -m backend.main
```

### Step 2: Run the Automated Test

```bash
# In a new terminal
cd DND-Neural-Network
./test_session_datasets.sh
```

### Step 3: Frontend Testing

1. Open the frontend application
2. Navigate to the dataset upload section
3. Upload a custom dataset
4. Open a new browser tab/incognito window
5. Verify the dataset is not visible in the new session

### Step 4: Cleanup Testing

Wait 24+ hours or manually trigger cleanup:

```bash
# Check current sessions
ls -la datasets/sessions/

# Manually test cleanup (modify timestamps)
find datasets/sessions/ -name ".last_access" -exec touch -t 202301010000 {} \;

# Restart server to trigger cleanup
```

## Expected Results

✅ **Successful Implementation Should Show:**

- Each session gets unique UUID directory
- Datasets are isolated between sessions
- Built-in datasets available to all sessions
- Automatic cleanup removes old sessions
- Frontend works with session cookies
- No cross-contamination between users

❌ **Common Issues to Watch For:**

- Session cookies not being sent/received
- Datasets appearing in wrong sessions
- Cleanup not working properly
- Directory permissions issues
- CORS configuration problems
