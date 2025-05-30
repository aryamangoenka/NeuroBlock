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
    "Iris"

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

# Test 8: Cross-session isolation check
run_test "Session 1 Does NOT See Session 2 Dataset" \
    "curl -s -b session1.txt http://localhost:5000/api/datasets/custom | grep -v test2 | grep test1" \
    "test1"

run_test "Session 2 Does NOT See Session 1 Dataset" \
    "curl -s -b session2.txt http://localhost:5000/api/datasets/custom | grep -v test1 | grep test2" \
    "test2"

# Test 9: Directory Structure
run_test "Session Directories Created" \
    "ls datasets/sessions/ | wc -l | grep -E '[2-9]'" \
    "[2-9]"

# Test 10: Access Files Created
run_test "Access Tracking Files Created" \
    "find datasets/sessions/ -name '.last_access' | wc -l | grep -E '[2-9]'" \
    "[2-9]"

# Test 11: Dataset Deletion
run_test "Delete Dataset from Session 1" \
    "curl -s -b session1.txt -X DELETE http://localhost:5000/api/datasets/delete/test1" \
    "success"

run_test "Verify Session 1 Dataset Deleted" \
    "curl -s -b session1.txt http://localhost:5000/api/datasets/custom" \
    '"datasets":\[\]'

run_test "Verify Session 2 Dataset Still Exists" \
    "curl -s -b session2.txt http://localhost:5000/api/datasets/custom" \
    "test2"

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