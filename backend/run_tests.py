#!/usr/bin/env python
"""
Test runner script for DND Neural Network tests.
Runs all tests and generates a coverage report.
"""

import os
import sys
import unittest
import coverage
import argparse

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests_with_coverage(verbose=False, html_report=False):
    """Run tests with coverage and generate report."""
    # Start coverage measurement
    cov = coverage.Coverage(
        source=["backend"],  # Only measure coverage in the backend directory
        omit=[
            "*/tests/*",
            "*/test_*",
            "*/run_tests.py",
            "*/__pycache__/*",
        ]
    )
    cov.start()

    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern="test_*.py"
    )

    verbosity = 2 if verbose else 1
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)

    # Stop coverage measurement and generate report
    cov.stop()
    cov.save()

    print("\nCoverage Summary:")
    cov.report()

    if html_report:
        html_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_html")
        cov.html_report(directory=html_dir)
        print(f"\nHTML coverage report generated in {html_dir}")

    return result


def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run DND Neural Network tests with coverage.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    args = parser.parse_args()

    print("Running DND Neural Network tests...")
    result = run_tests_with_coverage(verbose=args.verbose, html_report=args.html)

    # Return non-zero exit code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)

    print("\nAll tests passed!")


if __name__ == "__main__":
    main() 