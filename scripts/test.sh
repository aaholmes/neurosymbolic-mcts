#!/bin/bash
set -e

echo "=== Running Unit Tests ==="
cargo test --lib -- --test-threads=4
cargo test --test unit_tests -- --test-threads=4

echo "=== Running Integration Tests ==="
cargo test --test integration_tests -- --test-threads=1

echo "=== Running Property Tests ==="
cargo test --test property_tests -- --test-threads=2

echo "=== All tests passed ==="
