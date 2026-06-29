# Trader Gemini - Tests

This directory contains validation and integrity tests for the trading system.

## Test Files

### test_validation.py
**Security & Configuration Validation**
- Scans for hardcoded API keys
- Verifies .env file configuration
- Validates config.py implementation
- Checks exception handling quality

**Run:** `python tests/test_validation.py`

### test_integrity.py
**System Integrity Checks**
- Validates Python syntax across all files
- Verifies core module imports
- Checks for circular dependencies

**Run:** `python tests/test_integrity.py`

## Running All Tests

```bash
# From project root
python tests/test_validation.py
python tests/test_integrity.py
```

## Expected Results

All tests should pass (✅) if:
1. `.env` file exists with valid API keys
2. No hardcoded secrets in source code
3. All Python files have valid syntax
4. No circular import dependencies

## CI/CD Integration

These tests can be integrated into your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run Security Tests
  run: python tests/test_validation.py

- name: Run Integrity Tests
  run: python tests/test_integrity.py
```
