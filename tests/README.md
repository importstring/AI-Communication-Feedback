# AI-Presentation-Feedback Tests

This directory contains tests for the AI-Presentation-Feedback application.

## Test Structure

- `tests/`: Root test directory
  - `test_factors/`: Tests for modules in the `src/factors` directory
    - `test_body_language.py`: Tests for body language analysis
    - `test_emotions.py`: Tests for emotion analysis
    - `test_volume.py`: Tests for volume analysis
    - `test_helper.py`: Tests for helper functions
  - `test_recording_tools.py`: Tests for recording functionality
  - `test_simple.py`: Simple tests that demonstrate the test setup
  - `run_tests.py`: Test runner script for all tests

## Running Tests

To run all tests:

```
python -m unittest discover
```

To run a specific test file:

```
python -m unittest tests/test_simple.py
```

To run a specific test case:

```
python -m unittest tests.test_simple.TestSimple
```

## Dependencies

The tests require the same dependencies as the main application. Some tests may require additional mocking if dependencies are not available.

## Test Coverage

The tests cover:

1. **Recording Tools**

   - Video and audio recording functionality
   - Path management
   - Resource handling

2. **Body Language Analysis**

   - Joint mapping
   - Pose estimation
   - Data conversion

3. **Emotion Analysis**

   - Basic emotion classification
   - Multimodal emotion analysis
   - Frame, audio, and text processing

4. **Volume Analysis**

   - Audio processing
   - Volume statistics computation
   - Visualization

5. **Helper Functions**
   - Data storage and retrieval
   - Path management
