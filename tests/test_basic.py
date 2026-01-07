"""Basic tests for mgnify-methods package."""

import mgnify_methods
import numpy as np
import json
from mgnify_methods.utils.plot import _make_json_serializable


def test_hello() -> None:
    """Test the hello function."""
    result = mgnify_methods.hello()
    assert isinstance(result, str)
    assert "Hello from mgnify-methods!" == result


def test_package_has_version() -> None:
    """Test that the package has a version attribute."""
    assert hasattr(mgnify_methods, "__version__")

    
def test_json_serialization():
    """Test function to verify JSON serialization works with numpy types."""
    test_data = {
        'numpy_int': np.int64(42),
        'numpy_float': np.float64(3.14),
        'numpy_array': np.array([1, 2, 3]),
        'nested_dict': {
            'numpy_val': np.int32(100),
            'regular_val': 'test'
        },
        'list_with_numpy': [np.float32(1.5), 'string', np.int16(5)]
    }
    
    serialized = _make_json_serializable(test_data)
    
    # Test that it can be serialized to JSON without raising TypeError
    json_string = json.dumps(serialized, indent=2)
    assert isinstance(json_string, str)
    
    # Test that serialized data can be deserialized
    deserialized = json.loads(json_string)
    assert isinstance(deserialized, dict)
    
    # Verify converted values are correct Python types
    assert deserialized['numpy_int'] == 42
    assert deserialized['numpy_float'] == 3.14
    assert deserialized['numpy_array'] == [1, 2, 3]
    assert deserialized['nested_dict']['numpy_val'] == 100
    assert deserialized['nested_dict']['regular_val'] == 'test'
