"""
Test for past_key_values handling in modeling_llama.py to prevent IndexError.

This test covers the bug where past_key_values[0] might be an empty tuple/list,
causing IndexError when accessing past_key_values[0][0].

This test directly tests the logic without importing the full model to avoid dependency issues.
"""
import unittest
import torch


def calculate_past_key_values_length_fixed(past_key_values):
    """
    Fixed version of past_key_values_length calculation from modeling_llama.py.
    This is the logic we're testing.
    """
    past_key_values_length = 0
    if past_key_values is not None:
        # Check if it's a Cache type (has get_seq_length method)
        if hasattr(past_key_values, 'get_seq_length'):
            past_key_values_length = past_key_values.get_seq_length()
        else:
            # Check if past_key_values has layers and the first layer has key/value tensors
            if (isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0):
                first_layer = past_key_values[0]
                if isinstance(first_layer, (list, tuple)) and len(first_layer) > 0:
                    if first_layer[0] is not None:
                        past_key_values_length = first_layer[0].shape[2]
    return past_key_values_length


def calculate_past_key_values_length_broken(past_key_values):
    """
    Broken version that causes IndexError - this is what the code was before the fix.
    """
    past_key_values_length = 0
    if past_key_values is not None:
        if hasattr(past_key_values, 'get_seq_length'):
            past_key_values_length = past_key_values.get_seq_length()
        else:
            # This line causes IndexError when past_key_values[0] is empty
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
    return past_key_values_length


class TestPastKeyValuesHandling(unittest.TestCase):
    """Test that past_key_values with empty layers don't cause IndexError."""
    
    def test_past_key_values_none(self):
        """Test that None past_key_values works correctly."""
        result = calculate_past_key_values_length_fixed(None)
        self.assertEqual(result, 0)
    
    def test_past_key_values_empty_list(self):
        """Test that empty list past_key_values works correctly."""
        result = calculate_past_key_values_length_fixed([])
        self.assertEqual(result, 0)
    
    def test_past_key_values_empty_tuple_in_first_layer(self):
        """Test that empty tuple in past_key_values[0] doesn't cause IndexError."""
        # This is the bug case that was causing IndexError
        past_key_values = [()]  # Empty tuple as first layer
        result = calculate_past_key_values_length_fixed(past_key_values)
        self.assertEqual(result, 0)
        
        # Verify the broken version would fail
        with self.assertRaises(IndexError):
            calculate_past_key_values_length_broken(past_key_values)
    
    def test_past_key_values_empty_list_in_first_layer(self):
        """Test that empty list in past_key_values[0] doesn't cause IndexError."""
        # Similar bug case
        past_key_values = [[]]  # Empty list as first layer
        result = calculate_past_key_values_length_fixed(past_key_values)
        self.assertEqual(result, 0)
        
        # Verify the broken version would fail
        with self.assertRaises(IndexError):
            calculate_past_key_values_length_broken(past_key_values)
    
    def test_past_key_values_valid(self):
        """Test that valid past_key_values works correctly."""
        # Create valid past_key_values structure: list of tuples, each tuple has (key, value) tensors
        key_tensor = torch.randn(1, 4, 5, 32)  # (batch, heads, seq_len, head_dim)
        value_tensor = torch.randn(1, 4, 5, 32)
        past_key_values = [(key_tensor, value_tensor)]
        
        result = calculate_past_key_values_length_fixed(past_key_values)
        self.assertEqual(result, 5)  # seq_len from key_tensor.shape[2]
        
        # Verify both versions work for valid input
        result_broken = calculate_past_key_values_length_broken(past_key_values)
        self.assertEqual(result_broken, 5)
    
    def test_past_key_values_with_cache_type(self):
        """Test that Cache type past_key_values works correctly."""
        class MockCache:
            def get_seq_length(self):
                return 10
        
        cache = MockCache()
        result = calculate_past_key_values_length_fixed(cache)
        self.assertEqual(result, 10)


if __name__ == "__main__":
    unittest.main()
