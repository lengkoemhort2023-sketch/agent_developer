"""
Python Implementation of TONL (Token-Optimized Notation Language)
Based on the specification from https://github.com/tonl-dev/tonl

This implementation provides basic TONL encoding/decoding functionality
for use in Python applications, particularly for LLM prompt optimization.
"""

import re
import json
from typing import Any, Dict, List, Union, Optional


class TONLEncoder:
    """TONL Encoder for serializing Python data structures to TONL format"""

    def __init__(self, smart: bool = True):
        self.smart = smart

    def encode(self, data: Any) -> str:
        """Encode Python data to TONL format"""
        if isinstance(data, dict):
            return self._encode_object(data)
        elif isinstance(data, list):
            return self._encode_array(data)
        elif isinstance(data, str):
            # Escape backslashes and quotes for JSON compatibility
            escaped = data.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(data, bool):
            return "true" if data else "false"
        elif isinstance(data, (int, float)):
            return str(data)
        elif data is None:
            return "null"
        else:
            # Fallback to string representation
            escaped = str(data).replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'

    def _encode_object(self, obj: Dict[str, Any]) -> str:
        """Encode a dictionary object"""
        if not obj:
            return "{}"

        pairs = []
        for key, value in obj.items():
            encoded_value = self.encode(value)
            pairs.append(f'"{key}":{encoded_value}')

        return "{" + ",".join(pairs) + "}"

    def _encode_array(self, arr: List[Any]) -> str:
        """Encode a list/array"""
        if not arr:
            return "[]"

        # Check if this is an array of similar objects (for tabular format)
        if self.smart and self._is_homogeneous_object_array(arr) and not self._has_newlines(arr):
            return self._encode_tabular(arr)

        # Regular array encoding
        encoded_items = [self.encode(item) for item in arr]
        return "[" + ",".join(encoded_items) + "]"

    def _is_homogeneous_object_array(self, arr: List[Any]) -> bool:
        """Check if array contains similar objects that can be tabularized"""
        if len(arr) < 2:
            return False

        if not all(isinstance(item, dict) for item in arr):
            return False

        # Check if all objects have the same keys
        first_keys = set(arr[0].keys())
        return all(set(obj.keys()) == first_keys for obj in arr)

    def _has_newlines(self, data: Any) -> bool:
        """Check if data contains newlines that would break tabular format"""
        if isinstance(data, str):
            return '\n' in data
        elif isinstance(data, dict):
            return any(self._has_newlines(v) for v in data.values())
        elif isinstance(data, list):
            return any(self._has_newlines(item) for item in data)
        return False

    def _encode_tabular(self, arr: List[Dict[str, Any]]) -> str:
        """Encode array of similar objects in tabular format"""
        if not arr:
            return "[]"

        # Get column names and infer types
        columns = list(arr[0].keys())
        headers = []

        for col in columns:
            col_type = self._infer_type(arr[0][col])
            headers.append(f'"{col}":{col_type}')

        # Create header row
        header_str = ",".join(headers)

        # Create data rows
        data_rows = []
        for row in arr:
            row_data = []
            for col in columns:
                value = row[col]
                if isinstance(value, str):
                    # Escape backslashes and quotes for JSON compatibility
                    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                    row_data.append(f'"{escaped}"')
                elif isinstance(value, bool):
                    row_data.append("true" if value else "false")
                elif value is None:
                    row_data.append("null")
                else:
                    row_data.append(str(value))
            data_rows.append(",".join(row_data))

        # Combine header and data
        return header_str + "\n" + "\n".join(data_rows)

    def _infer_type(self, value: Any) -> str:
        """Infer TONL type from Python value"""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            # Try to be specific about integer sizes
            if 0 <= value <= 255:
                return "u8"
            elif -128 <= value <= 127:
                return "i8"
            elif 0 <= value <= 65535:
                return "u16"
            elif -32768 <= value <= 32767:
                return "i16"
            elif 0 <= value <= 4294967295:
                return "u32"
            elif -2147483648 <= value <= 2147483647:
                return "i32"
            else:
                return "i64"
        elif isinstance(value, float):
            return "f64"
        elif isinstance(value, str):
            return "str"
        elif value is None:
            return "null"
        else:
            return "str"  # Default fallback


class TONLDecoder:
    """TONL Decoder for parsing TONL format back to Python data structures"""

    def __init__(self):
        self.pos = 0
        self.text = ""

    def decode(self, tonl_str: str) -> Any:
        """Decode TONL string to Python data structure"""
        self.text = tonl_str.strip()
        self.pos = 0

        # Check if this is tabular format (contains newlines and type hints)
        if '\n' in self.text and ':' in self.text and self._looks_like_tabular():
            return self._decode_tabular()

        # Regular TONL parsing
        return self._parse_value()

    def _looks_like_tabular(self) -> bool:
        """Check if the text looks like tabular format"""
        lines = self.text.split('\n')
        if len(lines) < 2:
            return False

        # Check if first line contains type hints (col:type format)
        first_line = lines[0]
        return ':' in first_line and ',' in first_line

    def _decode_tabular(self) -> List[Dict[str, Any]]:
        """Decode tabular format back to list of dictionaries"""
        lines = self.text.split('\n')
        if len(lines) < 2:
            return []

        # Parse header
        header_line = lines[0]
        columns = []
        types = {}

        for col_spec in header_line.split(','):
            if ':' in col_spec:
                parts = col_spec.split(':', 1)
                col_name = parts[0].strip()
                if col_name.startswith('"') and col_name.endswith('"'):
                    col_name = col_name[1:-1]
                col_type = parts[1].strip()
                columns.append(col_name)
                types[col_name] = col_type
            else:
                col_name = col_spec.strip()
                if col_name.startswith('"') and col_name.endswith('"'):
                    col_name = col_name[1:-1]
                columns.append(col_name)
                types[col_name] = "str"

        # Parse data rows
        result = []
        for line in lines[1:]:
            if not line.strip():
                continue

            values = self._parse_csv_line(line)
            if len(values) != len(columns):
                continue  # Skip malformed rows

            row_dict = {}
            for i, col in enumerate(columns):
                value = values[i]
                col_type = types.get(col, "str")
                row_dict[col] = self._convert_value(value, col_type)

            result.append(row_dict)

        return result

    def _parse_csv_line(self, line: str) -> List[str]:
        """Parse a CSV line handling quoted strings"""
        values = []
        current = ""
        in_quotes = False

        i = 0
        while i < len(line):
            char = line[i]

            if char == '"':
                if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                    # Escaped quote
                    current += '"'
                    i += 1
                else:
                    # Toggle quote state
                    in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                # End of field
                values.append(current)
                current = ""
            else:
                current += char

            i += 1

        # Add the last field
        values.append(current)
        return values

    def _convert_value(self, value: str, type_hint: str) -> Any:
        """Convert string value to appropriate Python type based on type hint"""
        value = value.strip()

        if value == "null":
            return None
        elif value == "true":
            return True
        elif value == "false":
            return False
        elif type_hint in ["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"]:
            try:
                return int(value)
            except ValueError:
                return 0
        elif type_hint in ["f32", "f64"]:
            try:
                return float(value)
            except ValueError:
                return 0.0
        elif type_hint == "str":
            # Remove surrounding quotes if present
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1].replace('\\"', '"')
            return value
        else:
            # Default string handling
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1].replace('\\"', '"')
            return value

    def _parse_value(self) -> Any:
        """Parse a single TONL value"""
        self._skip_whitespace()

        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")

        char = self.text[self.pos]

        if char == '{':
            return self._parse_object()
        elif char == '[':
            return self._parse_array()
        elif char == '"':
            return self._parse_string()
        elif char.isdigit() or char == '-':
            return self._parse_number()
        elif self.text.startswith("true", self.pos):
            self.pos += 4
            return True
        elif self.text.startswith("false", self.pos):
            self.pos += 5
            return False
        elif self.text.startswith("null", self.pos):
            self.pos += 4
            return None
        else:
            raise ValueError(f"Unexpected character: {char}")

    def _parse_object(self) -> Dict[str, Any]:
        """Parse a TONL object"""
        self.pos += 1  # Skip '{'
        obj = {}

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.text[self.pos] == '}':
                self.pos += 1
                break

            # Parse key
            key = self._parse_string_or_identifier()
            self._skip_whitespace()

            if self.text[self.pos] != ':':
                raise ValueError("Expected ':' in object")
            self.pos += 1

            # Parse value
            value = self._parse_value()
            obj[key] = value

            self._skip_whitespace()
            if self.text[self.pos] == ',':
                self.pos += 1
            elif self.text[self.pos] == '}':
                continue
            else:
                raise ValueError("Expected ',' or '}' in object")

        return obj

    def _parse_array(self) -> List[Any]:
        """Parse a TONL array"""
        self.pos += 1  # Skip '['
        arr = []

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.text[self.pos] == ']':
                self.pos += 1
                break

            value = self._parse_value()
            arr.append(value)

            self._skip_whitespace()
            if self.text[self.pos] == ',':
                self.pos += 1
            elif self.text[self.pos] == ']':
                continue
            else:
                raise ValueError("Expected ',' or ']' in array")

        return arr

    def _parse_string(self) -> str:
        """Parse a quoted string"""
        if self.text[self.pos] != '"':
            raise ValueError("Expected string")

        self.pos += 1  # Skip opening quote
        result = ""
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char == '"':
                self.pos += 1
                break
            elif char == '\\' and self.pos + 1 < len(self.text):
                # Handle escape sequences
                next_char = self.text[self.pos + 1]
                if next_char == '"':
                    result += '"'
                    self.pos += 2
                elif next_char == '\\':
                    result += '\\'
                    self.pos += 2
                else:
                    result += char
                    self.pos += 1
            else:
                result += char
                self.pos += 1

        return result

    def _parse_string_or_identifier(self) -> str:
        """Parse a string or bare identifier"""
        if self.text[self.pos] == '"':
            return self._parse_string()
        else:
            # Parse bare identifier (for object keys)
            start = self.pos
            while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] in '_-'):
                self.pos += 1
            return self.text[start:self.pos]

    def _parse_number(self) -> Union[int, float]:
        """Parse a number"""
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] in '.-+eE'):
            self.pos += 1

        num_str = self.text[start:self.pos]
        try:
            if '.' in num_str or 'e' in num_str.lower():
                return float(num_str)
            else:
                return int(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: {num_str}")

    def _skip_whitespace(self):
        """Skip whitespace characters"""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1


# Convenience functions
def encodeTONL(data: Any, smart: bool = True) -> str:
    """Encode Python data to TONL format"""
    encoder = TONLEncoder(smart=smart)
    return encoder.encode(data)


def decodeTONL(tonl_str: str) -> Any:
    """Decode TONL string to Python data"""
    decoder = TONLDecoder()
    return decoder.decode(tonl_str)


# Test functions
def test_basic_functionality():
    """Test basic TONL encoding/decoding"""
    print("Testing TONL Basic Functionality")
    print("=" * 40)

    # Test data
    test_cases = [
        {"name": "John", "age": 25, "active": True},
        [1, 2, 3, "hello", False],
        {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]},
        "simple string",
        42,
        True,
        None
    ]

    for i, data in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Original: {data}")

        try:
            encoded = encodeTONL(data)
            print(f"Encoded:  {encoded}")

            decoded = decodeTONL(encoded)
            print(f"Decoded:  {decoded}")

            # Check if round-trip works
            if data == decoded:
                print("[OK] Round-trip successful")
            else:
                print("[FAIL] Round-trip failed")
                print(f"Expected: {data}")
                print(f"Got:      {decoded}")

        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n" + "=" * 40)


def test_tabular_format():
    """Test tabular format for arrays of objects"""
    print("\nTesting TONL Tabular Format")
    print("=" * 40)

    # Test tabular data
    users = [
        {"id": 1, "name": "Alice", "age": 25, "active": True},
        {"id": 2, "name": "Bob", "age": 30, "active": False},
        {"id": 3, "name": "Charlie", "age": 35, "active": True}
    ]

    print("Original data:")
    for user in users:
        print(f"  {user}")

    try:
        encoded = encodeTONL(users)
        print(f"\nEncoded (tabular):\n{encoded}")

        decoded = decodeTONL(encoded)
        print(f"\nDecoded:")
        for user in decoded:
            print(f"  {user}")

        if users == decoded:
            print("\n[OK] Tabular round-trip successful")
        else:
            print("\n[FAIL] Tabular round-trip failed")

    except Exception as e:
        print(f"[ERROR] Tabular test error: {e}")


if __name__ == "__main__":
    test_basic_functionality()
    test_tabular_format()






