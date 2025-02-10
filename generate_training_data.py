import json
from pathlib import Path

def generate_training_examples():
    # Create example Python code snippets
    training_data = [
        {
            "code": """def calculate_sum(a, b):
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b""",
            "description": "Function to add two numbers"
        },
        {
            "code": """def is_palindrome(text):
    \"\"\"Check if a string is palindrome.\"\"\"
    text = text.lower()
    return text == text[::-1]""",
            "description": "Function to check palindrome"
        },
        {
            "code": """class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height""",
            "description": "Class representing a rectangle"
        }
    ]

    # Add more examples
    for i in range(5):
        training_data.append({
            "code": f"""def process_list_{i}(items):
    \"\"\"Process a list of items.\"\"\"
    return [item.upper() if isinstance(item, str) else item * 2 for item in items]""",
            "description": f"Function to process list items (example {i})"
        })

    # Save the training data
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "code_examples.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Generated {len(training_data)} training examples")
    return output_dir / "code_examples.json"

if __name__ == "__main__":
    generate_training_examples()