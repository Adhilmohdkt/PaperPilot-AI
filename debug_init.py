import os

filepath = r'C:\Users\ASUS\Desktop\PaperPilot AI\backend\retrieval\_\_init__.py'

# List all files in retrieval directory
retrieval_dir = r'C:\Users\ASUS\Desktop\PaperPilot AI\backend\retrieval'
all_files = os.listdir(retrieval_dir)
print(f'All files in retrieval: {all_files}')

# Find the __init__.py file
for f in all_files:
    if f.startswith('init') and f.endswith('.py'):
        full_path = os.path.join(retrieval_dir, f)
        print(f'\nFound init file: {full_path}')
        with open(full_path, 'rb') as fh:
            data = fh.read()
        print(f'Size: {len(data)}')
        print(f'Hex: {data.hex()}')
        print(f'Repr: {repr(data)}')
        nulls = [i for i, b in enumerate(data) if b == 0]
        print(f'Null byte positions: {nulls}')