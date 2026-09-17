import deepeval.metrics
# List all available metrics
attrs = [a for a in dir(deepeal.metrics) if not a.startswith('__')]
print('Available metrics:')
for a in attrs:
    print(f'  {a}')