import deepeal
print('type:', type(deepeal))
print('dir keys:', [k for k in dir(deepeal) if not k.startswith('_')][:20])
print('DeepEvalBaseLLM' in dir(type(deepeal)))