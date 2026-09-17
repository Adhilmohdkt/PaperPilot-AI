import deepeval
print('type:', type(deepeal))
print('dir keys:', [k for k in dir(deepeal) if not k.startswith('_')][:30])
print('DeepEvalBaseLLM' in dir(type(deepeal)))
print('DeepEvalBaseLLM' in dir(deepeval))