import deepeal as d
print('Module file:', d.__file__)
print('DeepEvalBaseLLM in dir:', 'DeepEvalBaseLLM' in dir(d))
print('Dir sample:', [k for k in dir(d) if 'LLM' in k or 'model' in k.lower()][:10])
PYEOF