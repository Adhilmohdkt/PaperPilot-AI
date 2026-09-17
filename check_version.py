import deepeal
print('version:', deepeal.__version__)
print('Dir:', [x for x in dir(deepeal) if not x.startswith('_')][:10])