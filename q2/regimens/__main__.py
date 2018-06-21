from . import all_regimens

print('Available regimens:')
for regimen in all_regimens:
    print('\t', regimen, sep='')