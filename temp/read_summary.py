import re
content = open('temp/test_results.txt', encoding='utf-16-le', errors='replace').read()
match = re.search(r'ACCURACY REPORT.*', content, re.DOTALL)
if match:
    print(match.group(0))
else:
    print("Could not find ACCURACY REPORT")
