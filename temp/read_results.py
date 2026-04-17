content = open('temp/test_results.txt', encoding='utf-16-le', errors='replace').read()
lines = [l.strip() for l in content.splitlines() if l.strip()]
for l in lines:
    print(l)
