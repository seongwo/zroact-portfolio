import re

path = '/home/capstone2/zroact-stage2/benchmark2/results/viz_standalone.html'
with open(path, 'r', encoding='utf-8') as f:
    html = f.read()

print('Length of html:', len(html))
print('Sections found:')
for m in re.finditer(r'<script[^>]*>', html):
    start = m.start()
    end = html.find('</script>', start)
    print(html[start:start+100], '... length:', end - start if end != -1 else 'no end')

print('\nFirst 2000 chars:')
print(html[:2000])

print('\nLast 1000 chars:')
print(html[-1000:])
