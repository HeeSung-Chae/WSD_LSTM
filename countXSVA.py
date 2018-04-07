import glob
import os

dir = '../SkipGram/NoContent/posSentence/'
tGlob = glob.glob(os.path.join(dir, '*.txt'))

xsvCount = 0
xsaCount = 0
for a in tGlob:
    fileOpen = open(a, 'r', encoding='utf-8')
    fileList = fileOpen.readlines()

    print(a)
    for b in fileList:
        xsvCount += b.count('XSV')
        xsaCount += b.count('XSA')
    print(xsvCount, xsaCount)

print()
print(xsvCount, xsaCount)
