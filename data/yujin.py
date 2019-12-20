w = open('yujin.txt', 'w', encoding='UTF8')
with open('유진데이터 텍스트파일.txt', 'rt', encoding='UTF8') as f:
	for s in f:
		#print(s[:-3]+'\t'+s[-2:-1])
		w.write(s[:-3]+'\t'+s[-2:-1]+'\n')