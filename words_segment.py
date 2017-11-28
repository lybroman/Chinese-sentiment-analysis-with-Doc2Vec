__author__ = "lybroman@hotmail.com"

# this script is especially for Chinese docs,
# for English related docs, just doc.split(' ') is fine~

# using third-party package jieba to
import jieba, os, traceback

label = ['train', 'test']

for tag in label:
    # score from 1 -> 5
    for score in range(1, 6):
        with open(f'{score}_{tag}.txt', 'w', encoding='utf-8') as f:
            for file in os.listdir(f'./{score}_{tag}'):
                try:
                    with open(f'./{score}_{tag}/{file}', 'r', encoding='gb2312') as ff:
                        # since doc won't be too long, so read all at once
                        line = ff.read()

                        # filter some non-related chars
                        filter_chars = "\r\n，。；！,.:;：、"
                        trans_dict = dict.fromkeys((ord(_) for _ in filter_chars), '')
                        line = line.translate(trans_dict)

                        # words segment
                        it = jieba.cut(line, cut_all=False)
                        _ = []
                        for w in it:
                            _.append(w)

                        f.write(' '.join(_) + '\n')
                except:
                    # bypass some bad samples for decoding errors
                    # print(traceback.format_exc())
                    # print(f'failed to parse {file}')
                    pass

