__author__ = "lybroman@hotmail.com"

import os, shutil, collections

counter_dict = collections.defaultdict(int)

src_dir = {'train': './train', 'test': './test'}

# I use the new string formatting feature f-string in python v3,
# please change the format string according to your python version
for i in range(1, 6):
    os.mkdir(f'./{i}_train')
    os.mkdir(f'./{i}_test')

for tag, folder in src_dir.items():
    for file in os.listdir(folder):
        score = round(float(file.split('.txt')[0].split('_')[-1]))
        counter_dict[f'{score}_{tag}'] += 1
        shutil.copy2(f'./{folder}/{file}', f'./{score}_{tag}/{file}'.format(score, file))

# defaultdict(<class 'int'>,
# {'2_train': 3019, '1_train': 2981, '5_train': 6000, '4_test': 1500, '2_test': 2000, '5_test': 500})
print(counter_dict)
