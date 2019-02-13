import os


label_dir = '/home/zzh/ocr/dataset/ctpn_train/label'

txt_list = os.listdir(label_dir)

for txt in txt_list:
    with open(os.path.join(label_dir, txt), 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls, x1, y1, x2, y2 = line.split('\t')
        width = float(x2) - float(x1)
        height = float(y2) - float(y1)
        if float(x1)<0 or float(x2)<0 or float(y1)<0 or float(y2)<0 :
            print('error file: ', txt, 'pos:', line)



"""
64
0174
"""