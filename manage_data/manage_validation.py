

validation_label_path = r"C:\Users\DELL\Desktop\datasets\jester\label\jester-v1-validation.csv"
num2label_path = r"C:\Users\DELL\Desktop\datasets\jester\index\num2label.txt"


label2num = {}
with open(num2label_path, 'r') as f:
    for line in f:
        num, label = line[:-1].split(';')
        num = int(num)
        label2num[label] = num

idx2num = {}

with open(validation_label_path, "r") as f:
    for l in f:
        line = l[:-1]
        idx, label = line.split(";")
        idx = int(idx)
        num = label2num[label]
        idx2num[idx] = num


import glob

lst = glob.glob(r"C:\Users\DELL\Desktop\datasets\jester\jester1\*")

with open(r"C:\Users\DELL\Desktop\datasets\jester\index\nolabel_validation.txt", 'w') as ff:
    with open(r"C:\Users\DELL\Desktop\datasets\jester\index\idx2path&num_validation.txt", 'w') as f:
        for file in lst:
            idx = int(file.split("\\")[-1])
            try:
                f.write("{};{};{}\n".format(idx, file, idx2num[idx]))
            except:
                ff.write("{};{}\n".format(idx, file))





