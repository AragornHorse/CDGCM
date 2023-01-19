

train_label_path = r"C:\Users\DELL\Desktop\datasets\jester\label\jester-v1-train.csv"

num2label = []
idx2num = {}

with open(train_label_path, "r") as f:
    for l in f:
        line = l[:-1]
        idx, label = line.split(";")
        idx = int(idx)
        if label not in num2label:
            num2label.append(label)
        num = num2label.index(label)
        idx2num[idx] = num

print(num2label)
print(idx2num)

with open(r"C:\Users\DELL\Desktop\datasets\jester\index\num2label.txt", 'w') as f:
    for idx, label in enumerate(num2label):
        f.write("{};{}\n".format(idx, label))

import glob

lst = glob.glob(r"C:\Users\DELL\Desktop\datasets\jester\jester1\*")

with open(r"C:\Users\DELL\Desktop\datasets\jester\index\nolabel.txt", 'w') as ff:
    with open(r"C:\Users\DELL\Desktop\datasets\jester\index\idx2path&num.txt", 'w') as f:
        for file in lst:
            idx = int(file.split("\\")[-1])
            try:
                f.write("{};{};{}\n".format(idx, file, idx2num[idx]))
            except:
                ff.write("{};{}\n".format(idx, file))



