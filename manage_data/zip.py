import zipfile

zip_file = zipfile.ZipFile(r"C:\Users\DELL\Desktop\datasets\jester\archive2.zip")
f_content = zip_file.namelist()
print(f_content)
total_num = len(f_content)
num = 0

for file in f_content:
    zip_file.extract(file, r"C:\Users\DELL\Desktop\datasets\jester\jester1")
    num += 1
    print("{}/{}".format(num, total_num))

zip_file.close()

