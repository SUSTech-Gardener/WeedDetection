import os

image_files = []
print(os.path.join(os.getcwd(), "training/darknet/data/agri_data/data"))
# os.chdir(os.path.join("../darknet/data", "agri_data/data"))
for filename in os.listdir(os.path.join(os.getcwd(), "training/darknet/data/agri_data/data")):
    if filename.endswith(".jpeg"):
        image_files.append("data/agri_data/data/" + filename)
# os.chdir("..")
# print(image_files)
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
# os.chdir("..")