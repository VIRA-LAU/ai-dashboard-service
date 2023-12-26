import os
import sys

# with open(self.img_index) as f:
#     lines = [x.strip().split(" ") for x in f.readlines()]
# if len(lines[0]) == 2:
#     self.image_set_index = [x[0] for x in lines]
#     self.frame_id = [int(x[1]) for x in lines]
# else:
#     self.image_set_index = ["%s/%06d" % (x[0], int(x[2])) for x in lines]
#     self.pattern = [x[0] + "/%06d" for x in lines]
#     self.frame_id = [int(x[1]) for x in lines]
#     self.frame_seg_id = [int(x[2]) for x in lines]
#     self.frame_seg_len = [int(x[3]) for x in lines]

def rename():
    path="../PTSEFormer/data_root/new_actions/Data/train/"
    new_file_name = "00000"
    for file_name in os.listdir(path):
        source = path + file_name
        file_name_red = file_name.split('.xml')[0]
        file_name_red = file_name_red.split('.rf')[0]
        file_name_red = file_name_red[:-4]
        index = file_name_red[-5:]
        dest = path + new_file_name + "_" + index + ".jpg"
        os.rename(source, dest)

# def rename():
#     path="../PTSEFormer/data_root/new_actions/Annotations/train/"
#     new_file_name = "00000"
#     for file_name in os.listdir(path):
#         source = path + file_name
#         file_name_red = file_name.split('.jpg')[0]
#         # file_name_red = file_name_red.split('.rf')[0]
#         # file_name_red = file_name_red[:-4]
#         # index = file_name_red[-5:]
#         dest = path + file_name_red + ".xml"
#         os.rename(source, dest)


def script_train():
    # "../PTSEFormer/data_root/new_actions/Data"
    # "../PTSEFormer/data_root/new_actions/Annotations"
    txt_file = "../PTSEFormer/data_root/new_actions/ImageSets/train.txt"
    for file_name in os.listdir("../PTSEFormer/data_root/new_actions/Annotations/train"):
        file_name_red = file_name.split('.xml')[0]
        index = file_name_red.split('_')[1]
        line = file_name_red + " " + str(index) + "\n"
        with open(txt_file, 'a') as f:
            f.write(line)

def script_val():
    # "../PTSEFormer/data_root/new_actions/Data"
    # "../PTSEFormer/data_root/new_actions/Annotations"
    txt_file = "../PTSEFormer/data_root/new_actions/ImageSets/valid.txt"
    for file_name in os.listdir("../PTSEFormer/data_root/new_actions/Annotations/valid"):
        file_name_red = file_name.split('.xml')[0]
        index = file_name_red.split('_')[1]
        line = file_name_red + " " + str(index) + "\n"
        with open(txt_file, 'a') as f:
            f.write(line)
