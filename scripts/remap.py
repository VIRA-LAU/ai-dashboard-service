import os
import sys

# ['basketball', 'netbasket', 'netempty'] to ['netbasket', 'netempty', 'basketball']

# names: ['basketball', 'netbasket', 'netempty']

# ['dribbling', 'no action', 'shooting', 'walking'] to [ 'no action', 'walking', 'shooting', 'dribbling' ]

# ['dribbling', 'no action', walking'] exception

def change_netbasket(dir):
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'r') as f:
            lines = f.readlines()

        for i in range(0,len(lines)):
            if(lines[i][0] == '0'):
                lines[i] = lines[i].replace("0", "2", 1)

            elif(lines[i][0] == '1'):
                lines[i] = lines[i].replace("1", "0", 1)

            elif(lines[i][0] == '2'):
                lines[i] = lines[i].replace("2", "1", 1)

        with open(os.path.join(dir, file), "w") as fo:
            fo.writelines(lines)


def change_action(dir):
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'r') as f:
            lines = f.readlines()

        for i in range(0,len(lines)):
            if(lines[i][0] == '0'):
                lines[i] = lines[i].replace("0", "3", 1)

            elif(lines[i][0] == '1'):
                lines[i] = lines[i].replace("1", "0", 1)

            elif(lines[i][0] == '3'):
                lines[i] = lines[i].replace("3", "1", 1)

        with open(os.path.join(dir, file), "w") as fo:
            fo.writelines(lines)


def combine_netbasket(dir):
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'r') as f:
            lines = f.readlines()

        for i in range(0,len(lines)):
            if(lines[i][0] == '0'):
                lines[i] = lines[i].replace("0", "4", 1)

            elif(lines[i][0] == '1'):
                lines[i] = lines[i].replace("1", "5", 1)

            elif(lines[i][0] == '2'):
                lines[i] = lines[i].replace("2", "6", 1)

        with open(os.path.join(dir, file), "w") as fo:
            fo.writelines(lines)


def copy_data(source, dest):
    for file in os.listdir(source):
        with open(os.path.join(source, file), 'r') as f:
            lines = f.readlines()

        with open(os.path.join(dest, file), 'a') as f:
            f.writelines(lines)


if __name__ == "__main__":
    change_netbasket("datasets/training_datasets/object_detection/netbasket/PhoneDatasetFive/train/labels")
    change_netbasket("datasets/training_datasets/object_detection/netbasket/PhoneDatasetFive/valid/labels")
    change_netbasket("datasets/training_datasets/object_detection/netbasket/PhoneDatasetFive/test/labels")
    # change_action("datasets/training_datasets/object_detection/actions/PhoneDatasetFive/train/labels")
    # change_action("datasets/training_datasets/object_detection/actions/PhoneDatasetFive/valid/labels")
    # change_action("datasets/training_datasets/object_detection/actions/PhoneDatasetFive/test/labels")
    # combine_netbasket("datasets/training_datasets/HardwareTwo_netbasket/labels")
    # copy_data("datasets/training_datasets/HardwareTwo_netbasket/labels", "datasets/training_datasets/HardwareDatasetTwo_Combined")