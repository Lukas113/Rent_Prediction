# -*- coding: utf-8 -*-

#modifies not correct lines
correct_obj = []
incorrect_obj = []
with open("object_file - modified.csv", "r") as f:
    for line in f:
        data_obj = line.split(",")
        if len(data_obj) == 18:
            correct_obj.append(line)
        else:
            incorrect_obj.append(line)
            
print("incorrect objects: ", len(incorrect_obj))
print("correct objects: ", len(correct_obj))

with open("object_file - modified_v2.csv", "w") as writer:
    writer.writelines(correct_obj)
    
with open("object_file - modified_v2_incorrect_objects.csv", "w") as inc_writer:
    inc_writer.writelines(incorrect_obj)
