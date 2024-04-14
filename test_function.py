# import pandas as pd

# # Read the CSV file
# file_path = r"D:\download\list_bbox_celeba.txt"
# with open(file_path,"r") as f:
#     f.readline()  # Skip the header
#     data = [line.strip().split(" ") for line in f.readlines()]

# clean_data = {}
# for line in data:
#     file_name = line.pop(0)
    
#     # Convert values to integers and calculate x2, y2
#     line = [int(value.replace(" ","")) for value in line if value.replace(" ","") != ""]
#     line[2] += line[0]  # x2 = x1 + width
#     line[3] += line[1]  # y2 = y1 + height
#     clean_data[file_name] = line

# # Convert dictionary to DataFrame
# df = pd.DataFrame.from_dict(clean_data, orient='index', columns=['x1', 'y1', 'x2', 'y2'])

# # Add image_id column
# df['image_id'] = df.index

# # Reorder columns
# df = df[['image_id', 'x1', 'y1', 'x2', 'y2']]

# # Write to a new CSV file
# output_file = r"D:\download\list_bbox_celeba_x1_y1_x2_y2.txt"
# df.to_csv(output_file, sep=" ", index=False)

from resources.landmarks.extract_face import extrace_face
extrace_face(r"D:\download\oneface",r"D:\download\oneface_extrated2",annotation_path=r"D:\download\list_bbox_celeba_x1_y1_x2_y2.txt",type_landmarks="face_alignment")

