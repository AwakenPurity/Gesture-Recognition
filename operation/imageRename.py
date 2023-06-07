import os


def rename_files(directory, className):
    # 获取目录下的所有文件名
    files = os.listdir(directory)

    # 初始化编号
    count = 1

    # 遍历文件名
    for file_name in files:
        # 构建旧文件路径
        old_path = os.path.join(directory, file_name)

        # 构建新文件名
        new_file_name = className + str(count) + os.path.splitext(file_name)[1]

        # 构建新文件路径
        new_path = os.path.join(directory, new_file_name)

        # 重命名文件
        os.rename(old_path, new_path)

        # 增加计数
        count += 1


# 指定目录路径
# paper_directory_path = 'C:/Users/10935/Desktop/Gesture Recognition/datasets/paper/'
# rename_files(paper_directory_path, "paper")

# rock_directory_path = 'C:/Users/10935/Desktop/Gesture Recognition/datasets/rock/'
# rename_files(rock_directory_path, "rock")

# scissor_directory_path = 'C:/Users/10935/Desktop/Gesture Recognition/datasets/scissor/'
# rename_files(scissor_directory_path, "scissor")

# scissor_directory_path = 'C:/Users/10935/Desktop/Gesture Recognition/datasets/environment/'
# rename_files(scissor_directory_path, "environment")

scissor_directory_path = 'C:/Users/10935/Desktop/Gesture Recognition/datasets/people/'
rename_files(scissor_directory_path, "people")