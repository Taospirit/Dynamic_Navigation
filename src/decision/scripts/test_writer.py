import os
folder = 'data_collection'
write_file_name = 'dqn_pass'

def write_data(folder_name, file_name, data, create=True, file_type='txt'): # wirte data into file
    assert isinstance(folder_name, str) and isinstance(file_name, str), \
        'folder_name and file_name must be str'
    dir_path = os.path.dirname(__file__)
    file_dir_path = os.path.join(dir_path, folder_name)
    # check folder exist 
    if not os.path.exists(file_dir_path):
        os.mkdir(file_dir_path)

    file_ = file_name + '.' + file_type
    file_path = file_dir_path + '/' + file_
    open_type = 'a'

    if not file_ in os.listdir(file_dir_path): # 不存在时直接建立
        print (f'not find the file in {folder_name}, create new file {file_}')
    elif not create: # 存在文件: 选择 覆盖 \ 新建 default=新建
        print (f'find the file in {folder_name}, overwrite file {file_}')
        open_type = 'w'
    else:
        num = 0
        while os.path.exists(file_path):
            num += 1
            file_ = file_name + str(num) + '.' + file_type
            file_path = file_dir_path + '/' + file_
        print (f'find the file in {folder_name}, create new file {file_}')

    with open(file_path, open_type) as f:
        f.write(data+'\n')

write_data(folder, 1, 'hhhh')
write_data(folder, 'qqqq', 'aaaa', create=False)
