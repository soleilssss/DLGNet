'''
加载数据的原图，局部的图形。
'''

import PIL.Image as Image 
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
def make_dataset(path):
    imgs = []
    img_types = ['adenoma','cancer','normal','polyp']
    for i in range(4):
        img_dir_path = os.path.join(path,img_types[i],'image')
        img_name = os.listdir(img_dir_path)
        for name in img_name:
            img_path = os.path.join(img_dir_path,name)
            crop_path = img_path.replace('image','faster')
            #lbp_path = img_path.replace('image','LBP')
            imgs.append((img_path, crop_path, i, name))
    return imgs


class BowelDataset(data.Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_path, crop_path, label, name = self.imgs[index]
        img_x = Image.open(img_path)
        crop_x = Image.open(crop_path)
        if self.transform is not None:
            img_x, crop_x = self.transform(img_x, crop_x)
            
        return img_x, crop_x, label, name

    def __len__(self):
        return len(self.imgs)

    def getPath(self):
        return self.imgs



if __name__ == '__main__':
    '''
    error1 = [90, 110, 115, 123, 125, 134, 145, 155, 158, 162, 167]
    error2 = [126]
    error3 = [3, 4, 9, 22, 26, 37, 39, 41, 42, 51, 53, 75, 76]
    error4 = [183, 197, 232, 252, 265, 266, 272, 283, 291, 343, 345]
    error5 = [292]
    error6 = [224]
    error7 = [490]
    error8 = [451]
    error9 = [378, 409]
    error10 = [561, 571, 609, 622, 634, 653, 692, 711]
    error11 = []
    error12 = [566, 575, 594, 599, 615, 619, 633, 648, 684]
    '''



    error1 = [33, 39, 86, 89, 104, 114, 167, 168]
    error2  = [11, 25, 103, 129]
    error3 =  [0, 17, 22, 23, 31, 41, 58, 67, 75, 83, 87, 88, 137, 145]
    error4 = [181, 184, 244, 261, 326]
    error5 = [240]
    error6 = []
    error7 = [501]
    error8 = [508]
    error9 = [447, 467, 472, 534]
    error10 = [555, 557, 561, 569, 576, 578, 586, 603, 610, 617, 626, 633, 636, 643, 657, 658, 681, 691, 700, 712, 719]
    error11 = []
    error12 = [604, 663, 669, 679, 687]
    save_path = r'D:\肠镜分类结果\易错图像'
    # 先找到路径名，然后把相应的图像复制到已经准备好的文件夹中。
    # 为了避免其中位置的移动，直接打开图像，另存到一个准备好的文件夹中
    data = BowelDataset(r'D:\肠镜分类结果\double_code\test')
    img_types = ['adenoma','cancer','normal','polyp']
    path = data.getPath()
    for i in error1:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[0], img_types[1], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error2:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[0], img_types[2], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error3:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[0], img_types[3], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error4:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[1], img_types[0], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error5:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[1], img_types[2], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error6:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[1], img_types[3], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error7:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[2], img_types[0], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error8:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[2], img_types[1], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error9:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[2], img_types[3], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error10:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[3], img_types[0], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error11:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[3], img_types[1], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)

    for i in error12:
        img_path = path[i][0]
        save_name = os.path.join(save_path, img_types[3], img_types[2], os.path.split(img_path)[-1])
        img_array = Image.open(img_path)
        img_array.save(save_name)
