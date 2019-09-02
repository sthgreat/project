import os
import random
import xml.etree.ElementTree as ET


def getRecord_ByXml(Xml_path, data_path):
    '''
    从单一Xml文件中拿取记录
    :param Xml_path: xml文件路径
    :param data_path: 训练集路径
    :return: 一个包含众多record的列表
    '''
    xml = Xml_path.split('/')[-1]
    train_data_name = xml.split('.')[0] + '.jpg'
    record_li = []

    tree = ET.parse(Xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    for object in objects:
        if(object.find('name').text == 'ship'):
            bndbox = object.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            record = data_path + train_data_name + ',' + xmin + ',' + ymin + ',' + xmax + ',' + ymax + ',' + object.find('name').text + '\n'
            record_li.append(record)
    print(record_li)
    return record_li


def Write_ShipRecordToFile(data_path, LabelFile_path, Xml_path):
    '''
    获取船只信息
    :param data_path:
    :param LabelFile_path:
    :param Xml_path:
    :return:
    '''
    fileList = os.listdir(Xml_path)
    for file in fileList:
        xmlfile = Xml_path + file
        record_li = getRecord_ByXml(xmlfile, data_path)
        with open(LabelFile_path, 'a') as f:
            for record in record_li:
                f.writelines(record)
                

def getRecord_ByXml_noShip(Xml_path, data_path):
    xml = Xml_path.split('/')[-1]
    train_data_name = xml.split('.')[0] + '.jpg'

    label_li = compute_noShipLabel(Xml_path)  # label_li [xmin,xmax,ymin,ymax]
    if (len(label_li) == 0):
        return ''
    record = data_path + train_data_name + ',' + str(label_li[0]) + ',' + str(label_li[2])\
             + ',' + str(label_li[1]) + ',' + str(label_li[3]) + ',' + 'notShip' + '\n'
    return record


def compute_noShipLabel(Xml_path):
    # 输入单一xml文件，读取该文件获得船只样本，
    # 考虑到训练集的问题（补全非海上非船只样本），预计单一图片只节选一块样本，最后再删除些记录
    # 统一返回一个list, list中包含样本方框的四个值，若没有符合标准的，返回一个空列表
    xml = Xml_path.split('/')[-1]
    tree = ET.parse(Xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    object_matrix = []
    for object in objects:
        if (object.find('name').text == 'ship'):
            bndbox = object.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            object_matrix.append([xmin, xmax, ymin, ymax])
    # 获得了包含有已有方框的矩阵matrix，接下来需要根据该矩阵生成一个不包含船只信息的方框label
    label_li = compute_noShipLabel_method1(object_matrix)  # label_li [xmin,xmax,ymin,ymax]
    if len(label_li) == 0:
        return []
    else:
        return label_li


def compute_noShipLabel_method1(Matrix):
    '''
    返回一个带有四个角点数据的list(只有一个框)
    :param Matrix:
    :return:
    '''
    x_length = random.randint(15, 30)
    y_length = random.randint(15, 30)
    # 一个循环，反复判断方框是否符合要求
    flag = True
    turn = 0
    while flag:
        # 随机左上角点位置
        point_x = random.randint(1, 225)  # xmin
        point_y = random.randint(1, 225)  # ymax

        point_x_rightUp = point_x + x_length
        point_y_rightUp = point_y

        point_x_leftDown = point_x
        point_y_leftDown = point_y + y_length

        point_x_rightDown = point_x + x_length
        point_y_rightDown = point_y + y_length

        for area in Matrix:  # area是一个一维list [xmin,xmax,ymin,ymax]
            if (int(area[0]) <= point_x <= int(area[1])) and (int(area[2]) <= point_y <= int(area[3])):
                break
            if (int(area[0]) <= point_x_rightUp <= int(area[1])) and (int(area[2]) <= point_y_rightUp <= int(area[3])):
                break
            if (int(area[0]) <= point_x_leftDown <= int(area[1])) and (int(area[2]) <= point_y_leftDown <= int(area[3])):
                break
            if (int(area[0]) <= point_x_rightDown <= int(area[1])) and (int(area[2]) <= point_y_rightDown <= int(area[3])):
                break
            return [point_x, point_x + x_length, point_y, point_y + y_length]  # [xmin,xmax,ymin,ymax]
        turn += 1
        if turn == 30:
            flag = False
    return []


def Write_noShipRecordToFile(data_path, LabelFile_path, Xml_path):
    '''
    获取船只信息
    :param data_path:
    :param LabelFile_path:
    :param Xml_path:
    :return:
    '''
    fileList = os.listdir(Xml_path)
    for file in fileList:
        xmlfile = Xml_path + file
        record = getRecord_ByXml_noShip(xmlfile, data_path)
        if (record == ''):
            continue
        with open(LabelFile_path, 'a') as f:
            f.writelines(record)


if __name__ == '__main__':
    file_path = 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\labelXml\Gao_ship_hh_02016082544040302.xml'
    data_path = 'F:\python_workplace\project\keras_frcnn-master\data\ship_detection/'
    label_path = 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\label.txt'
    Write_ShipRecordToFile(data_path, label_path, 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\labelXml/')
    # label_path = 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\label_noship.txt'
    # Write_noShipRecordToFile(data_path, label_path, 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\labelXml/')
