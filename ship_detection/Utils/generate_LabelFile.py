import os
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


def Write_RecordToFile(data_path, LabelFile_path, Xml_path):
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


if __name__ == '__main__':
    file_path = 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\labelXml\Gao_ship_hh_02016082544040302.xml'
    data_path = 'F:\python_workplace\project\keras_frcnn-master\data\ship_detection/'
    label_path = 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\label.txt'
    Write_RecordToFile(data_path,label_path, 'F:\python_workplace\project\keras_frcnn-master\data\labelFile\labelXml/')
