import os

import torch
import torch.utils.data
from PIL import Image
import sys
import scipy.io as scio
import cv2
import numpy
from tqdm import tqdm
# from maskrcnn_benchmark.data.transforms import Compose

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class IP102Dataset(torch.utils.data.Dataset):
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """

    """
    THESE CLASSES HAVE NO TEST IMAGE:
        Viteus vitifoliae
        Colomerus vitis
        Polyphagotars onemus latus
        Phyllocoptes oleiverus ashmead
        Parlatoria zizyphus lucus
        Brevipoalpus lewisi mcgregor
    """

    CLASSES = ("__background__ ", 'Rice leaf roller', 'Rice leaf caterpillar', 'Paddy stem maggot', 'Asiatic rice borer', 'Yellow rice borer', 
    'Rice gall midge', 'Rice stemfly', 'Brown plant hopper', 'White backed plant hopper', 'Small brown plant hopper', 
    'Rice water weevil', 'Rice leafhopper', 'Grain spreader thrips', 'Rice shell pest', 'Grub', 'Mole cricket', 'Wireworm', 
    'White margined moth', 'Black cutworm', 'Large cutworm', 'Yellow cutworm', 'Red spider', 'Corn borer', 'Army worm', 
    'Aphids', 'Potosiabre vitarsis', 'Peach borer', 'English grain aphid', 'Green bug', 'Bird cherry-oataphid', 
    'Wheat blossom midge', 'Penthaleus major', 'Longlegged spider mite', 'Wheat phloeothrips', 'Wheat sawfly', 
    'Cerodonta denticornis', 'Beet fly', 'Flea beetle', 'Cabbage army worm', 'Beet army worm', 'Beet spot flies', 
    'Meadow moth', 'Beet weevil', 'Sericaorient alismots chulsky', 'Alfalfa weevil', 'Flax budworm', 'Alfalfa plant bug', 
    'Tarnished plant bug', 'Locustoidea', 'Lytta polita', 'Legume blister beetle', 'Blister beetle', 'Therioaphis maculata buckton', 
    'Odontothrips loti', 'Thrips', 'Alfalfa seed chalcid', 'Pieris canidia', 'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 
    'Colomerus vitis', 'Brevipoalpus lewisi mcgregor', 'Oides decempunctata', 'Polyphagotars onemus latus', 
    'Pseudococcus comstocki kuwana', 'Parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus', 
    'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 
    'Panonchus citri mcgregor', 'Phyllocoptes oleiverus ashmead', 'Icerya purchasi maskell', 'Unaspis yanonensis', 
    'Ceroplastes rubens', 'Chrysomphalus aonidum', 'Parlatoria zizyphus lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 
    'Tetradacus c bactrocera minax', 'Dacus dorsalis(hendel)', 'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 
    'Phyllocnistis citrella stainton', 'Toxoptera citricidus', 'Toxoptera aurantii', 'Aphis citricola vander goot', 
    'Scirtothrips dorsalis hood', 'Dasineura sp', 'Lawana imitata melichar', 'Salurnis marginella guerr', 
    'Deporaus marginatus pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 
    'Sternochetus frigidus', 'Cicadellidae')

    # remove 6 classes which has no test sample
    CLASSES96 = ("__background__ ", 'Rice leaf roller', 'Rice leaf caterpillar', 'Paddy stem maggot', 'Asiatic rice borer', 'Yellow rice borer', 
    'Rice gall midge', 'Rice stemfly', 'Brown plant hopper', 'White backed plant hopper', 'Small brown plant hopper', 
    'Rice water weevil', 'Rice leafhopper', 'Grain spreader thrips', 'Rice shell pest', 'Grub', 'Mole cricket', 'Wireworm', 
    'White margined moth', 'Black cutworm', 'Large cutworm', 'Yellow cutworm', 'Red spider', 'Corn borer', 'Army worm', 
    'Aphids', 'Potosiabre vitarsis', 'Peach borer', 'English grain aphid', 'Green bug', 'Bird cherry-oataphid', 
    'Wheat blossom midge', 'Penthaleus major', 'Longlegged spider mite', 'Wheat phloeothrips', 'Wheat sawfly', 
    'Cerodonta denticornis', 'Beet fly', 'Flea beetle', 'Cabbage army worm', 'Beet army worm', 'Beet spot flies', 
    'Meadow moth', 'Beet weevil', 'Sericaorient alismots chulsky', 'Alfalfa weevil', 'Flax budworm', 'Alfalfa plant bug', 
    'Tarnished plant bug', 'Locustoidea', 'Lytta polita', 'Legume blister beetle', 'Blister beetle', 'Therioaphis maculata buckton', 
    'Odontothrips loti', 'Thrips', 'Alfalfa seed chalcid', 'Pieris canidia', 'Apolygus lucorum', 'Limacodidae', 
    'Oides decempunctata', 
    'Pseudococcus comstocki kuwana', 'Parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus', 
    'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 
    'Panonchus citri mcgregor', 'Icerya purchasi maskell', 'Unaspis yanonensis', 
    'Ceroplastes rubens', 'Chrysomphalus aonidum', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 
    'Tetradacus c bactrocera minax', 'Dacus dorsalis(hendel)', 'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 
    'Phyllocnistis citrella stainton', 'Toxoptera citricidus', 'Toxoptera aurantii', 'Aphis citricola vander goot', 
    'Scirtothrips dorsalis hood', 'Dasineura sp', 'Lawana imitata melichar', 'Salurnis marginella guerr', 
    'Deporaus marginatus pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 
    'Sternochetus frigidus', 'Cicadellidae')

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, external_proposal=False, old_classes=[],
                 new_classes=[], excluded_classes=[], is_train=True):
        self.root = data_dir
        self.image_set = split  # train, validation, test
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.use_external_proposal = external_proposal

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self._proposalpath = os.path.join(self.root, "EdgeBoxesProposals", "%s.mat")

        self._img_height = 0
        self._img_width = 0

        self.old_classes = old_classes
        self.new_classes = new_classes
        self.exclude_classes = excluded_classes
        self.is_train = is_train


        # load data from all categories
        # self._normally_load_voc()

        # do not use old data
        if self.is_train:  # training mode
            print('ip102.py | in training mode')
            self._load_img_from_NEW_cls_without_old_data()
        else:
            print('ip102.py | in test mode')
            self._load_img_from_NEW_and_OLD_cls_without_old_data()

    def _normally_load_voc(self):
        """ load data from all 20 categories """

        # print("voc.py | normally_load_voc | load data from all 20 categories")
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.final_ids = self.ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # image_index : image_id

        cls = IP102Dataset.CLASSES96
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class_name : class_id

    def _load_img_from_NEW_and_OLD_cls_without_old_data(self):
        self.ids = []
        total_classes = self.new_classes + self.old_classes
        for w in range(len(total_classes)):
            category = total_classes[w]
            img_per_categories = []
            with open(self._imgsetpath % "{0}_{1}".format(category, self.image_set)) as f:
                buff = f.readlines()
            buff = [x.strip("\n") for x in buff]
            for i in range(len(buff)):
                a = buff[i]
                b = a.split(' ')
                if b[1] == "-1":  # do not contain the category object
                    pass
                elif b[2] == '0':  # contain the category object -- difficult level
                    if self.is_train:
                        pass
                    else:
                        img_per_categories.append(b[0])
                        self.ids.append(b[0])
                else:
                    img_per_categories.append(b[0])
                    self.ids.append(b[0])
            # print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | number of images in {0}_{1}: {2}'.format(category, self.image_set, len(img_per_categories)))

        # check for image ids repeating
        self.final_ids = []
        for id in self.ids:
            repeat_flag = False
            for final_id in self.final_ids:
                if id == final_id:
                    repeat_flag = True
                    break
            if not repeat_flag:
                self.final_ids.append(id)
        # print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = IP102Dataset.CLASSES96
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def _load_img_from_NEW_cls_without_old_data(self):
        self.ids = []
        for incremental in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                x = buff[i]
                x = x.split(' ')
                if x[1] == '-1':
                    pass
                elif x[2] == '0':  # include difficult level object
                    if self.is_train:
                        pass
                    else:
                        img_ids_per_category.append(x[0])
                        self.ids.append(x[0])
                else:
                    img_ids_per_category.append(x[0])
                    self.ids.append(x[0])
            # print('ip102.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)
            # print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = IP102Dataset.CLASSES96
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.final_ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.use_external_proposal:
            proposal = self.get_proposal(index)
            proposal = proposal.clip_to_image(remove_empty=True)
        else:
            proposal = None

        # draw_image(img, target, proposal, "{0}_{1}_voc_getitem".format(index, img_id))

        if self.transforms is not None:
            # if not self.is_train:
            #     visualize_transform = Compose([self.transforms.transforms[1], self.transforms.transforms[3]])
            #     img_vis, target_vis, proposal_vis = visualize_transform(img, target, proposal)
            #     img, target, proposal = self.transforms(img, target, proposal)
            #     return img, target, proposal, img_vis
            img, target, proposal = self.transforms(img, target, proposal)

        return img, target, proposal, index

    def __len__(self):
        return len(self.final_ids)

    def get_groundtruth(self, index):
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        self._img_height = height
        self._img_width = width
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def get_proposal(self, index):
        boxes = []

        img_id = self.final_ids[index]
        proposal_path = self._proposalpath % "{0}".format(img_id)
        proposal_raw_data = scio.loadmat(proposal_path)
        proposal_data = proposal_raw_data['bbs']
        proposal_length = proposal_data.shape[0]
        for i in range(2000):
            # print('i: {0}'.format(i))
            if i >= proposal_length:
                break
            left = proposal_data[i][0]
            top = proposal_data[i][1]
            width = proposal_data[i][2]
            height = proposal_data[i][3]
            score = proposal_data[i][4]
            right = left + width
            bottom = top + height
            box = [left, top, right, bottom]
            boxes.append(box)
        img_height = self._img_height
        img_width = self._img_width

        boxes = torch.tensor(boxes, dtype=torch.float32)
        proposal = BoxList(boxes, (img_width, img_height), mode="xyxy")

        return proposal

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            # name = obj.find("name").text.lower().strip() # IP102中有大写字母，去掉lower()  // wangcong
            name = obj.find("name").text.strip()


            old_class_flag = False
            for old in self.old_classes:
                if name == old:
                    old_class_flag = True
                    break
            exclude_class_flag = False
            for exclude in self.exclude_classes:
                if name == exclude:
                    exclude_class_flag = True
                    break

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]
            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            if exclude_class_flag:
                pass
                #print('voc.py | incremental train | object category belongs to exclude categoires: {0}'.format(name))
            elif self.is_train and old_class_flag:
                pass
                #print('voc.py | incremental train | object category belongs to old categoires: {0}'.format(name))
            else:
                boxes.append(bndbox)
                gt_classes.append(self.class_to_ind[name])
                difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return IP102Dataset.CLASSES96[class_id]

    def get_img_id(self, index):
        img_id = self.final_ids[index]
        return img_id



# create txt file for each class
def create_singleclass_txt(data_dir, split):
    print("---------------processing each class samples in {}.----------------".format(split))
    # use_difficult = False
    # transforms = None
    # dataset = IP102Dataset(data_dir, split, use_difficult, transforms)
    xml_file_path = os.path.join(data_dir, 'Annotations/')  # xml文件路径
    save_Path = os.path.join(data_dir, 'ImageSets/Main')
    # total_xml = os.listdir(xml_file_path)  # 得到文件夹下所有文件名称
    id_list_file = os.path.join(save_Path, '{0}.txt'.format(split))
    imgIds = [id_.strip() for id_ in open(id_list_file)]
    print("sample length:", len(imgIds))

    count_class_sample_path = os.path.join(save_Path, '{0}_class_count.txt'.format(split))
    count_class_sample_file = open(count_class_sample_path, 'w')



    for idx in tqdm(range(len(IP102Dataset.CLASSES)-1)): # remove background
        class_name = IP102Dataset.CLASSES[idx+1]
        count = 0
        # 创建txt
        class_txt = open(os.path.join(save_Path, str(class_name) + '_{}.txt'.format(split)), 'w')
        for k in range(len(imgIds)):
            xml_name = imgIds[k]  # xml的名称
            # print(xml_name)
            xml_path = os.path.join(xml_file_path, xml_name + '.xml')
            # 将获取的xml文件名送入到dom解析
            tree = ET.parse(xml_path)  # 输入xml文件具体路径
            root = tree.getroot()
            # 创建一个空列表，用于保存xml中object的name
            object_names = []
            for object in root.findall('object'):
                # 获取xml object标签<name>  IP102里标注的是0-101的数字 已替换为名称
                # object_name = IP102Dataset.CLASSES[int(object.find('name').text)+1] 
                object_name = object.find('name').text
                object_names.append(object_name)

            if len(object_names) > 0 and class_name in object_names:  # 存在object（矩形框并且class_name在object_name列表中  # 这里的类别对应上面VOC_CLASSES
                class_txt.write(xml_name + '  ' + str(1) + "\n")
                count += 1
            else:
                class_txt.write(xml_name + ' ' + str(-1) + "\n")
        class_txt.close()
        count_class_sample_file.write(class_name + "," + str(count) + "\n")
        
    count_class_sample_file.close()

# name in xml: number --> text
def change_xml_objname(data_dir):
    xml_file_path = os.path.join(data_dir, 'Annotations/')  # xml文件路径
    total_xml = os.listdir(xml_file_path)  # 得到文件夹下所有文件名称
    for k in range(len(total_xml)):
        xml_name = total_xml[k][:-4]  # xml的名称
        # print(xml_name)
        xml_path = os.path.join(xml_file_path, xml_name + '.xml')
        # 将获取的xml文件名送入到dom解析
        tree = ET.parse(xml_path)  # 输入xml文件具体路径
        root = tree.getroot()

        for object in root.findall('object'):
            # 获取xml object标签<name>
            class_id = object.find('name').text.replace(" ", "")
            print(class_id + "--")
            try:
                class_id = int(class_id)              
            except:
                print(xml_name)
            else:
                object_name = IP102Dataset.CLASSES[class_id + 1]
                print(object_name)
                object.find('name').text = object_name

        tree.write(xml_path)



def main():
    data_dir = "/home/Datasets/IP102_v1.1/Detection/VOC2007"
    # split = "test"  # train, val, test
    # use_difficult = False
    # transforms = None
    # dataset = PascalVOCDataset(data_dir, split, use_difficult, transforms)

    create_singleclass_txt(data_dir, "test")
    create_singleclass_txt(data_dir, "trainval")

    # change_xml_objname(data_dir)



if __name__ == '__main__':
    main()

    