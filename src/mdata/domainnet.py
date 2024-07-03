import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
DIVISIONS = {
    'Furniture': ['bathtub', 'bed', 'bench', 'ceiling_fan', 'chair', 'chandelier', 'couch', 'door', 'dresser',
                  'fence', 'fireplace', 'floor_lamp', 'hot_tub', 'ladder', 'lantern', 'mailbox', 'picture_frame', 
                  'pillow', 'postcard', 'see_saw', 'sink', 'sleeping_bag', 'stairs', 'stove', 'streetlight', 
                  'suitcase', 'swing_set', 'table', 'teapot', 'toilet', 'toothbrush', 'toothpaste', 'umbrella',
                  'vase', 'wine_glass'],
    'Mammal': ['bat', 'bear', 'camel', 'cat', 'cow', 'dog', 'dolphin', 'elephant', 'giraffe', 'hedgehog', 'horse', 
               'kangaroo', 'lion', 'monkey', 'mouse', 'panda', 'pig', 'rabbit', 'raccoon', 'rhinoceros', 'sheep', 
               'squirrel', 'tiger', 'whale', 'zebra'],
    'Tool': ['anvil', 'axe', 'bandage', 'basket', 'boomerang', 'bottlecap', 'broom', 'bucket', 'compass', 'drill', 
             'dumbbell', 'hammer', 'key', 'nail', 'paint_can', 'passport', 'pliers', 'rake', 'rifle', 'saw', 
             'screwdriver', 'shovel', 'skateboard', 'stethoscope', 'stitches', 'sword', 'syringe', 'wheel'],
    'Cloth': ['belt', 'bowtie', 'bracelet', 'camouflage', 'crown', 'diamond', 'eyeglasses', 'flip_flops', 'hat', 
              'helmet', 'jacket', 'lipstick', 'necklace', 'pants', 'purse', 'rollerskates', 'shoe', 'shorts', 
              'sock', 'sweater', 't-shirt', 'underwear', 'wristwatch'],
    'Electricity': ['calculator', 'camera', 'cell_phone', 'computer', 'cooler', 'dishwasher', 'fan', 'flashlight', 
                    'headphones', 'keyboard', 'laptop', 'light_bulb', 'megaphone', 'microphone', 'microwave', 'oven', 
                    'power_outlet', 'radio', 'remote_control', 'spreadsheet', 'stereo', 'telephone', 'television', 
                    'toaster', 'washing_machine'],
    'Building': ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'barn', 'bridge', 'castle', 'church', 'diving_board', 'garden', 
                 'garden_hose', 'golf_club', 'hospital', 'house', 'jail', 'lighthouse', 'pond', 'pool', 'skyscraper', 
                 'square', 'tent', 'waterslide', 'windmill'],
    'Office': ['alarm_clock', 'backpack', 'bandage', 'binoculars', 'book', 'candle', 'calendar', 'clock', 'coffee_cup', 'crayon', 
               'cup', 'envelope', 'eraser', 'map', 'marker', 'mug', 'nail', 'paintbrush', 'paper_clip', 'pencil', 'scissors'],
    'Human Body': ['arm', 'beard', 'brain', 'ear', 'elbow', 'eye', 'face', 'finger', 'foot', 'goatee', 'hand', 'knee', 
                   'leg', 'moustache', 'mouth', 'nose', 'skull', 'smiley_face', 'toe', 'tooth'],
    'Road_Transportation': ['ambulance', 'bicycle', 'bulldozer', 'bus', 'car', 'firetruck', 'motorbike', 'pickup_truck', 
                            'police_car', 'roller_coaster', 'school_bus', 'tractor', 'train', 'truck', 'van'],
    'Food': ['birthday_cake', 'bread', 'cake', 'cookie', 'donut', 'hamburger', 'hot_dog', 'ice_cream', 'lollipop', 'peanut', 
             'pizza', 'popsicle', 'sandwich', 'steak'],
    'Nature': ['beach', 'cloud', 'hurricane', 'lightning', 'moon', 'mountain', 'ocean', 'rain', 'rainbow', 'river', 
               'snowflake', 'star', 'sun', 'tornado'],
    'Cold_Blooded': ['crab', 'crocodile', 'fish', 'frog', 'lobster', 'octopus', 'scorpion', 'sea_turtle', 'shark', 
                     'snail', 'snake', 'spider'],
    'Music': ['cello', 'clarinet', 'drums', 'guitar', 'harp', 'piano', 'saxophone', 'trombone', 'trumpet', 'violin'],
    'Fruit': ['apple', 'banana', 'blackberry', 'blueberry', 'grapes', 'pear', 'pineapple', 'strawberry', 'watermelon'],
    'Sport': ['baseball', 'baseball_bat', 'basketball', 'flying_saucer', 'hockey_puck', 'hockey_stick', 'snorkel', 
              'soccer_ball', 'tennis_racquet', 'yoga'],
    'Tree': ['bush', 'cactus', 'flower', 'grass', 'house_plant', 'leaf', 'palm_tree', 'tree'],
    'Bird': ['bird', 'duck', 'flamingo', 'owl', 'parrot', 'penguin', 'swan'],
    'Vegetable': ['asparagus', 'broccoli', 'carrot', 'mushroom', 'onion', 'peas', 'potato', 'string_bean'],
    'Shape': ['circle', 'hexagon', 'line', 'octagon', 'squiggle', 'triangle', 'zigzag'],
    'Kitchen': ['fork', 'frying_pan', 'hourglass', 'knife', 'lighter', 'matches', 'spoon', 'wine_bottle'],
    'Water_Transportation': ['aircraft_carrier', 'canoe', 'cruise_ship', 'sailboat', 'speedboat', 'submarine'],
    'Sky_Transportation': ['airplane', 'helicopter', 'hot_air_balloon', 'parachute'],
    'Insect': ['ant', 'bee', 'butterfly', 'mosquito'],
    'Others': ['The_Mona_Lisa', 'angel', 'animal_migration', 'campfire', 'cannon', 'dragon', 'feather', 'fire_hydrant', 
               'mermaid', 'snowman', 'stop_sign', 'teddy-bear', 'traffic_light']
}

MAPPING = {key: i for i, key in enumerate(DIVISIONS.keys())}

def read_text(fpath):
    with open(fpath) as f:
        content = f.readlines()
    return content

def parse_paths(fpath):
    content = read_text(fpath)
    paths = []
    categories = []
    for line in content:
        paths.append(line.split()[0])
        categories.append(paths[-1].split('/')[1])
    return paths, categories

class DomainNet_Dataset(Dataset):
    def __init__(self, directory, domain, divs=None, datatype='train', transform=None):
        self.directory = directory
        self.domain = domain
        self.divs = divs if divs is not None else []
        self.datatype = datatype
        self.transform = transform
        
        new_divisions = {item: division for division, items in DIVISIONS.items() for item in items}
        
        file_name = f"{domain}_{datatype}.txt"
        paths, categories = parse_paths(os.path.join(directory, file_name))
        divisions = [new_divisions[cat] for cat in categories]
        
        df = pd.DataFrame({'cat': categories, 'div': divisions, 'split': [datatype] * len(paths)}, index=paths)
        if self.divs:
            df = df[df['div'].isin(self.divs)]
        label_counts = df['div'].value_counts()
        min_count = label_counts.min()
        balanced_df = pd.DataFrame()
        for label in df['div'].unique():
            label_df = df[df['div'] == label]
            resampled_df = label_df.sample(n=min_count, replace=False, random_state=123)
            balanced_df = pd.concat([balanced_df, resampled_df])

        self.images = balanced_df.index.tolist()
        self.labels = balanced_df['div'].tolist()        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        label = self._get_label(idx)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, idx):
        if self.divs:
            return {key: i for i, key in enumerate(self.divs)}.get(self.labels[idx], -1)
        return MAPPING.get(self.labels[idx], -1)

def domainNet_loaders(directory='/home/ubuntu/DA_CIC/data/domainNet', divs=None, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # values from ImageNet dataset
    ])    

    dataloaders = []
    for d in DOMAINS:
        train_dataset = DomainNet_Dataset(directory, d, divs=divs, datatype='train', transform=transform)
        test_dataset = DomainNet_Dataset(directory, d, divs=divs, datatype='test', transform=transform)
        
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dataloaders.append(combined_loader)

    return dataloaders
