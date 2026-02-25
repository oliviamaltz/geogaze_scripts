import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
#this is added to read the bbox csv
import csv

import sys
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

###Adding the sys.path to the local cornet repo to access cornet (run file is outside the repo currently)
CORN_NET_REPO = Path("/zpool/vladlab/active_drive/omaltz/git_repos/CORnet")
sys.path.insert(0, str(CORN_NET_REPO))

import cornet
print("Imported cornet from:", cornet.__file__)


from PIL import Image
Image.warnings.simplefilter('ignore')

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Training')

#NOT USING IMAGENET FOLDERS
# parser.add_argument('--data_path', required=True,
#                     help='path to ImageNet folder that contains train and val folders')

#NEW COMMANDLINE ARGUMENTS FOR OPENIMAGE IMAGES AND BOUNDING BOXES
#argument for folder with training images with OpenImage naming convention 
parser.add_argument('--train_images', required=True,
                    help='path to folder of training images (ImageID.jpg)')
#argument for folder with training bounding boxes, one line per bbox annotation plus and LabelNames
parser.add_argument('--train_csv', required=True,
                    help='path to training bbox CSV (columns include ImageID, LabelName, XMin, ...)')

#argument for folder with validation images with OpenImage naming convention 
parser.add_argument('--val_images', required=True,
                    help='path to folder of validation images (ImageID.jpg)')
#argument for folder with validation images with OpenImage naming convention 
parser.add_argument('--val_csv', required=True,
                    help='path to validation bbox CSV')

#number of classes in the final output layer (v3 training set has 440 classes so you want an output layer of length 40)
parser.add_argument('--num_classes', default=440, type=int,
                    help='number of categories / output logits')


parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z',
                    help='which model to train')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

##BUILING THE FINAL (440) CLASS OUTPUT LAYER  whic is literally taking a LabelName from the bbox csv and mapping it to an index (0, 439)
def build_label_map_from_csv(csv_path, expected_classes=440):
    """
    Reads a bbox CSV and returns a dict: LabelName (str) -> class_index (int).
    Uses sorted unique labels so the mapping is deterministic, so it will be the same mapping for every run
    Ex. [('/m/011k07', 0), ('/m/012074', 1), ('/m/0120dh', 2), ('/m/01226z', 3), ('/m/012n7d', 4)]
    """
    #create empty set 
    labels = set()

    #read each csv rows as dictionarieskeyed by column name, groaps the LabelName value and ads it to the set (no diplicates in set)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["LabelName"])
    #alphabetizes it and checks that the number of unique labels names matches the number of expected classes (440)
    labels = sorted(labels)
    if expected_classes is not None and len(labels) != expected_classes:
        print(f"WARNING: found {len(labels)} unique LabelName values in {csv_path} "
              f"(expected {expected_classes}).")
        
    #gives each LabelName an idx  and is the mapping that will be used to fil last linear layer 
    return {lab: i for i, lab in enumerate(labels)}

FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)

####changed for last linear layer 
def get_model(pretrained=False):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    

    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=FLAGS.times)
    else:
        model = model(pretrained=pretrained, map_location=map_location)

    try:
        m = model.module
    except:
        m = model
    m.decoder.linear = nn.Linear(512, FLAGS.num_classes) #Replace the last linear layer with a layer of length num_classes (440)
    print("Decoder out_features:", m.decoder.linear.out_features)


    if FLAGS.ngpus == 0 and hasattr(model, "module"):
        model = model.module  # remove DataParallel
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.1,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=1,  # how often save model weigths (I want to save every epoch)
          save_model_secs=60 * 10  # how often save model (in sec)
          ):

    #this is adding the map produced from reading the bbox csv
    label_to_idx = build_label_map_from_csv(FLAGS.train_csv, expected_classes=FLAGS.num_classes)
    print("Example label mapping:", list(label_to_idx.items())[:5])

    ##ADDED FOR DEBUGGING
    ds = MultiLabelBBoxCSVDataset(FLAGS.train_images, FLAGS.train_csv, label_to_idx, transform=None)
    print("Dataset size:", len(ds)) #should be 393955
    x, y = ds[0]
    print("First image size:", x.size if hasattr(x, "size") else type(x)) 
    print("Target shape:", y.shape) #should be a 440 vector for the class number
    print("Num positives in first target:", int(y.sum().item())) #bounding boxes found in first image 

    model = get_model(pretrained=False)
    trainer = ImageNetTrain(model, label_to_idx)
    validator = ImageNetVal(model,label_to_idx)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch'] + 1
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()
    
    ###save best checkpoint
    best_val_loss = float('inf')
    best_epoch = -1

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(start_epoch, FLAGS.epochs + 1, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    val_record = validator()
                    results[validator.name] = val_record

                    # Save best checkpoint by val loss (lower is better)
                    if FLAGS.output_path is not None and ('loss' in val_record):
                        if val_record['loss'] < best_val_loss:
                            best_val_loss = val_record['loss']
                            best_epoch = epoch

                            best_ckpt = {
                                'flags': FLAGS.__dict__.copy(),
                                'epoch': epoch,
                                'best_val_loss': best_val_loss,
                                'best_epoch': best_epoch,
                                'state_dict': model.state_dict(),
                                'optimizer': trainer.optimizer.state_dict(),
                            }
                            torch.save(best_ckpt, os.path.join(FLAGS.output_path, 'best_checkpoint.pth.tar'))
                            print(f"[BEST] Saved best checkpoint at epoch={epoch}, val_loss={best_val_loss:.6f}")

                    trainer.model.train()


            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()


def test(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(np.reshape(output, (len(output), -1)))

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    with torch.no_grad():
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {FLAGS.data_path}')
        for fname in tqdm.tqdm(fnames):
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)

    if FLAGS.output_path is not None:
        fname = f'CORnet-{FLAGS.model}_{layer}_{sublayer}_feats.npy'
        np.save(os.path.join(FLAGS.output_path, fname), model_feats)

###CREATES A DATASET CLASS FROM BOUNDINGBOX CSV FOR EACH IMAGE
class MultiLabelBBoxCSVDataset(Dataset):
    """
    CSV -> (image, multi-hot target) dataset.
    Assumes image files are exactly: {ImageID}.jpg
    so tunring the csv with bounidn boxes into something pytorch can understand 
    """

    def __init__(self, images_root, csv_path, label_to_idx, transform=None):
        ###sets parameters to whatthey need to point towards 
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.label_to_idx = label_to_idx
        self.num_classes = len(label_to_idx) #440
        self.transform = transform #transform in the image processing pipeline 

        # 1) Build mapping: ImageID -> set of class indices present in that image
        #initialize a dictionary where each key gets an empty set (duplicates ignored so if there are two horse bounding boxes, it says horse is present)
        img_to_labels = defaultdict(set)
        #open csv to read image nema and label 
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageID"]
                label = row["LabelName"]

                # Only keep labels known in label_to_idx
                #when it reads the label string it converts it to the approcreate index  and adds that class index to an images set of labels
                #so fi image "0007" has bb label x, and x maps to index i index_to_labels["0007"] = {i}
                if label in self.label_to_idx:
                    img_to_labels[image_id].add(self.label_to_idx[label])

        # 2) Keep only images that actually exist in folder
        #build sample list self-items where each item is a path to an image and its set of indecies corrosponding to labels 
        #skips image that are missing BUT NO IMAGES SHOULD BE MISSING 
        self.items = []
        missing_count = 0
        for image_id, idx_set in img_to_labels.items():
            img_path = self.images_root / f"{image_id}.jpg"
            if img_path.exists():
                self.items.append((img_path, idx_set))
            else:
                missing_count += 1
        #if no bboxes match images         
        if len(self.items) == 0:
            raise RuntimeError(
                f"No matched images found. Expected files like {self.images_root}/<ImageID>.jpg"
            )
        
        #if there are some missing images 
        if missing_count > 0:
            print(f"Warning: {missing_count} ImageIDs in CSV had no matching .jpg file in {self.images_root}")

    #number of usable images 
    def __len__(self):
        return len(self.items)
    
    #retruns one image path and its index set
    def __getitem__(self, idx):
        img_path, idx_set = self.items[idx]

        #image transformation done for ImageNet images 
        # 3) Load image
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)

        #build empyt linear vector of size 440 and id label index is present makes it "1" at that index 
        # 4) Build multi-hot target vector
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(idx_set) > 0:
            y[list(idx_set)] = 1.0

        return im, y

class ImageNetTrain(object):

    def __init__(self, model,label_to_idx):
        self.name = 'train'
        self.model = model
        self.label_to_idx = label_to_idx
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        #use multi-label classification
        self.loss = nn.BCEWithLogitsLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    #NEW DATA LOADER
    # def data(self):
    #     dataset = torchvision.datasets.ImageFolder(
    #         os.path.join(FLAGS.data_path, 'train'),
    #         torchvision.transforms.Compose([
    #             torchvision.transforms.RandomResizedCrop(224),
    #             torchvision.transforms.RandomHorizontalFlip(),
    #             torchvision.transforms.ToTensor(),
    #             normalize,
    #         ]))
    #     data_loader = torch.utils.data.DataLoader(dataset,
    #                                               batch_size=FLAGS.batch_size,
    #                                               shuffle=True,
    #                                               num_workers=FLAGS.workers,
    #                                               pin_memory=True)
    #     return data_loader

    #NEW CODE FOR MULTICLASS LABEL TRAINING 
    ###need this for custom dataset 
    ### at this point dataset[i]= (image, maultiset vector) where vector has presence of objects 
    def data(self):
        dataset = MultiLabelBBoxCSVDataset(
            images_root=FLAGS.train_images,
            csv_path=FLAGS.train_csv,
            label_to_idx=self.label_to_idx,
            #what cornet exects for images becuase imagenet 
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        )
        #wrapper for datset so it can be a batch iterator 
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.workers,
            pin_memory=(FLAGS.ngpus > 0),
        )
        ###returning iterator for each training loop 
        ###
        return data_loader


    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        # record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        # record['top1'] /= len(output)
        # record['top5'] /= len(output)
        record['learning_rate'] = self.lr.get_last_lr()[0]


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class ImageNetVal(object):

    def __init__(self, model,label_to_idx):
        self.name = 'val'
        self.model = model
        self.label_to_idx = label_to_idx
        self.data_loader = self.data()
        ###change accumulating total loss then dividing by dataset size
        #using BCE for multi-label classification, becuase cross empty is for single class 
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    # def data(self):
    #     dataset = torchvision.datasets.ImageFolder(
    #         os.path.join(FLAGS.data_path, 'val_in_folders'),
    #         torchvision.transforms.Compose([
    #             torchvision.transforms.Resize(256),
    #             torchvision.transforms.CenterCrop(224),
    #             torchvision.transforms.ToTensor(),
    #             normalize,
    #         ]))
    #     data_loader = torch.utils.data.DataLoader(dataset,
    #                                               batch_size=FLAGS.batch_size,
    #                                               shuffle=False,
    #                                               num_workers=FLAGS.workers,
    #                                               pin_memory=True)

    #     return data_loader

    ####NEW DATALOADER FOR THE 440 linear layer and OpenImage Validation set:

    def data(self):
        dataset = MultiLabelBBoxCSVDataset(
            images_root=FLAGS.val_images,
            csv_path=FLAGS.val_csv,
            label_to_idx=self.label_to_idx,
            #not randomized transformation becuase this is a validation set 
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.workers,
            pin_memory=(FLAGS.ngpus > 0),
        )
        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0.0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    inp = inp.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                # p1, p5 = accuracy(output, target, topk=(1, 5))
        #         record['top1'] += p1
        #         record['top5'] += p5

        # for key in record:
        #     record[key] /= len(self.data_loader.dataset.samples)
        record['loss'] /= len(self.data_loader.dataset)
        record['dur'] = (time.time() - start) / max(len(self.data_loader), 1)


        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
