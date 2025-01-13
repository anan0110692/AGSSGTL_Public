# %%
##################### Import Libraries#############################
#! pip install -r /GeoAI/users/anan/experiments/cleanDA/requirements.txt
#from unet4 import UNet
global Update_u
import dill
# sys.path.append("../")
from utili.misc.misc import Xavi_init_weights,get_param,set_parameter_requires_grad
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
from collections import OrderedDict
import os
from os import path
import numpy as np
import random
import matplotlib
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from scipy import io as sio
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import math
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from torchsummary import summary
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
# from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import copy
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
import math
import time
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
from scipy import io as sio
import scipy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as dataf
import torch
import matplotlib
import random
import numpy as np
from os import path
import os
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../")
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
# from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import matplotlib.markers as mmarkers
import matplotlib.lines as mlines
##################### Import Libraries#############################
#! pip install -r /GeoAI/users/anan/experiments/cleanDA/requirements.txt
#from unet4 import UNet
import dill
# sys.path.append("../")
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
from collections import OrderedDict
import os
from os import path
import numpy as np
import random
import matplotlib
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from scipy import io as sio
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import math
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from torchsummary import summary
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
# from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import copy
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
import math
import time
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
from scipy import io as sio
import scipy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as dataf
import torch
import matplotlib
import random
import numpy as np
from os import path
import os
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
sys.path.append("../")
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
# from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import pickle
import lightning  as L

from pytorch_lightning import seed_everything
from sklearn.metrics import jaccard_score
from matplotlib.colors import hex2color
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Normalize
import torchmetrics
import torchgan
import skimage



# %%
# source_dataset_seed=41
# source_dataset_seed_list=[41,50,61]
# target_dataset_seed=41
# target_dataset_seed_list=[41,50,61]
# models_initilization_seed=100

# %% 


def mapping_mask(mask):
    mapped_mask=np.ones_like(mask)*-100   
    unique_classes=np.unique(mask)
    mapped_classes=np.zeros((unique_classes.max().astype(np.int64)+1,))
    mapped_classes[unique_classes[unique_classes!=-100].astype(np.int64)]=np.arange(1,unique_classes[unique_classes!=-100].size+1)
    mapped_mask[mask!=-100]=mapped_classes[mask[mask!=-100].astype(np.int64)]
    
    return mapped_mask

# %%
Dataset_name='C2Seg-AB'
def select_well_represented_common_classes(split=True):

    Source_data = sio.loadmat('Dataset/Datafiles/Augsburg_resampled.mat')
    Source_cube=Source_data['HSI']
    
    Source_cube = Source_cube.astype(np.float32)

    ####Band Normalization#########################
    # for i in range(Source_cube.shape[-1]):
    #     _range = np.max(Source_cube[:, :, i]) - np.min(Source_cube[:, :, i])
    #     Source_cube[:, :, i] = (Source_cube[:, :, i] - np.min(Source_cube[:, :, i])) / _range




    ##################################################################3
    Source_label =  Source_data['label']
    Source_label = Source_label.astype(np.float32)
    Source_DATA_dic={'raw':Source_cube,'label':Source_label}



    Target_data = sio.loadmat('Dataset/Datafiles/Berlin_resampled.mat')
    Target_cube=Target_data['HSI']
    Target_cube = Target_cube.astype(np.float32)
     ####Band Normalization#########################
    # for i in range(Target_cube.shape[-1]):
    #     _range = np.max(Target_cube[:, :, i]) - np.min(Target_cube[:, :, i])
    #     Target_cube[:, :, i] = (Target_cube[:, :, i] - np.min(Target_cube[:, :, i])) / _range




    ##################################################################3
   
    
   
    Target_label =  Target_data['label']
    Target_label = Target_label.astype(np.float32)
    Target_DATA_dic={'raw':Target_cube,'label':Target_label}

   
   

    Source_unique_classes,Source_histo= np.unique(Source_DATA_dic['label'],return_counts=True)
    Source_mask= Source_DATA_dic['label'].astype(np.int64)
    Target_mask= Target_DATA_dic['label'].astype(np.int64)
    Target_unique_classes,Target_histo=  np.unique(Target_DATA_dic['label'],return_counts=True)
    common_classes,S_common_idx,T_common_idx= np.intersect1d(Source_unique_classes,Target_unique_classes,assume_unique=True,return_indices=True)
    common_from_S_histo= Source_histo[S_common_idx]
    common_from_S=     Source_unique_classes[S_common_idx]
    Underrepresend_classes=  common_from_S[common_from_S_histo<np.quantile(common_from_S_histo,.3) ]
    if  not split:
        Source_mask[np.isin(Source_mask.copy(),Underrepresend_classes)]=-100
        Target_mask[np.isin(Target_mask.copy(),Underrepresend_classes)]=-100
        return Source_DATA_dic['Mosaic_image'],Source_mask, Target_DATA_dic['Mosaic_image'],Target_mask
    else:
        common_classes,S_common_idx,T_common_idx= np.intersect1d(Source_unique_classes,Target_unique_classes,assume_unique=True,return_indices=True)

       
        
        return Source_DATA_dic['raw'],Source_DATA_dic['label'], Target_DATA_dic['raw'], Target_DATA_dic['label'] , common_classes.size

# %%
select_well_represented_common_classes(split=True)[-1]

# %%
def clnum():
    
    return select_well_represented_common_classes(split=True)[4]

# %%
def Source_data_generator(batch_size=1,num_workers=16):
    Source_dataset=select_well_represented_common_classes(split=True)
    HU_cube = Source_dataset[0][None].astype(np.float32)/(pow(2,14)-1)
    
    HU_cube=torch.from_numpy(HU_cube[:,:,:,:]).permute((0,3,1,2))
    HU_cube_original=HU_cube[0].unsqueeze(0)
    HU_cub_transformation=HU_cube[1:]
    # HU_cube=torch.from_numpy(HU_cube[:,:,:,:,:]).permute((0,1,4,2,3))
    ch_num=HU_cube.shape[1]
    TR_label=Source_dataset[1]
    
    TR_label = TR_label.astype(np.float32)
    TR_label[TR_label==0]= -100
    original_unique_classes=np.unique(TR_label)
    TR_label= mapping_mask(TR_label)
    TR_label=np.squeeze(TR_label)
    cl_num = int(np.max(TR_label))
    source_dataset_seed=41
    seed_everything(source_dataset_seed,workers=True)
    random_state = check_random_state(source_dataset_seed)
    ######################################################################################33
    All_Train_labels_idx = np.array([])
    All_Val_labels_idx = np.array([])
    classes = np.unique(  TR_label.flatten())
    classes = classes[classes != -100]
    #classes = classes[classes == 1]
    for i in classes:
        class_indices = np.argwhere(TR_label == i)
        class_train_idx, class_val_idx, _, _ = train_test_split(class_indices, np.zeros(
            (class_indices.shape[0],)), train_size=.8, random_state=42)
        if All_Train_labels_idx.size == 0:
            All_Train_labels_idx = class_train_idx
        else:
            All_Train_labels_idx = np.concatenate(
                (All_Train_labels_idx, class_train_idx))

        if All_Val_labels_idx.size == 0:
            All_Val_labels_idx = class_val_idx
        else:
            All_Val_labels_idx = np.concatenate(
                (All_Val_labels_idx, class_val_idx))


   

    Image_rotation_transform = transforms.RandomRotation(degrees=180)
    Mask_rotation_trasnform= transforms.RandomRotation(degrees=180,fill=-100)
    TR_labell= np.ones_like(TR_label)*-100
    
    VA_label = np.ones_like(TR_label)*-100
    VA_label[All_Val_labels_idx[:, 0], All_Val_labels_idx[:, 1]] = TR_label[All_Val_labels_idx[:, 0], All_Val_labels_idx[:, 1]]
    TR_labell[All_Train_labels_idx[:, 0], All_Train_labels_idx[:, 1]] = TR_label[All_Train_labels_idx[:, 0], All_Train_labels_idx[:, 1]]
    TR_labell=mapping_mask(TR_labell)
    TR_labell = torch.from_numpy(TR_labell)
    TR_labell = TR_labell[None, :, :]
   
    VA_label=mapping_mask(VA_label)
    VA_label = torch.from_numpy(VA_label)
    VA_label = VA_label[None, :, :]
   
    
    dataset_T = dataf.TensorDataset(HU_cube, TR_labell)
    train_loader = dataf.DataLoader(dataset_T, batch_size=batch_size,num_workers=num_workers)
   
    dataset_V = dataf.TensorDataset(HU_cube, VA_label)
    Validation_loader = dataf.DataLoader(dataset_V, batch_size=batch_size,num_workers=num_workers)
    return train_loader, Validation_loader,original_unique_classes,ch_num

# %%
def Target_data_genrator(batch_size=1, num_workers=16,Num_of_Samples=None):
   
    
    Source_dataset=select_well_represented_common_classes(split=True)

    HU_cube = Source_dataset[2][None].astype(np.float32)
    ch_num=HU_cube.shape[-1]
    HU_cube=torch.from_numpy(HU_cube[:,:,:,:]).permute((0,3,1,2))
    TR_label=Source_dataset[3]
    
    TR_label = TR_label.astype(np.float32)
    TR_label[TR_label==0]= -100
    original_unique_classes=np.unique(TR_label)
    TR_label= mapping_mask(TR_label)
    TR_label=np.squeeze(TR_label)
    cl_num = int(np.max(TR_label))

    target_dataset_seed=41
    seed_everything(target_dataset_seed,workers=True)
    random_state = check_random_state(target_dataset_seed)
    #######################################################################################
    All_Train_labels_unlabeled_idx = np.array([])
    All_Train_labels_labeled_idx=np.array([])
    All_Val_labels_labeled_idx=np.array([])
    All_test_labels_idx = np.array([])
    with open('Dataset/Datafiles/Fixed_Test_dataloader/Fixed_Germany_wishpers_MOSAIC_All_Normalization_Type_1_Berlin_test_labels.pkl', 'rb') as file:
           All_test_labels_idx= dill.load(file)
    All_test_labels_idx_dict={tuple(row): i for i, row in enumerate(All_test_labels_idx)}
    classes = np.unique(  TR_label.flatten())
    classes = classes[classes != -100]
   # classes = classes[(classes == 13)]
    for i in classes:
        class_indices = np.argwhere( TR_label  == i)
        selected_class_indices_idx= [ All_test_labels_idx_dict.get(tuple(row))==None for i,row in enumerate(class_indices)]
        class_indices=class_indices[selected_class_indices_idx]
        if Num_of_Samples is not None:
           
           class_indices= train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size= Num_of_Samples, random_state=42)[0]



        class_train_unlabeled_idx=class_indices
        class_train_idx=class_indices
       
        
        # class_train_idx, class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), test_size=.3, random_state=42)
       
           

        
        class_train_unlabeled_idx_,class_train_labeled_idx,_,_= train_test_split(class_train_idx, np.zeros(
            (class_train_idx.shape[0],)), test_size=3, random_state=42)
        
        class_train_labeled_idx,class_val_labeled_idx,_,_= train_test_split(class_train_labeled_idx, np.zeros(
            (class_train_labeled_idx.shape[0],)), train_size=.8, random_state=42)
        
        if All_Train_labels_unlabeled_idx.size == 0:
            All_Train_labels_unlabeled_idx = class_train_unlabeled_idx
        else:
            All_Train_labels_unlabeled_idx = np.concatenate(
                (All_Train_labels_unlabeled_idx, class_train_unlabeled_idx))
       
        if  All_Train_labels_labeled_idx.size == 0:
            All_Train_labels_labeled_idx= class_train_labeled_idx
        else:
          All_Train_labels_labeled_idx = np.concatenate(
                (All_Train_labels_labeled_idx, class_train_labeled_idx))
       #######################################################################
        if  All_Val_labels_labeled_idx.size == 0:
           All_Val_labels_labeled_idx= class_val_labeled_idx
        else:
           All_Val_labels_labeled_idx = np.concatenate(
                ( All_Val_labels_labeled_idx, class_val_labeled_idx))


       
        # if All_test_labels_idx.size == 0:
        #     All_test_labels_idx = class_test_idx
        # else:
        #     All_test_labels_idx = np.concatenate(
        #         (All_test_labels_idx, class_test_idx))
     
    TE_label = np.ones_like(TR_label)*-100
    TR_labell=  np.ones_like(TR_label)*-100
    TR_label_labeled=  np.ones_like(TR_label)*-100
    VA_label_labeled=  np.ones_like(TR_label)*-100
    TE_label [All_test_labels_idx[:, 0], All_test_labels_idx[:, 1]] =  TR_label[All_test_labels_idx[:, 0], All_test_labels_idx[:, 1]]
    TR_labell[All_Train_labels_unlabeled_idx[:, 0], All_Train_labels_unlabeled_idx[:, 1]] =  TR_label[All_Train_labels_unlabeled_idx[:, 0], All_Train_labels_unlabeled_idx[:, 1]]
    TR_label_labeled[All_Train_labels_labeled_idx[:, 0], All_Train_labels_labeled_idx[:, 1]] =  TR_label[All_Train_labels_labeled_idx[:, 0], All_Train_labels_labeled_idx[:, 1]]
    VA_label_labeled[All_Val_labels_labeled_idx[:, 0], All_Val_labels_labeled_idx[:, 1]] =  TR_label[All_Val_labels_labeled_idx[:, 0], All_Val_labels_labeled_idx[:, 1]]

   

   
  
   
    TR_labell  = torch.from_numpy(TR_labell )
    TR_labell  = TR_labell [None, :, :]
    TE_label = torch.from_numpy(TE_label)
    TE_label = TE_label[None, :, :]
    TR_label_labeled  = torch.from_numpy(TR_label_labeled)
    TR_label_labeled  = TR_label_labeled [None, :, :]
    VA_label_labeled  = torch.from_numpy(VA_label_labeled)
    VA_label_labeled  = VA_label_labeled [None, :, :]
   
    dataset_T = dataf.TensorDataset(HU_cube,TR_labell )
    train_loader = dataf.DataLoader(dataset_T, batch_size=batch_size,num_workers=num_workers)
    dataset_TE = dataf.TensorDataset(HU_cube, TE_label)
    TEST_loader = dataf.DataLoader(dataset_TE,  batch_size=batch_size,num_workers=num_workers)
    dataset_train_labeled = dataf.TensorDataset(HU_cube,TR_label_labeled )
    train_labeled_loader= dataf.DataLoader(  dataset_train_labeled,  batch_size=batch_size,num_workers=num_workers)
    dataset_val_labeled = dataf.TensorDataset(HU_cube,VA_label_labeled )
    val_labeled_loader= dataf.DataLoader(  dataset_val_labeled,  batch_size=batch_size,num_workers=num_workers)
    train_unlabeled_loader=train_loader




        
    

   


 
 
 
 
 
 
 
    return train_unlabeled_loader, TEST_loader,   train_labeled_loader, val_labeled_loader, TR_label
    # return train_unlabeled_loader, TEST_loader,   train_labeled_loader, val_labeled_loader,All_test_labels_idx

# %%
def Color_map_generator():
    lut_colors = {
   
    1: "#000000",
                 2: '#00FFFF',
                3: '#FFFFFF',
                 4: '#FF0000',
                 5: '#DDA0DD',
                 6: '#9400D3',
                 7: '#FF00FF',
                 8: '#FFFF00',
                 9: '#CD853F',
                 10: '#BDB76B',
                 11: '#00FF00',
                 12: '#9ACD32',
                 13: '#8B4513',
                 14: '#483D8B'

 
  
   }

    
   
    Color_map_array=[list(int(i * 255) for i in hex2color(v)) for k, v in lut_colors.items()]
    Color_map_array=np.array(Color_map_array)
   
    UH_map = ListedColormap(Color_map_array.astype(np.float32)/256.0)
    
    return UH_map

# %%
class Source_Datamodule(L.LightningDataModule):
    
    def __init__(self, batch_size=3, num_workers=16  ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        Source_data_generator_Output_tuple=Source_data_generator(self.batch_size, self.num_workers)
        # Target_data_generator_output_tuple=Target_data_genrator(self.batch_size, self.num_workers)
        self.train_loader=Source_data_generator_Output_tuple[0]
        # self.Target_train_loader=Target_data_generator_output_tuple[2]
        # self.Target_val_loader=Target_data_generator_output_tuple[3]
        self.val_loader=Source_data_generator_Output_tuple[1]
        self.original_unique_classes=Source_data_generator_Output_tuple[2]
        

        self.classes_labels_array= np.array( [' ',"Surface water", "Street", "Urban Fabric", "Industrial, commercial and transport", "Mine, dump, and construction sites", "Artificial, vegetated areas", "Arable Land", "Permanent Crops", "Pastures", "Forests", "Shrub", "Open spaces with no vegetation", "Inland wetlands" ,'ignored']
)
        self.color_map=Color_map_generator()
        self.ch_num=Source_data_generator_Output_tuple[3]
        self.cl_num=self.original_unique_classes[self.original_unique_classes!=-100].size
    def setup(self, stage=None):
       pass
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader

# %%
class Target_Datamodule(L.LightningDataModule):
    
    def __init__(self, batch_size=1, num_workers=16 ,Num_of_Samples=None ):
        super().__init__()
        with open('Dataset/Datafiles/Fixed_Test_dataloader/Fixed_Germany_wishpers_MOSAIC_Normalization_Type_1_Berlin_test_dataloader.pkl', 'rb') as file:
            Target_test_Dataloader= dill.load(file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        Target_data_generator_Output_tuple=Target_data_genrator(self.batch_size, self.num_workers,Num_of_Samples=Num_of_Samples)
        self.train_loader=Target_data_generator_Output_tuple[0]
        self.test_loader= Target_data_generator_Output_tuple[1]
        self.Full_GT_mask=  Target_data_generator_Output_tuple[-1]
    def setup(self, stage=None):
       pass
    def train_dataloader(self):
        return self.train_loader
    def test_dataloader(self):
        return self.test_loader

# %%
class DA_Datamodule(L.LightningDataModule):
        
        def __init__(self, batch_size=1, num_workers=16, Source_Datamodule_arg=None, Num_of_Samples=None  ):
            super().__init__()
            self.batch_size = batch_size
            self.num_workers = num_workers

            if Source_Datamodule_arg is None:
                _Source_Datamodule=Source_Datamodule(self.batch_size, self.num_workers)
            else:
                _Source_Datamodule=Source_Datamodule_arg
            _Target_Datamodule=Target_Datamodule(self.batch_size, self.num_workers,Num_of_Samples=Num_of_Samples)
            self.ch_num=_Source_Datamodule.ch_num
            self.Source_dataloader=_Source_Datamodule.train_dataloader()
            self.Target_dataloader=_Target_Datamodule.train_dataloader()
            self.Target_test_dataloader=_Target_Datamodule.test_dataloader()
            self.classes_labels_array=_Source_Datamodule.classes_labels_array
            self.color_map=Color_map_generator()
            self.Total_class_num= _Source_Datamodule.classes_labels_array.size
            self.cl_num=_Source_Datamodule.cl_num
            self.ch_num=_Source_Datamodule.ch_num
            self.Full_GT_mask=_Target_Datamodule.Full_GT_mask
            self.original_unique_classes=_Source_Datamodule.original_unique_classes
            self.Dataset_name=Dataset_name
        def setup(self, stage=None):
            pass
        def train_dataloader(self):
            return [self.Target_dataloader, self.Source_dataloader]
       
        def get_batch(self,batch_Datset='Target'):
            DA_loader=self.train_dataloader()
            batch_list=next(iter(DA_loader))
            if batch_Datset=='Target':
                return  next(iter(DA_loader[0]))
            else:
                return  next(iter(DA_loader[1]))





# #UNet+Spearable CONV


# #UNet+Spearable CONV


@torch.no_grad()
def find_pad(model,x):
       
        name_list=[]
      
        for name,_ in model.named_children():
            if "upconv" in name:
                name_list.append(name)
        for i in range(len(name_list)):
            temp=getattr(model,name_list[i])
            if isinstance(temp,torchgan.layers.spectralnorm.SpectralNorm2d):
                
                modell=temp.module
                if not hasattr(modell,'weight'):
                    temp_x=torch.rand((3,modell.in_channels,100,100),device=x.device)
                    temp_out=temp(temp_x)
                modell=temp.module
                setattr(modell,'output_padding',(0,0))
            else:
                setattr(temp,'output_padding',(0,0))
       
        encoders_output_shapes=[]
        decoders_output_shapes=[]
        out_sizes_list=[]
        out=x
        for name,Sub_model in model.named_children():
           
            if ~("final_layer" in name) and len(decoders_output_shapes)==0 :
                # out=Sub_model.cuda(GPU_NUM)(out)
                # out=Sub_model(out)
                ###################################################################
                if len(out_sizes_list)==0:
                        input_shape=x.shape
                else:
                        input_shape=out_sizes_list[-1]
              
                if isinstance(Sub_model,torch.nn.modules.container.Sequential):
                   out_shape=input_shape
                   for i in Sub_model:
                       if isinstance(i,torch.nn.modules.conv.Conv2d):
                           Last_conv_layer=i
              
                   out_shape=(input_shape[0], Last_conv_layer.out_channels,*input_shape[2:])
                   out_sizes_list.append(out_shape)
               
               #-------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,torch.nn.modules.pooling.MaxPool2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))
                    # D_temp= torch.tensor((input_shape[-3]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    # D=torch.floor(D_temp)

                    H_temp= torch.tensor((input_shape[-2]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    H=torch.floor(H_temp)
                    
                    W_temp= torch.tensor((input_shape[-1]+2*Padding_size[1]-  Dilation_size[1]*(Kernal_size[1]-1)-1)/ Stride_size[1]+1)
                    W=torch.floor(W_temp)

                    out_shape=(input_shape[0],input_shape[1] ,H,W)
                    out_sizes_list.append(out_shape)

                    #--------------------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,nn.ConvTranspose2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))

                    
                    # D=(input_shape[-3]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    H=(input_shape[-2]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    W=(input_shape[-1]-1)*Stride_size[1]-2*Padding_size[1]+Dilation_size[1]*( Kernal_size[1]-1)+1
                    out_shape=(input_shape[0], Sub_model.out_channels,H,W)
                    out_sizes_list.append(out_shape)
                    #---------------------------------------------------------------------------------------------
                elif  isinstance(Sub_model,nn.Conv2d):
                    out_shape=(input_shape[0], Sub_model.out_channels,*input_shape[2:])
                    out_sizes_list.append(out_shape)
                #----------------------------------------------------------------------------
                else:
                    out_shape=input_shape
                    out_sizes_list.append(out_shape)


               ######################################################################
                if "encoder" in name:
                    encoders_output_shapes.append(out_sizes_list[-1])
                elif "upconv" in name:
                    decoders_output_shapes.append(out_sizes_list[-1])
            else:
                if "upconv" in name:
                    input_shape=encoders_output_shapes[-len(decoders_output_shapes)]
                   
                    
                     
                else:
                    input_shape= out_sizes_list[-1]
                    # out=Sub_model.cuda(GPU_NUM)(out)
                    if "final" in name:
                       continue
                    
                if isinstance(Sub_model,torch.nn.modules.container.Sequential):
                   out_shape=input_shape
                   for i in Sub_model:
                       if isinstance(i,torch.nn.modules.conv.Conv2d):
                           Last_conv_layer=i
              
                   out_shape=tuple(input_shape[0], Last_conv_layer.out_channels,*input_shape[2:])
                   out_sizes_list.append(out_shape)
               
               #-------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,torch.nn.modules.pooling.MaxPool2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))
                    # D_temp= torch.tensor((input_shape[-3]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    # D=torch.floor(D_temp)

                    H_temp= torch.tensor((input_shape[-2]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    H=torch.floor(H_temp)
                    
                    W_temp= torch.tensor((input_shape[-1]+2*Padding_size[1]-  Dilation_size[1]*(Kernal_size[1]-1)-1)/ Stride_size[1]+1)
                    W=torch.floor(W_temp)

                    out_shape=tuple(input_shape[0],input_shape[1] ,H,W)
                    out_sizes_list.append(out_shape)

                    #--------------------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,nn.ConvTranspose2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))

                    
                    # D=(input_shape[-3]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    H=(input_shape[-2]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    W=(input_shape[-1]-1)*Stride_size[1]-2*Padding_size[1]+Dilation_size[1]*( Kernal_size[1]-1)+1
                    out_shape=(input_shape[0], Sub_model.out_channels,H,W)
                    out_sizes_list.append(out_shape)
                    #---------------------------------------------------------------------------------------------
                elif  isinstance(Sub_model,nn.Conv2d):
                    out_shape=(input_shape[0], Sub_model.out_channels,*input_shape[2:])
                    out_sizes_list.append(out_shape)
                #----------------------------------------------------------------------------
                else:
                    out_shape=input_shape
                    out_sizes_list.append(out_shape)
              
              
               
              
              
                if "encoder" in name:
                    encoders_output_shapes.append(out_sizes_list[-1])
                elif "upconv" in name:
                    decoders_output_shapes.append(out_sizes_list[-1])

    
        decoders_output_shapes.reverse()
        output_padding=[]

        for ii in range(len(decoders_output_shapes)):
            temp_enc_shape= encoders_output_shapes[ii]
            temp_dec_shape= decoders_output_shapes[ii]
            output_padding.append((int( temp_enc_shape[-2]-temp_dec_shape[-2]),int( temp_enc_shape[-1]-temp_dec_shape[-1])))
        output_padding.reverse()
        return output_padding
        






class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=21*2,model_type=None,input_shape=None):
        super(UNet, self).__init__()
        features = init_features
        
        # self.attention=SE_Spatial_attention(input_shape)
        # self.attention=Self_attention(input_shape)
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2=UNet._block1(features,2*features,name='enc2')
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3=UNet._block1(2*features,4*features,name='enc3')
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4=UNet._block1(4*features,8*features,name='enc4')
        self.pool4=nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder5=UNet._block1(8*features,16*features,name='enc5')
        # self.pool5=nn.MaxPool2d(kernel_size=2, stride=2)
      
        # self.upconv1 = nn.ConvTranspose2d( features * 16, 16*features, kernel_size=2, stride=2)
        # self.decoder1= ( nn.Conv2d( in_channels=32*features,   out_channels=8*features, kernel_size=3, padding=1 ))
      
        self.upconv2 = nn.ConvTranspose2d( features * 8, 8*features, kernel_size=2, stride=2,groups=features * 8)
        self.decoder2= ( nn.Conv2d( in_channels=16*features,   out_channels=4*features, kernel_size=3, padding=1 ,groups=4*features))
      
        self.upconv3 = nn.ConvTranspose2d( features * 4, 4*features, kernel_size=2, stride=2,groups=features * 4)
        self.decoder3= (nn.Conv2d(  in_channels=8*features, out_channels=2*features, kernel_size=3, padding=1,groups= 2*features ))

        self.upconv4 = nn.ConvTranspose2d( features * 2, 2*features, kernel_size=2, stride=2,groups= 2*features)
        self.decoder4= (nn.ConvTranspose2d(  in_channels=4*features, out_channels=features, kernel_size=3, padding=1,groups= features ))

        self.upconv5 = nn.ConvTranspose2d( features * 1, features, kernel_size=2, stride=2,groups= features)
        self.decoder5= (nn.Conv2d(  in_channels=2*features, out_channels=features, kernel_size=3, padding=1,groups= features ))
        input_dim_list=list(input_shape)
        input_dim_list[1]=features*1
        Input_dim_attnetion=tuple(input_dim_list)
        # self.attention=SE_Spatial_attention(Input_dim_attnetion)
        self.final_layer = nn.Conv2d(in_channels=features*1, out_channels=out_channels, kernel_size=1)
        self.use_attention=None
        self.Spectral_flag=np.array([True])
        self.Normalize=np.array([True])
        
       
    
        
    def forward(self, x,Attention_weight= False,Update_u=True,Normalize=True):
        self.Spectral_flag[0]=Update_u
        self.Normalize[0]=Normalize
        output_pad=find_pad(self,x)
        name_list=[]
        for name,_ in self.named_children():
            if "upconv" in name:
                name_list.append(name)
        for i in range(len(name_list)):
            temp_sub_model=getattr(self,name_list[i])
            if isinstance(temp_sub_model,torchgan.layers.spectralnorm.SpectralNorm2d):
                temp_sub_model=temp_sub_model.module
            setattr(temp_sub_model,'output_padding',output_pad[i])
  
      
        if self.use_attention==None:
            self.use_attention=Attention_weight
        

        
       
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3=self.encoder3(self.pool2(enc2))
        enc4=self.encoder4(self.pool3(enc3))
        bott=self.pool4(enc4)
        # enc5=self.encoder5(self.pool4(enc4))
        # bott=self.pool5(enc5)
        # dec1 = self.upconv(bott)
        # dec1 = torch.cat((dec1, enc5), dim=1)
        # dec1=self.decoder1(dec1)
        dec1=self.upconv2(bott)
        dec1=torch.cat((dec1, enc4), dim=1)
        
        dec2=self.decoder2(dec1)
        dec2=self.upconv3(dec2)
        dec2=torch.cat((dec2, enc3), dim=1)
        dec3=self.decoder3(dec2)
        dec3=self.upconv4(dec3)
        dec3=torch.cat((dec3, enc2), dim=1)
        dec4=self.decoder4(dec3)
        dec4=self.upconv5(dec4)
        dec4=torch.cat((dec4, enc1), dim=1)
        dec1=self.decoder5(dec4)
        # if Attention_weight and self.use_attention:
            
        #     dec1=self.attention(dec1)
        return (self.final_layer(dec1),(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            groups=21
                        ),
                    ),
                   (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU()),
                   (
                       name + "conv2",
                       nn.Conv2d(
                           in_channels=features,
                           out_channels=features,
                           kernel_size=3,
                           padding=1,
                           bias=True,
                           groups=features
                       ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                   (name + "relu2", nn.ReLU()),
                ]
            )
        )

    def _block1(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            groups=in_channels
                        ),
                    ),
                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU()),
                   
                ]
            )
        )


        
        


 # //////////////////// Configiration////////////////////////////////////////
cfg={"LR_PRE": 10e-4,"EPOCH_PRE" :200,'EPOCH_DA':150,'LRT':10e-5,'LRD':10e-4,"number_of_target_labeled_sampels":10}        


# //////////////////// Light_source////////////////////////////
class Light_source_train(L.LightningModule):
    def __init__(self, ch_num, cl_num,S_cnn,LR) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.source_model=S_cnn
        self.method=method
        self.automatic_optimization = False
        self.train_loss_step=[]
        self.LR=LR
        
    def training_step(self, batch, batch_idx): 
      
        Optimizer=self.optimizers()
        S_x,S_y=batch
        
        S_y = S_y.type(torch.long)
        S_y[S_y != -100]= S_y[S_y != -100]-1
        
        
        S_ypre=self.source_model(S_x)[0]
        S_loss= torch.nn.functional.cross_entropy(S_ypre,S_y)

        loss=  S_loss
        Optimizer.zero_grad()
        self.manual_backward(loss)
        Optimizer.step()
        self.train_loss_step.append(loss.item())
        self.log('train loss',loss,on_epoch=True,prog_bar=True)
        return loss            
        

     
        
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.source_model.parameters(), lr= self.LR)
   
    @torch.no_grad()
    def validatie(self):
       
      
          list_dataloaders=self.trainer.datamodule.val_dataloader()
          S_dataloader=list_dataloaders
          sum_S_loss=0
          S_list_of_batch_sizes=[]
          for S_batch in S_dataloader:
              S_x,S_y=S_batch
              S_y = S_y.type(torch.long).to(self.device)
              S_x=S_x.to(self.device)
              S_y[S_y != -100]= S_y[S_y != -100]-1
              S_ypre=self.source_model.eval()(S_x)[0]
              S_loss= torch.nn.functional.cross_entropy(S_ypre,S_y)
              sum_S_loss+=S_loss*len(S_y)
              S_list_of_batch_sizes.append(len(S_y))

       

          loss=sum_S_loss/( torch.sum(torch.tensor(S_list_of_batch_sizes)))
          self.log('Source val loss',loss,on_epoch=True,prog_bar=True)
          self.source_model.train()
          return loss

    def on_train_epoch_end(self):      
          Epoch_loss=np.mean(self.train_loss_step)
          self.train_loss_step=[]
          
          self.logger.experiment.add_scalar ('Epoch training loss',  Epoch_loss,self.current_epoch)
          self.logger.experiment.add_scalar ('Epoch validation loss',  self.validatie(),self.current_epoch)
          
         
          
       
  
   


# //////////////////// Source_train////////////////////////////////////////////////////////////////////////////////
#Source Train
def SourceTrain(Dataset,Train=True, batch_size=4,num_workers=16,Result_path=None,random_seed=None,lightgin_log_path=None,LR=None, EPOCH=None, accelerator=None, devices=None,Exp_name=None):
    Source_Datamodule=Dataset.Source_Datamodule(batch_size=batch_size,num_workers=num_workers)
    ch_num=Source_Datamodule.ch_num
    cl_num= Source_Datamodule.cl_num
   
   
        
       
 
    if random_seed is not None:
        seed_everything(random_seed,workers=True)
    Checkpoint_callback=ModelCheckpoint(dirpath=Result_path,filename='S_best_model',monitor='Source val loss')
    batch=next(iter(Source_Datamodule.train_loader))

    Source_model=UNet(ch_num,cl_num,input_shape= batch[0].shape)

    Xavi_init_weights(Source_model)

    light_source_model= Light_source_train(ch_num, cl_num,Source_model,LR=LR  )
    Tensorborad_logger=L.pytorch.loggers.tensorboard.TensorBoardLogger(name=Exp_name,save_dir=lightgin_log_path)
    
    
    if accelerator=="gpu":

        trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator,max_epochs=EPOCH,callbacks=[Checkpoint_callback], devices=devices,deterministic='warn',benchmark=False)
    else:
        trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator,max_epochs=EPOCH,deterministic='warn',benchmark=False,callbacks=[Checkpoint_callback])

    
    
    trainer.fit(model=light_source_model, datamodule=Source_Datamodule)
    return light_source_model.source_model.cpu(),Source_Datamodule, light_source_model, Checkpoint_callback.best_model_path
    
#///////////////////////////////////////////////////Attention_layer////////////////////////////////////
class SE_Spatial_attention(nn.Module):
    def __init__(self,input_shape,model='Target'):
        super(SE_Spatial_attention,self).__init__()
        if model=='Target':
            self.Liner1=nn.Linear(input_shape[1],2)
            # self.Batch1=nn.BatchNorm2d(2)
            self.Batch1=nn.BatchNorm1d(2)
            self.Liner2=nn.Linear(2,1)
            # self.Batch2=nn.BatchNorm2d(1)
            self.Batch2= nn.BatchNorm1d(1)
            self.in_channels=input_shape[1]
           
        
        else:
            self.model_type='D'
            self.Liner1=nn.Linear(input_shape[1],100)
            # self.Batch1=nn.BatchNorm2d(2)
            self.Batch1=nn.BatchNorm1d(100)
            self.Liner2=nn.Linear(100,50)
            # self.Batch2=nn.BatchNorm2d(1)
            self.Batch2= nn.BatchNorm1d(50)
            self.Liner3=nn.Linear(50,1)
            self.Batch3=nn.BatchNorm1d(1)
            
            self.in_channels=input_shape[1]
        
    def forward(self,x,get_attention_weights=False):
        if not hasattr(self, 'model_type'):
            x_copy=torch.tensor(x)
            x_copy=x_copy.permute((0,2,3,1))
            out1=self.Liner1(x_copy)
            out1=self.Batch1(out1.reshape(-1,2)).reshape(out1.shape).permute(0,3,1,2)
            out1=F.leaky_relu(out1)
            out2=self.Liner2(out1.permute(0,2,3,1))
            out2=self.Batch2(out2.reshape(-1,1)).reshape(out2.shape).permute(0,3,1,2)
            out2=F.sigmoid(out2.permute(0,2,3,1))+1
            out2=out2.squeeze(dim=-1)
            out2=out2[:,None,:,:]
            if not get_attention_weights:
                return out2*x
            else:
                return out2*x,out2
        else:
            x_copy=torch.tensor(x)
            x_copy=x_copy.permute((0,2,3,1))
            out1=self.Liner1(x_copy)
            out1=self.Batch1(out1.reshape(-1,100)).reshape(out1.shape).permute(0,3,1,2)
            out1=F.leaky_relu(out1)
            out2=self.Liner2(out1.permute(0,2,3,1))
            out2=self.Batch2(out2.reshape(-1,50)).reshape(out2.shape).permute(0,3,1,2)
            out2=F.leaky_relu(out2)
            out3=self.Liner3(out2.permute(0,2,3,1))
            out3=self.Batch3(out3.reshape(-1,1)).reshape(out3.shape).permute(0,3,1,2)
            out3=F.sigmoid(out3.permute(0,2,3,1))+1
            out3=out3.squeeze(dim=-1)
            out3=out3[:,None,:,:]
            if not get_attention_weights:
                return out3*x
            else:
                return out3*x,out3
            
#///////////////////////////////////////////////////Discriminator////////////////////////////////////  
def find_pad_light_D(model,x):
       
        name_list=[]
      
        for name,_ in model.named_children():
            if "upconv" in name:
                name_list.append(name)
        for i in range(len(name_list)):
            temp=getattr(model,name_list[i])
            if isinstance(temp,torchgan.layers.spectralnorm.SpectralNorm2d):
                
                modell=temp.module
                if not hasattr(modell,'weight'):
                    temp_x=torch.rand((3,modell.in_channels,100,100),device=x.device)
                    temp_out=temp(temp_x)
                modell=temp.module
                setattr(modell,'output_padding',(0,0))
            else:
                setattr(temp,'output_padding',(0,0))
       
        encoders_output_shapes=[]
        decoders_output_shapes=[]
        out_sizes_list=[]
        out=x
        for name,Sub_model in model.named_children():
           
            if ~("final_layer" in name) and len(decoders_output_shapes)==0 :
                # out=Sub_model.cuda(GPU_NUM)(out)
                # out=Sub_model(out)
                ###################################################################
                if len(out_sizes_list)==0:
                        input_shape=x.shape
                else:
                        input_shape=out_sizes_list[-1]
              
                if isinstance(Sub_model,torch.nn.modules.container.Sequential):
                   out_shape=input_shape
                   for i in Sub_model:
                       if isinstance(i,torch.nn.modules.conv.Conv2d):
                           Last_conv_layer=i
              
                   out_shape=(input_shape[0], Last_conv_layer.out_channels,*input_shape[2:])
                   out_sizes_list.append(out_shape)
               
               #-------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,torch.nn.modules.pooling.MaxPool2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))
                    # D_temp= torch.tensor((input_shape[-3]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    # D=torch.floor(D_temp)

                    H_temp= torch.tensor((input_shape[-2]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    H=torch.floor(H_temp)
                    
                    W_temp= torch.tensor((input_shape[-1]+2*Padding_size[1]-  Dilation_size[1]*(Kernal_size[1]-1)-1)/ Stride_size[1]+1)
                    W=torch.floor(W_temp)

                    out_shape=(input_shape[0],input_shape[1] ,H,W)
                    out_sizes_list.append(out_shape)

                    #--------------------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,nn.ConvTranspose2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))

                    
                    # D=(input_shape[-3]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    H=(input_shape[-2]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    W=(input_shape[-1]-1)*Stride_size[1]-2*Padding_size[1]+Dilation_size[1]*( Kernal_size[1]-1)+1
                    out_shape=(input_shape[0], Sub_model.out_channels,H,W)
                    out_sizes_list.append(out_shape)
                    #---------------------------------------------------------------------------------------------
                elif  isinstance(Sub_model,nn.Conv2d):
                    out_shape=(input_shape[0], Sub_model.out_channels,*input_shape[2:])
                    out_sizes_list.append(out_shape)
                #----------------------------------------------------------------------------
                else:
                    out_shape=input_shape
                    out_sizes_list.append(out_shape)


               ######################################################################
                if "encoder" in name:
                    encoders_output_shapes.append(out_sizes_list[-1])
                elif "upconv" in name:
                    decoders_output_shapes.append(out_sizes_list[-1])
            else:
                if "upconv" in name:
                    input_shape=encoders_output_shapes[-len(decoders_output_shapes)]
                   
                    
                     
                else:
                    input_shape= out_sizes_list[-1]
                    # out=Sub_model.cuda(GPU_NUM)(out)
                    if "final" in name:
                       continue
                    
                if isinstance(Sub_model,torch.nn.modules.container.Sequential):
                   out_shape=input_shape
                   for i in Sub_model:
                       if isinstance(i,torch.nn.modules.conv.Conv2d):
                           Last_conv_layer=i
              
                   out_shape=tuple(input_shape[0], Last_conv_layer.out_channels,*input_shape[2:])
                   out_sizes_list.append(out_shape)
               
               #-------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,torch.nn.modules.pooling.MaxPool2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))
                    # D_temp= torch.tensor((input_shape[-3]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    # D=torch.floor(D_temp)

                    H_temp= torch.tensor((input_shape[-2]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    H=torch.floor(H_temp)
                    
                    W_temp= torch.tensor((input_shape[-1]+2*Padding_size[1]-  Dilation_size[1]*(Kernal_size[1]-1)-1)/ Stride_size[1]+1)
                    W=torch.floor(W_temp)

                    out_shape=tuple(input_shape[0],input_shape[1] ,H,W)
                    out_sizes_list.append(out_shape)

                    #--------------------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,nn.ConvTranspose2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))

                    
                    # D=(input_shape[-3]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    H=(input_shape[-2]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    W=(input_shape[-1]-1)*Stride_size[1]-2*Padding_size[1]+Dilation_size[1]*( Kernal_size[1]-1)+1
                    out_shape=(input_shape[0], Sub_model.out_channels,H,W)
                    out_sizes_list.append(out_shape)
                    #---------------------------------------------------------------------------------------------
                elif  isinstance(Sub_model,nn.Conv2d):
                    out_shape=(input_shape[0], Sub_model.out_channels,*input_shape[2:])
                    out_sizes_list.append(out_shape)
                #----------------------------------------------------------------------------
                else:
                    out_shape=input_shape
                    out_sizes_list.append(out_shape)
              
              
               
              
              
                if "encoder" in name:
                    encoders_output_shapes.append(out_sizes_list[-1])
                elif "upconv" in name:
                    decoders_output_shapes.append(out_sizes_list[-1])

    
        decoders_output_shapes.reverse()
        output_padding=[]

        for ii in range(len(decoders_output_shapes)):
            temp_enc_shape= encoders_output_shapes[ii]
            temp_dec_shape= decoders_output_shapes[ii]
            output_padding.append((int( temp_enc_shape[-2]-temp_dec_shape[-2]),int( temp_enc_shape[-1]-temp_dec_shape[-1])))
        output_padding.reverse()
        return output_padding
class D(nn.Module):

    def __init__(self, Input_dim, out_channels=1, init_features=21*2*1):
        super(D, self).__init__()

        features = init_features
        
        # self.attention=SE_Spatial_attention(Input_dim)
        
        
        self.encoder1 = D._block(Input_dim[1], features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2=D._block1(features,2*features,name='enc2')
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3=D._block1(2*features,4*features,name='enc3')
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4=D._block1(4*features,8*features,name='enc4')
        self.pool4=nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder5=UNet._block1(8*features,16*features,name='enc5')
        # self.pool5=nn.MaxPool2d(kernel_size=2, stride=2)
      
        # self.upconv1 = nn.ConvTranspose2d( features * 16, 16*features, kernel_size=2, stride=2)
        # self.decoder1= ( nn.Conv2d( in_channels=32*features,   out_channels=8*features, kernel_size=3, padding=1 ))
      
        self.upconv2 = nn.ConvTranspose2d( features * 8, 8*features, kernel_size=2, stride=2,groups=8*features)
        self.decoder2= ( nn.Conv2d( in_channels=16*features,   out_channels=4*features, kernel_size=3, padding=1,groups= 4*features))
      
        self.upconv3 = nn.ConvTranspose2d( features * 4, 4*features, kernel_size=2, stride=2,groups=4*features)
        self.decoder3= (nn.Conv2d(  in_channels=8*features, out_channels=2*features, kernel_size=3, padding=1,groups= 2*features))

        self.upconv4 = nn.ConvTranspose2d( features * 2, 2*features, kernel_size=2, stride=2,groups= 2*features)
        self.decoder4= (nn.ConvTranspose2d(  in_channels=4*features, out_channels=features, kernel_size=3, padding=1,groups= features ))

        self.upconv5 = nn.ConvTranspose2d( features * 1, features, kernel_size=2, stride=2,groups= features)
        self.decoder5= (nn.Conv2d(  in_channels=2*features, out_channels=features, kernel_size=3, padding=1,groups= features ))
        self._conv = nn.Conv2d(in_channels=features*1, out_channels=out_channels, kernel_size=1)
        input_dim_list=list(Input_dim)
        input_dim_list[1]=out_channels
        Input_dim_attnetion=tuple(input_dim_list)
        self.attention=SE_Spatial_attention(Input_dim_attnetion,model='D')
        self.use_attention=None
        self.Spectral_flag=np.array([True])
        self.Normalize=np.array([True])
        
       

    
            
    def forward(self, x,Attention_weight= False,  Update_u=True,Normalize=True):
       self.Spectral_flag[0]=Update_u
       self.Normalize[0]=Normalize
       output_pad=find_pad_light_D(self,x)
       name_list=[]
       for name,_ in self.named_children():
            if "upconv" in name:
                name_list.append(name)
       for i in range(len(name_list)):
            temp_sub_model=getattr(self,name_list[i])
            if isinstance(temp_sub_model,torchgan.layers.spectralnorm.SpectralNorm2d):
                temp_sub_model=temp_sub_model.module
            setattr(temp_sub_model,'output_padding',output_pad[i])
  
      
       if self.use_attention==None:
            self.use_attention=Attention_weight
        

        
       
       enc1 = self.encoder1(x)
       enc2 = self.encoder2(self.pool1(enc1))
       enc3=self.encoder3(self.pool2(enc2))
       enc4=self.encoder4(self.pool3(enc3))
       bott=self.pool4(enc4)
        # enc5=self.encoder5(self.pool4(enc4))
        # bott=self.pool5(enc5)
        # dec1 = self.upconv(bott)
        # dec1 = torch.cat((dec1, enc5), dim=1)
        # dec1=self.decoder1(dec1)
       dec1=self.upconv2(bott)
       dec1=torch.cat((dec1, enc4), dim=1)
        
       dec2=self.decoder2(dec1)
       dec2=self.upconv3(dec2)
       dec2=torch.cat((dec2, enc3), dim=1)
       dec3=self.decoder3(dec2)
       dec3=self.upconv4(dec3)
       dec3=torch.cat((dec3, enc2), dim=1)
       dec4=self.decoder4(dec3)
       dec4=self.upconv5(dec4)
       dec4=torch.cat((dec4, enc1), dim=1)
       dec1=self.decoder5(dec4)
       dec1=self._conv(dec1)
       if Attention_weight and self.use_attention:
          dec1=self.attention(dec1)
       Update_u=True
       return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                            bias=True,
                            groups=21*2
                        ),
                    ),
                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU()),
                   (
                       name + "conv2",
                       nn.Conv2d(
                           in_channels=features,
                           out_channels=features,
                           kernel_size=3,
                           padding='same',
                           bias=True,
                           groups=features
                       ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                   (name + "relu2", nn.ReLU()),
                ]
            )
        )

    def _block1(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                            bias=True,
                            groups=in_channels
                        ),
                    ),
                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU()),
                   
                ]
            )
        )
#//////////////////////////////////////////////For Discriminator//////////////////////////////////
def Dis_generator(Dis_class,Num_CL_D,Dummy_batch,S_cnn):
    # x,_=next(iter(S_loader_tSNE))
    x=Dummy_batch
    temp=S_cnn.eval()(x)[1]
    D_cnn= Dis_class(temp.shape,Num_CL_D)
    return D_cnn
#//////////////////////////////////////////Lighting_DA////////////////////////////////////////
class Light_DAtrain(L.LightningModule):
    def __init__(self, Source_model,Target_model,Discr_model,Exp_name=None,Use_Target_Attention=None,Use_Dis_Attention=None,EPOCH_DA_vin=None,LRT=None,LRD=None):
        super().__init__()
        if True:
            self.save_hyperparameters()
        self.S_cnn=Source_model
        self.T_cnn=Target_model
        self.D_cnn=Discr_model
        self.automatic_optimization = False
        self.Loss_funccc=torch.nn.CrossEntropyLoss()
        self.training_step_disscriminator=[]
        self.training_step_target=[]
        self.supervised_step_loss=[]
        self.discriminator_attention_visuals_batch=[]
        self.target_attention_visuals_batch=[]
        self.target_attention=[]
        self.discriminator_attention=[]
        self.epoch_list=[]
        self.first=True
        self.FID=[]
        # self.FID_model=FrechetInceptionDistance(feature=64,normalize=True)
        self.Dual_metric=None
        
        
        self.temp_class=1
        self.discriminator_attention_visuals_true_mask=[]
        self.target_attention_visuals_true_mask=[]
        self.Discriminator_multi_step_loss={}
        self.Exp_name=Exp_name
        self.Use_Target_Attention=Use_Target_Attention
        self.Use_Dis_Attention=Use_Dis_Attention
        self.EPOCH_DA_vin=EPOCH_DA_vin
        self.LRT=LRT
        self.LRD=LRD



      

  ##############################################################################################################
    
   
    def training_step(self, batch,batch_idx):
  
      T_optimizer,D_optimizer=self.optimizers()
      print(batch_idx)
      self.EPOCH_Num=  self.EPOCH_DA_vin
      T_patch_batch,T_idx_batch=batch[0]
      T_idx_batch.to(torch.long)
     #------------------------------------------------------------------------------------------------------------------------------------------------
      # T_label_batch_D= torch.full(tuple(T_patch_batch.shape[i] for i in [0,-2,-1]),self.trainer.datamodule.cl_num,device=self.device)
      T_label_batch_D= torch.full(T_idx_batch.shape,self.trainer.datamodule.cl_num,device=self.device)
      T_label_batch_D[T_idx_batch[:]==-100]=-100
      T_label_batch_T=torch.tensor(T_idx_batch)
      T_label_batch_T[T_label_batch_T!=-100]= T_label_batch_T[T_label_batch_T!=-100]-1
      T_label_batch_T=T_label_batch_T.to(torch.long)
      # T_label_batch_T= torch.full(tuple(T_patch_batch.shape[i] for i in [0,2,3]),1,device=self.device)
      # T_label_batch_T[T_idx_batch[:]==-100]=-100

      S_patch_batch,S_idx_batch=batch[1]
      # S_label_batch=torch.full(tuple( S_patch_batch.shape[i] for i in [0,2,3]),1,device=self.device)
      S_label_batch=torch.tensor(S_idx_batch)
      S_label_batch[S_idx_batch[:]!=-100]= S_label_batch[S_idx_batch[:]!=-100]-1
      S_label_batch=S_label_batch.to(torch.long)
      # S_label_batch[S_idx_batch[:]!=self.trainer.datamodule.courpted_class]=-100
      # S_label_batch[S_idx_batch[:]!=1]=-100
      

      


      ############################## Optimizing Discriminator #######################################
      Per_batch_discriminator_step_loss_list=[]
      for i in range(0,20):
        self.T_cnn.eval()
        T_feature_batch=self.T_cnn(T_patch_batch,Attention_weight=self.Use_Target_Attention)[1]
        # T_feature_batch=torch.cat((self.T_cnn(T_patch_batch,Attention_weight=self.Use_Target_Attention)[1],T_idx_batch[:,None]),1)
      
        # n = 0*torch.normal(0,.1,T_feature_batch.shape,device=self.device)
        
        self.S_cnn.eval()
        S_feature_batch=self.S_cnn(S_patch_batch)
        S_feature_batch=S_feature_batch[1]
       
        
        
        
        self.D_cnn.train()
        
        D_optimizer.zero_grad() 
        T_classi_batch_D=self.D_cnn(T_feature_batch,Attention_weight=  self.Use_Dis_Attention)
        
        
        T_loss_D=self.Loss_funccc(T_classi_batch_D,T_label_batch_D)
      #  
        # T_loss_D.backward(retain_graph=True)
        S_classi_batch=self.D_cnn(S_feature_batch,Attention_weight=  False)
      
        
        S_loss_D=self.Loss_funccc(S_classi_batch,S_label_batch)
        #S_loss_D.backward(retain_graph=False)
        Loss_D=T_loss_D+S_loss_D
      
        # self.trainer.logger.experiment.add_scalars('D_cnn_step_loss ', {str(self.current_epoch):Loss_D} ,i)
        # self.log('Discrimantor adversarial loss',Loss_D,prog_bar=True,on_epoch=True)
        
        # self.log_dict({'Discrimantor adversarial loss':Loss_D,'epoch':self.current_epoch},prog_bar=True,on_epoch=True)
        self.manual_backward(Loss_D,retain_graph=False)  
        Per_batch_discriminator_step_loss_list.append(Loss_D.item())#######################
      
        self.training_step_disscriminator.append(Loss_D.item())###########################  
      
        if i==0:
            self.plot_weights_mean_variance(model_type='D_cnn')
        D_optimizer.step()
        D_optimizer.zero_grad()
############################# Optimizing Generator ###############################################
      # Per_batch_discriminator_mean_loss_list=torch.stack(Per_batch_discriminator_step_loss_list).mean()
      Per_batch_discriminator_mean_loss_list= torch.tensor(Per_batch_discriminator_step_loss_list).to(torch.float32).mean().detach().item()
      self.training_step_disscriminator.append(Per_batch_discriminator_mean_loss_list)
      Per_batch_target_step_loss_list=[]
      Per_batch_target_supervised_loss_list=[]
     
     
      for i in range(0,5):
        self.D_cnn.eval()
        self.T_cnn.train()
        T_feature_batch2=self.T_cnn(T_patch_batch,Attention_weight= self.Use_Target_Attention)[1]
        # T_feature_batch2=torch.cat((self.T_cnn(T_patch_batch,Attention_weight=self.Use_Target_Attention)[1],T_idx_batch[:,None]),1)
        # T_feature_batch2=T_feature_batch
        # self.T_cnn.train()
      
        # n = 0*torch.normal(0,.1,T_feature_batch2.shape,device=self.device)
        T_feature_batch2=T_feature_batch2
    
        T_classi_batch_T=self.D_cnn( T_feature_batch2,Attention_weight=True)
        Loss_TTT=self.Loss_funccc(T_classi_batch_T,(T_label_batch_T))
        # Loss_TTT= self.Loss_funccc(T_classi_batch_T,(T_label_batch_D))
        
        # self.training_step_target.append(Loss_TTT)

        # self.log('Target adversarial learning',Loss_TTT,prog_bar=True,on_epoch=True)
        
        T_optimizer.zero_grad()
        self.manual_backward(Loss_TTT,retain_graph=False)
        T_optimizer.step()
        Per_batch_target_step_loss_list.append(Loss_TTT.detach().item())
      
        
        T_output= self.T_cnn.eval()(T_patch_batch,Attention_weight= self.Use_Target_Attention)[0]
        with torch.no_grad():
          Loss_T_supervised=self.Loss_funccc(T_output,(T_label_batch_T))
        Per_batch_target_supervised_loss_list.append( Loss_T_supervised.detach().item())###################################3
        
        # T_optimizer.zero_grad()
      # Per_batch_target_mean_loss_list=torch.stack(Per_batch_target_step_loss_list).mean()
      Per_batch_target_mean_loss_list= torch.tensor(Per_batch_target_step_loss_list).to(torch.float32).mean().detach().item()
      self.training_step_target.append(Per_batch_target_mean_loss_list)####################################
      

      # Per_batch_target_supervised_mean_loss_list= torch.stack( Per_batch_target_supervised_loss_list).mean()
      Per_batch_target_supervised_mean_loss_list=torch.tensor(Per_batch_target_supervised_loss_list).to(torch.float32).mean().detach().item()
     
      self.supervised_step_loss.append( Per_batch_target_supervised_mean_loss_list)############################
      
      return {"adversarial Target loss":Loss_TTT,"adversarial Discriminator loss": Loss_D }
    
      
    
    
    @torch.no_grad()
    def plot_weights_mean_variance(self,model_type='D_cnn'):
      if model_type=='D_cnn':
        model=self.D_cnn
      else:
        model=self.T_cnn
    # Iterate over each layer in the model
      All_weight_grad=[]
      for name, param in model.named_parameters():
          if 'weight' in name:
              # Create a new figure for each layer

              name_parts = name.split(".")
              tensor_board_tag_weights= os.path.join(model_type,*name_parts)
              name_parts[-1] = "grad"
              tensor_board_tag_grad= os.path.join(model_type,*name_parts)
              name_parts[-1] = "norm_2"
              tensor_board_tag_norm2= os.path.join(model_type,*name_parts)
              name_parts[-1] = "Big_norm"
              tensor_board_tag_Big_norm= os.path.join(model_type,"Big_norm")
                      

              weights = param.data.flatten()
              if param.grad is not None:
                gradients = param.grad.data.flatten()
                All_weight_grad.append(gradients)
                grad_norm2=torch.linalg.vector_norm(gradients)
                grad_mean=gradients.mean()
                grad_var=gradients.var()
                self.trainer.logger.experiment.add_scalars(tensor_board_tag_grad, {'G_Mean':grad_mean,'W_Variance':grad_var}, self.current_epoch)
                self.trainer.logger.experiment.add_scalar(tensor_board_tag_norm2, grad_norm2 , self.current_epoch)
                

              # Calculate mean and variance
            
              mean = weights.mean()
              variance = weights.var()
              
              self.trainer.logger.experiment.add_scalars( tensor_board_tag_weights, {'W_Mean':mean,'W_Variance':variance}, self.current_epoch)
      
      All_weight_grad=torch.cat(All_weight_grad)
      Big_norm=torch.linalg.vector_norm(All_weight_grad)
      self.trainer.logger.experiment.add_scalar(tensor_board_tag_Big_norm,  Big_norm , self.current_epoch)
              

             
    def calculate_Non_inception_FID(self):
       Dataloaders_list= self.trainer.train_dataloader
       Target_dataloader=Dataloaders_list[0]
       Source_dataloader=Dataloaders_list[1]

       All_Selected_Target_Features_Set=[]
       for batch in Target_dataloader:
          Target_Patch_batch,Target_true_mask_batch=batch
          Target_Patch_batch=Target_Patch_batch.to(self.device)
          Target_true_mask_batch=Target_true_mask_batch.to(self.device)
          Target_Feature_batch=self.T_cnn.eval()(Target_Patch_batch)[1].to(torch.float32).permute(0,2,3,1)
          Target_Feature_batch=Target_Feature_batch.reshape(-1,Target_Feature_batch.shape[-1])
          Target_true_mask_batch_flat=Target_true_mask_batch.flatten()
          Target_Feature_batch_selected=Target_Feature_batch[Target_true_mask_batch_flat!=-100]
          # Target_Feature_batch_selected=Target_Feature_batch[Target_true_mask_batch_flat== self.trainer.datamodule.courpted_class]
          All_Selected_Target_Features_Set.append(Target_Feature_batch_selected)
       All_Selected_Target_Features=torch.cat(All_Selected_Target_Features_Set,dim=0).permute(-1,-2)

       Selected_Targt_Feature_mean=All_Selected_Target_Features.mean(dim=1)+torch.zeros_like(All_Selected_Target_Features.t(),device=self.device)
       Selected_Targt_Feature_mean= Selected_Targt_Feature_mean.t()
       Selected_Targt_Feature_cov=All_Selected_Target_Features.mm((All_Selected_Target_Features-Selected_Targt_Feature_mean).t())-Selected_Targt_Feature_mean.mm((All_Selected_Target_Features-Selected_Targt_Feature_mean).t())
       Selected_Targt_Feature_cov=Selected_Targt_Feature_cov/(All_Selected_Target_Features.shape[-1]-1)
       Selected_Targt_Feature_mean=Selected_Targt_Feature_mean.mean(dim=-1)
   #------------------------------------------------------------------------------------------   
       All_Selected_Source_Features_Set=[]
       for batch in Source_dataloader:
          Source_Patch_batch,Source_true_mask_batch=batch
          Source_Patch_batch=Source_Patch_batch.to(self.device)
          Source_true_mask_batch=Source_true_mask_batch.to(self.device)
          Source_Patch_batch=self.S_cnn.eval()(Source_Patch_batch)[1].to(torch.float32).permute(0,2,3,1)
          Source_Patch_batch=Source_Patch_batch.reshape(-1,Source_Patch_batch.shape[-1])
          Source_true_mask_batch_flat=Source_true_mask_batch.flatten()
          Source_Feature_batch_selected= Source_Patch_batch[Source_true_mask_batch_flat!=-100]
          # Source_Feature_batch_selected= Source_Patch_batch[Source_true_mask_batch_flat== self.trainer.datamodule.courpted_class]
          All_Selected_Source_Features_Set.append(Source_Feature_batch_selected)
       All_Selected_Source_Features=torch.cat(All_Selected_Source_Features_Set,dim=0).permute(-1,-2)
       Selected_Source_Feature_mean=All_Selected_Source_Features.mean(dim=1)+torch.zeros_like(All_Selected_Source_Features.t(),device=self.device)
       Selected_Source_Feature_mean= Selected_Source_Feature_mean.t()
       Selected_Source_Feature_cov=All_Selected_Source_Features.mm((All_Selected_Source_Features-Selected_Source_Feature_mean).t())-Selected_Source_Feature_mean.mm((All_Selected_Source_Features-Selected_Source_Feature_mean).t())
       Selected_Source_Feature_cov=Selected_Source_Feature_cov/(All_Selected_Source_Features.shape[-1]-1)
       Selected_Source_Feature_mean=Selected_Source_Feature_mean.mean(dim=-1)

    #------------------------------------------------------------------------------------------   
       fid=_compute_fid(Selected_Targt_Feature_mean,Selected_Targt_Feature_cov,Selected_Source_Feature_mean,Selected_Source_Feature_cov)
       return fid







   
   
    @torch.no_grad()
    def calculate_FID(self):
      #  Dataloaders_list=self.trainer.datamodule.train_dataloader()
       Dataloaders_list= self.trainer.train_dataloader
       Target_dataloader=Dataloaders_list[0]
       Source_dataloader=Dataloaders_list[1]
       self.FID_model.reset()


       for batch in Target_dataloader:
            Target_Patch_batch,Target_true_mask_batch=batch
            
            if Target_Patch_batch.shape[0]==1:
              Target_Patch_batch=torch.cat((Target_Patch_batch,Target_Patch_batch),dim=0)
              Target_true_mask_batch=torch.cat((Target_true_mask_batch,Target_true_mask_batch),dim=0)
            Target_Patch_batch=Target_Patch_batch.to(self.device)
            Target_true_mask_batch=Target_true_mask_batch.to(self.device)
            Target_Feature_batch=self.T_cnn.eval()(Target_Patch_batch,Normalize=True)[1].to(torch.float32)
            Target_Feature_batch=Target_Feature_batch.permute(0,2,3,1).reshape(-1,Target_Feature_batch.shape[2]*Target_Feature_batch.shape[3],Target_Feature_batch.shape[1])
            PCA_Target_Feature_tupele=torch.pca_lowrank(Target_Feature_batch)
            Projected_Target_Feature=torch.matmul(Target_Feature_batch,PCA_Target_Feature_tupele[-1][:,:,0:3].to(self.device)).reshape(-1,Target_Patch_batch.shape[2],Target_Patch_batch.shape[3],3).permute(0,3,1,2)
            Projected_Target_Feature=Projected_Target_Feature/(Projected_Target_Feature.max()-Projected_Target_Feature.min())
            Projected_Target_Feature= Projected_Target_Feature-Projected_Target_Feature.min()
            Target_true_mask_batch=(Target_true_mask_batch[:,None] +torch.zeros_like(Projected_Target_Feature,device=self.device)).to(torch.int64)
            # Projected_Target_Feature[Target_true_mask_batch==-100]=0


            self.FID_model.update(Projected_Target_Feature, real=False)
            # self.FID_model.update(Projected_Target_Feature, real=True)


       for batch in Source_dataloader:
          Source_Patch_batch,Source_true_mask_batch=batch
          if Source_Patch_batch.shape[0]==1:
            Source_Patch_batch=torch.cat((Source_Patch_batch,Source_Patch_batch),dim=0)
            Source_true_mask_batch=torch.cat((Source_true_mask_batch,Source_true_mask_batch),dim=0)
          Source_Patch_batch=Source_Patch_batch.to(self.device)
          Source_true_mask_batch=Source_true_mask_batch.to(self.device)
          Source_Feature_batch=self.S_cnn.eval()(Source_Patch_batch)[1].to(torch.float32)
          Source_Feature_batch=Source_Feature_batch.permute(0,2,3,1).reshape(-1,Source_Feature_batch.shape[2]*Source_Feature_batch.shape[3],Source_Feature_batch.shape[1])
          PCA_Source_Feature_tupele=torch.pca_lowrank(Source_Feature_batch)
          Projected_Source_Feature=torch.matmul(Source_Feature_batch,PCA_Source_Feature_tupele[-1][:,:,0:3].to(self.device)).reshape(-1,Source_Patch_batch.shape[2],Source_Patch_batch.shape[3],3).permute(0,3,1,2)
          Projected_Source_Feature=Projected_Source_Feature/(Projected_Source_Feature.max()-Projected_Source_Feature.min())
          Projected_Source_Feature= Projected_Source_Feature-Projected_Source_Feature.min()
          Source_true_mask_batch=(Source_true_mask_batch[:,None] +torch.zeros_like(Projected_Source_Feature,device=self.device)).to(torch.int64)
          # Projected_Source_Feature[Source_true_mask_batch==-100]=0
          self.FID_model.update(Projected_Source_Feature, real=True)

       return self.FID_model.compute()

    def configure_optimizers(self):
        T_optimizer = torch.optim.Adam(get_param(self.T_cnn), lr=self.LRT,maximize=False)
        D_optimizer = torch.optim.Adam((self.D_cnn.parameters()),lr=self.LRD)
        return [T_optimizer,D_optimizer]
    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.Dual_metric is None:
           self.Dual_metric=-100.7
     
        # discriminator_mean = torch.stack(self.training_step_disscriminator).mean()
        discriminator_mean= torch.tensor(self.training_step_disscriminator).mean()

        # target_mean = torch.stack(self.training_step_target).mean()
        target_mean= torch.tensor(self.training_step_target).mean()
        # target_supervised_mean=torch.stack( self.supervised_step_loss).mean()
        target_supervised_mean= torch.tensor(self.supervised_step_loss).mean()
        self.logger.experiment.add_scalars  ('Adversarial phase',{"Discriminator loss": discriminator_mean, "Target loss": target_mean},self.current_epoch)
        self.logger.experiment.add_scalar ('Target supervised loss', target_supervised_mean,self.current_epoch)
        # if (discriminator_mean>2e-3 or target_mean>2e-3) and target_supervised_mean >.45 :
        if  target_supervised_mean >=1:     
          # if self.Dual_metric<0:
          #    self.Dual_metric=0
          # else:
             self.Dual_metric+=-.001
        self.log('Dual_metric',self.Dual_metric)
        # self.logger.experiment.add_scalar ('FID', self.calculate_Non_inception_FID(),self.current_epoch)
        # self.log('FID_metric', self.calculate_Non_inception_FID())
        self.log('Dummy metric',-self.current_epoch)
        # self.log('FID_metric', self.calculate_Non_inception_FID())
        self.log('target_supervised_metric',target_supervised_mean)
        # self.plot_weights_mean_variance(model_type='D_cnn')
        self.plot_weights_mean_variance(model_type='T_cnn')
       
        #For attention visualization##########################################################################
        if self.current_epoch%(16 if True else 1 )==0 :
          if self.D_cnn.use_attention and not 'full' in self.Exp_name:
            self.epoch_list.append(self.current_epoch)
        #---------------------------------------------------------------------------------------------#
            # attention_weights=self.D_cnn.eval().attention(self.S_cnn.eval()(self.discriminator_attention_visuals_batch[0])[1],get_attention_weights=True)[1]
        #---------------------------------------------------------------------------------------------#   
            attention_weights=self.T_cnn.eval()(self.discriminator_attention_visuals_batch[0])[1]
            encoder_list=[]
            for name, sub_module in self.D_cnn.eval().named_children():
          
                  if "encoder" in name:
                      attention_weights=sub_module(attention_weights)
                      encoder_list.append(attention_weights)
                  elif "decoder" in name:
                      x_concat=torch.cat((attention_weights,encoder_list[-1]),dim=1)
                      attention_weights=sub_module(x_concat)
                      encoder_list.pop()
                  else:
                      
                      if name=="attention":
                          attention_weights=sub_module(attention_weights,get_attention_weights=True)[1]
                          break
                      attention_weights=sub_module(attention_weights)
        #---------------------------------------------------------------------------------------------#  
            temp_dict={}
            for temp_class in torch.unique(self.discriminator_attention_visuals_true_mask[0][0]):
                temp_dict[str(temp_class.item())]=(torch.max(attention_weights[0][self.discriminator_attention_visuals_true_mask[0][0,None]==temp_class]))
            self.logger.experiment.add_scalars ('Weight_at_class_pixels', temp_dict,self.current_epoch)
            # self.logger.experiment.add_scalar ('Weight_at_class_pixels', torch.mean(attention_weights[self.discriminator_attention_visuals_true_mask[0][:,None]==self.temp_class]),self.current_epoch)
            self.discriminator_attention.append(attention_weights)
            self.logger.experiment.add_images('Discriminator Attention visualization',attention_weights.permute((0,2,3,1)),self.current_epoch,dataformats='NHWC')
            self.logger.experiment.add_images('Discriminator input visualization',self.discriminator_attention_visuals_batch[0][:,0:3],self.current_epoch,dataformats='NCHW')
         
       ############################################################################################################## 
          if self.T_cnn.use_attention and not 'full' in self.Exp_name:
            encoder_list=[]
            attention_weights= self.target_attention_visuals_batch[0]
            for name,sub_module in self.T_cnn.eval().named_children():
              if "encoder" in name:
                      attention_weights=sub_module(attention_weights)
                      encoder_list.append(attention_weights)
              elif "decoder" in name:
                  x_concat=torch.cat((attention_weights,encoder_list[-1]),dim=1)
                  attention_weights=sub_module(x_concat)
                  encoder_list.pop()
              else:
                  
                  if name=="attention":
                      attention_weights=sub_module(attention_weights,get_attention_weights=True)[1]
                      break
                  attention_weights=sub_module(attention_weights)


            self.target_attention.append( attention_weights)
        
          
            self.logger.experiment.add_images('Target Attention visualization',attention_weights.permute((0,2,3,1)),self.current_epoch,dataformats='NHWC')
          # self.logger.experiment.add_images('Features visualization',(self.T_cnn.eval()(self.discriminator_attention_visuals_batch[0])[1][0][:,None,:,:]),self.current_epoch,dataformats='NCHW')
        # free up the memory
        self.training_step_disscriminator.clear()
        self.training_step_target.clear()
        self.supervised_step_loss.clear()

#//////////////////////////////////////////DA_TRAIN///////////////////////////////////////////////////
def DATrain(Dataset,batch_size=10,num_workers=16,Result_path=None,Source_Datamodule=None,random_seed=None,lightgin_log_path=None,keep_train=False, Num_of_Target_train_samples=None,S_cnn=None,EPOCH=None,LRT=None,LRD=None,Exp_name=None,devices=None,accelerator=None,Use_Target_Attention=None,Use_Dis_Attention=None,i=None):

    DA_Datamodule=Dataset.DA_Datamodule(batch_size=batch_size,num_workers=num_workers,Source_Datamodule_arg=Source_Datamodule,Num_of_Samples=Num_of_Target_train_samples)
    
    ch_num=DA_Datamodule.ch_num
    cl_num=DA_Datamodule.cl_num
    Dummy_batch=DA_Datamodule.get_batch()
   
    T_cnn= UNet(ch_num,  cl_num,input_shape=Dummy_batch[0].shape)

   
    
  
    Dummy_batch=DA_Datamodule.get_batch(batch_Datset='Source')
    S_cnn_DUMMY= UNet(ch_num,  cl_num,input_shape=Dummy_batch[0].shape)
    if random_seed is not None:
      seed_everything(random_seed)
    D_cnn=Dis_generator(D, cl_num+1,Dummy_batch[0],S_cnn_DUMMY)
    Xavi_init_weights(D_cnn)
   
    EPOCH_DA=EPOCH
    
    
 
    st=copy.deepcopy(S_cnn.state_dict())
  
    T_cnn.load_state_dict(st)
    set_parameter_requires_grad(T_cnn)
    
   
    S_cnn.requires_grad_(False)
   
    

    ########################### Using lighting ###################################
    
    light_DA_model=  Light_DAtrain(S_cnn,T_cnn,D_cnn,Exp_name=Exp_name,Use_Target_Attention=Use_Target_Attention,Use_Dis_Attention=Use_Dis_Attention,EPOCH_DA_vin=EPOCH,LRT=LRT,LRD=LRD)
    # Checkpoint_callback=ModelCheckpoint(dirpath= Result_path,filename='DA_best_model_UH_{epoch}',monitor='Dummy metric')
    # Checkpoint_callback_best_pair=ModelCheckpoint(dirpath= Result_path,filename='DA_best_model_best_sul_{epoch}',monitor='target_supervised_mean')
    Checkpoint_callback=ModelCheckpoint(dirpath= Result_path,filename='DA_best_model_best_sul_{epoch}_RUN_NUM'+str(i),monitor='target_supervised_metric')

    Early_callbackk= EarlyStopping(monitor='Dual_metric',patience=15,check_on_train_epoch_end=True,mode='min',min_delta=0,strict=False)
    Tensorborad_logger=L.pytorch.loggers.tensorboard.TensorBoardLogger(name=Exp_name,save_dir=lightgin_log_path)
    # ddp = DDPStrategy(process_group_backend="nccl",find_unused_parameters=True)



    #####################keep training ##########################################
    if keep_train:
      Tensorborad_logger=L.pytorch.loggers.tensorboard.TensorBoardLogger(name=Exp_name,save_dir=lightgin_log_path,version=2)
      
      list_of_files = glob.glob(os.path.join(Result_path, '*.ckpt')) 
     ###########################################################################3
      latest_ckpt_path_string= '/project/prasad/Germany for CARYA/Results/Germany_wishpers_MOSAIC/DA/DA_best_model_best_sul_epoch=30.ckpt'
     
      ##########################Get the latest file ##########################################
      # latest_ckpt_path_string= max(list_of_files, key=os.path.getctime)
      ##########################Get by the epoch number ##########################################
      # import re

      # target_epoch = 298 # Change this to the epoch number you want to find

      # for file_path in list_of_files:
      #   match = re.search(r'epoch=(\d+)', file_path)
      #   if match and int(match.group(1)) == target_epoch:
      #     latest_ckpt_path_string=file_path
      #     break
      #   else:
      #     print(f"No file found for epoch {target_epoch}")
      ################################################################################################
      light_DA_model=  Light_DAtrain.load_from_checkpoint(latest_ckpt_path_string,Source_model=S_cnn,Target_model=T_cnn,Discr_model=D_cnn)
      trainer = L.Trainer(max_epochs=30,logger=Tensorborad_logger,devices=devices_DA,callbacks=[Checkpoint_callback,Early_callbackk],deterministic='warn')
      trainer.fit(model=light_DA_model, datamodule=DA_Datamodule,ckpt_path=latest_ckpt_path_string)
      if not "With Spectral Norm"  in Exp_name:
        return  light_DA_model.T_cnn,light_DA_model,Checkpoint_callback.best_model_path,DA_Datamodule
      else: 
        return  Checkpoint_callback.best_model_path,Dataset.DA_Datamodule(batch_size=batch_size,num_workers=num_workers,Source_Datamodule_arg=Source_Datamodule)
    if False:
      if accelerator_type_DA=="gpu":
        
        trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator_type_DA,max_epochs= EPOCH_DA,strategy='ddp_notebook_find_unused_parameters_true',devices=devices_DA,callbacks=[Checkpoint_callback, Early_callbackk],deterministic='warn')
        # trainer = L.Trainer(accelerator=accelerator_type_DA,max_epochs= EPOCH_DA,devices=devices_DA)
        # trainer = L.Trainer(accelerator=accelerator_type_DA,max_epochs= EPOCH_DA,devices=devices_DA)
      else:
        trainer = L.Trainer(logger=Tensorborad_logger,deterministic='warn',accelerator=accelerator_type_DA,max_epochs= EPOCH_DA,limit_train_batches=0.05, limit_val_batches=0.05)
    else:
       if accelerator=="gpu":
          # trainer = L.Trainer(accelerator=accelerator_type_DA,max_epochs= EPOCH_DA,strategy="ddp_notebook_find_unused_parameters_true",devices=devices_DA)
          trainer = L.Trainer(logger=Tensorborad_logger,deterministic='warn',accelerator=accelerator,max_epochs= EPOCH,devices=devices,callbacks=[ Checkpoint_callback,Early_callbackk])
       else:
          trainer = L.Trainer(logger=Tensorborad_logger,deterministic=True,accelerator=accelerator,max_epochs= EPOCH,limit_train_batches=1, limit_val_batches=1,callbacks=[Checkpoint_callback, Early_callbackk])
       
      ##########################################################################################3
     
    trainer.fit(model=light_DA_model, datamodule=DA_Datamodule)
 
    
    
    return  light_DA_model.T_cnn,light_DA_model,Checkpoint_callback.best_model_path,DA_Datamodule
#//////////////////////////Load from state dictionary////////////////////////////////////////
def load_from_state_dict(path=None):
    Adpated_models_path= path
    assert Adpated_models_path is not None, "path cannot be None"
    with open(Adpated_models_path, 'rb') as file:
    
        State_list= dill.load(file)  
    DA_Datamodule_obj=DA_Datamodule()
    ch_num=DA_Datamodule_obj.ch_num
    cl_num=DA_Datamodule_obj.cl_num
    Dummy_batch=DA_Datamodule_obj.get_batch()
    T_model_my=[]
    for s_dict in State_list:
        T_cnn= UNet(ch_num,  cl_num,input_shape=Dummy_batch[0].shape)
        T_cnn.load_state_dict(s_dict)
        T_model_my.append(T_cnn)
    return T_model_my