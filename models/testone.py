import sys
sys.path.insert(0,"..")
sys.path.insert(0,"../data/input_images")

from config import ModelConfig
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
from lib.kern_model import KERN
import matplotlib.pyplot as plt
import numpy
from torch.autograd import Variable
import torchvision.transforms as transforms
from dataloaders.input import custom

from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
import os

def imshow(img,filename):
    img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    npimg = img

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()
    plt.savefig(filename+'.png')


conf = ModelConfig()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
						  use_proposals=conf.use_proposals,
						  filter_non_overlap=conf.mode == 'sgdet')

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
											   batch_size=conf.batch_size,
											   num_workers=conf.num_workers,
											   num_gpus=conf.num_gpus)


detector = KERN(classes=test.ind_to_classes, rel_classes=test.ind_to_predicates,
				num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
				use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
				use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
				ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
				use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
				use_ggnn_rel=conf.use_ggnn_rel, ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
				ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim, ggnn_rel_output_dim=conf.ggnn_rel_output_dim,
				use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge)

detector.cuda()
ckpt = torch.load(conf.ckpt)

'''
#print (val)
print (val[0]['img'].view(-1,3,592,592).size())
#print (val[0]['img_size'])


transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#changed the num_worker to 0

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

dataiter = iter(testloader)
images, labels = dataiter.next()

optimistic_restore(detector, ckpt['state_dict'])
detector.eval()

for val_b, batch in enumerate(tqdm(val_loader)):
        print ('Hi')
        det_res = detector[batch]
        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i= det_res
        print('boxes_i :')
        print(boxes_i)
        print('length of boxes_i')
        print (len(boxes_i))
        print('obj_i :')
        print(objs_i)
        print('length of obj_i')
        print (len(objs_i))
        print('obj_scores :')
        print(obj_scores_i)
        print('length of obj_scores_i')
        print (len(obj_scores_i))
        print('rels_ is :')
        print(rels_i)
        print('length of rels_i')
        print (len(rels_i))
        print('pred_scores is :')
        print(pred_scores_i)
        print('length of pred_scores_i')
        print (len(pred_scores_i))
        print (len(pred_scores_i[0]))




        dataiter = iter(val_loader)
        images= dataiter.next()[0]
#print (images[0].data.numpy()[0,:,:,:].shape)
        imshow(images[0].data.numpy()[0,:,:,:],str(val_b))
'''
#print (dataiter.next()[0])
#print (val_loader.dataset.size())
#print (ckpt.keys())
#model.load_state_dict(ckpt )
#model.eval()
#images = torch.from_numpy(numpy.array(images))
#optimistic_restore(detector, ckpt['state_dict'])
#images = Variable(torch.from_numpy(numpy.array(images)))
#print (images.size())
#print (val[0].keys())
#print (train[0].keys())
#print (val[0]['gt_boxes'])
#print (val[0]['gt_classes'])
##print (val[0]['index'])
#print ('batch')
#for val_b, batch in enumerate(tqdm(val_loader)):#
#	print (dir(batch))
#	print (type(batch.im_sizes))
#	print (batch.im_sizes)
#	break	
	#sys.exit(0)

#print (np.asarray(val[0]['img_size']))
#corrected_size=[[[val[0]['img_size']val[0]['img_size']val[0]['img_size']]]]
#print (detector(Variable(val[0]['img'].view(-1,3,592,592).cuda()),np.asarray([val[0]['img_size']]),0,gt_boxes=val[0]['gt_boxes'], gt_classes=val[0]['gt_classes']  , gt_rels=val[0]['gt_relations']  , train_anchor_inds=val[0]['index']))

#print (detector(Variable(val[0]['img'].view(-1,3,592,592).cuda()),np.asarray([val[0]['img_size']]),0,gt_boxes=np.zeros([1,4]), gt_classes=np.zeros([1,3])  , gt_rels=np.zeros([1,3]) , train_anchor_inds=0))

#print ('testing on valloader: ')
#print (detector[batch])



all_inputs = os.listdir('/home/saeid/KERN/data/input_images')
result_path= "/home/saeid/KERN/results/"
input_path= "/home/saeid/KERN/data/input_images/"

for image in all_inputs:

	output = detector(custom(input_path+image)[0].cuda(),custom(input_path+image)[1],0,gt_boxes=np.zeros([1,4]), gt_classes=np.zeros([1,3])  , gt_rels=np.zeros([1,3]) , train_anchor_inds=0)

	print ('dir output')
	print (dir(output))
	(boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i)= output


	print ('boxes_i')
	print (boxes_i)
	print ('boxes_i.shape')
	print (boxes_i.shape)
	np.savetxt(result_path+image.split('.')[0]+"_boxes.csv",boxes_i,delimiter=",")
	print (boxes_i.shape)
	print ('objs_i')
	print (objs_i)
	np.savetxt(result_path+image.split('.')[0]+"_objs.csv",objs_i,delimiter=",")
	print (len(objs_i))
	print ('obj_scores_i')
	print (obj_scores_i)
	print (len(obj_scores_i))
	print ('rels_i')
	print (type(rels_i))
	np.savetxt(result_path+image.split('.')[0]+"_rels.csv",rels_i,delimiter=",")
	print (rels_i)
	print (rels_i.shape)
	print ('pred_scores_i')
	print (pred_scores_i)
	np.savetxt(result_path+image.split('.')[0]+"_pred_scores.csv",pred_scores_i,delimiter=",")
	print (pred_scores_i.shape)

	#input()

'''
print (type(Variable(val[0]['img'].view(-1,3,592,592))))
print (val[0]['gt_boxes'])
print (type(val[0]['gt_classes']))
print (type(val[0]['gt_relations']) )
print (val[0]['gt_relations'] )
print (type(val[0]['index']))
print (val[0]['index'])
'''
#if conf.mode == 'sgdet':
#    det_ckpt = torch.load(conf.ckpt)['state_dict']
#    detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#    detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#    detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#    detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])


