import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from tensorboard_logger import configure, log_value

import os
import tqdm

from data import dataset as dset
from models import models
from utils import utils

cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='.', help='base directory')
parser.add_argument('--p_model', default='resnet18', help='vgg16|vgg19|resnet18|resnet101|resnet152')
parser.add_argument('--dataset', default='zappos', help='mitstates|zappos')
parser.add_argument('--data_dir', default='data/ut-zap50k/', help='data/mit-states/|data/ut-zap50k/')
parser.add_argument('--cv_dir', default='cv/tmp/', help='dir to save checkpoints to')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')

# model parameters
parser.add_argument('--model', default='contribnet', help='visprodNN|visprodcontribNN|genmodel|contribnet')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of common embedding space')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout for layers in projector')
parser.add_argument('--glove_init', action='store_true', default=False, help='initialize inputs with word vectors')
parser.add_argument('--clf_init', action='store_true', default=False, help='initialize inputs with SVM weights')
parser.add_argument('--static_inp', action='store_true', default=False, help='do not optimize input representations')

# Decoupled detection ablations/regularizers
parser.add_argument('--lambda_enc', type=float, default=1.0) #1.0
parser.add_argument('--lambda_discr', type=float, default=2.0) #2.0
parser.add_argument('--lambda_triplet', type=float, default=0.2) #0.2
parser.add_argument('--lambda_recon', type=float, default=0.5) #2.0 for mitstates and 0.5 for zappos
parser.add_argument('--lambda_autoenc', type=float, default=2.0) #2.0

# optimization
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--save_every', type=int, default=10) # Must be multiple of and >= eval_val_every
parser.add_argument('--eval_val_every', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=1000)
args = parser.parse_args()

os.makedirs(os.path.join(args.base_dir, args.cv_dir), exist_ok=True)
utils.save_args(args)

#----------------------------------------------------------------#

def train(epoch):

	model.train()

	train_loss = 0.0
	for idx, data in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

		data = [d.cuda() for d in data]
		loss, _ = model(data)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()

	train_loss = train_loss/len(trainloader)
	log_value('train_loss', train_loss, epoch)
	print ('E: %d | L: %.2E'%(epoch, train_loss))


def test(epoch):

	model.eval()

	accuracies = list()

	for idx, data in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

		data = [d.cuda() for d in data]
		attr_truth, obj_truth = data[1], data[2]

		if args.model in ['genmodel', 'contribnet']:
			_, predictions, _ = model(data)
		else:
			_, predictions = model(data)

		results = evaluator.score_model(predictions, obj_truth)
		match_stats = evaluator.evaluate_predictions(results, attr_truth, obj_truth)
		accuracies.append(match_stats)

	accuracies = zip(*accuracies)
	accuracies = map(torch.mean, map(torch.cat, accuracies))
	attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc = accuracies

	log_value('test_attr_acc', attr_acc, epoch)
	log_value('test_obj_acc', obj_acc, epoch)
	log_value('test_closed_acc', closed_acc, epoch)
	log_value('test_open_acc', open_acc, epoch)
	log_value('test_objoracle_acc', objoracle_acc, epoch)
	print ('(test) E: %d | A: %.3f | O: %.3f | Cl: %.4f | Op: %.4f | OrO: %.4f'%(epoch, attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc))

	if epoch>0 and epoch%args.save_every==0:
		state = {
			'net': model.state_dict(),
			'epoch': epoch,
		}

		torch.save(state, os.path.join(args.base_dir, args.cv_dir)+'/%s_ckpt_E_%d_A_%.3f_O_%.3f_Cl_%.3f_Op_%.3f.t7'
			           %(args.p_model, epoch, attr_acc, obj_acc, closed_acc, open_acc))

#----------------------------------------------------------------#
trainset = dset.CompositionDatasetActivations(root=os.path.join(args.base_dir, args.data_dir), pmodel=args.p_model, phase='train', split='compositional-split')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testset = dset.CompositionDatasetActivations(root=os.path.join(args.base_dir, args.data_dir), pmodel=args.p_model, phase='test', split='compositional-split')
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

model = None
if args.model == 'visprodNN':
	model = models.VisualProductNN(trainset, args)
elif args.model =='genmodel':
	model = models.GenModel(trainset, args)
elif args.model =='contribnet':
	model = models.ContribNet(trainset, args)
elif args.model == 'visprodcontribNN':
	model = models.VisProdContribNN(trainset, args)

if args.model in ['visprodcontribNN', 'genmodel', 'contribnet']:
	evaluator = models.DecoupledEvaluator(trainset, model)
else:
	evaluator = models.Evaluator(trainset, model)

params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

model.cuda()
print (model)

start_epoch = 1
cv_dir = os.path.join(args.base_dir, args.cv_dir)
if args.load is not None:
	checkpoint = torch.load(os.path.join(cv_dir, args.load))
	model.load_state_dict(checkpoint['net'])
	start_epoch = checkpoint['epoch']+1
	print ('loaded model from', os.path.basename(args.load))

configure(cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs):
	train(epoch)
	if epoch%args.eval_val_every==0:
		with torch.no_grad():
			test(epoch)

# ======================================================================================================================
