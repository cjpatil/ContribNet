import torch
import torch.utils.data

from data import dataset as dset
import tqdm
from models import models
import os

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='.', help='base directory')
parser.add_argument('--p_model', default='resnet18', help='resnet18|vgg16|resnet101|resnet152|densenet161')
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='data/ut-zap50k/', help='data/mit-states/|data/ut-zap50k/')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')

# model parameters
parser.add_argument('--model', default='contribnet', help='visprodNN|visprodcontribNN|genmodel|contribnet')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of common embedding space')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout for layers in projector')
parser.add_argument('--glove_init', action='store_true', default=False, help='initialize inputs with word vectors')
parser.add_argument('--clf_init', action='store_true', default=False, help='initialize inputs with SVM weights')
parser.add_argument('--static_inp', action='store_true', default=False, help='do not optimize input representations')

# optimization
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=512)

args = parser.parse_args()

def test(epoch):
	model.eval()

	accuracies = list()

	pred_list = list()
	discr_list = list()

	# mname = model.__class__.__name__

	for idx, data in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

		data = [d.cuda() for d in data]
		attr_truth, obj_truth = data[1], data[2]

		if args.model in ['genmodel', 'contribnet']:
			_, predictions, discr = model(data)
			pred_list.append(predictions)
			discr_list.append(discr)
		else:
			_, predictions = model(data)

		results = evaluator.score_model(predictions, obj_truth)
		match_stats = evaluator.evaluate_predictions(results, attr_truth, obj_truth)
		accuracies.append(match_stats)

	accuracies = zip(*accuracies)
	accuracies = map(torch.mean, map(torch.cat, accuracies))
	attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc = accuracies

	print('(test) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.3f | OrO: %.3f' % (epoch, attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc))

	if args.model in ['genmodel', 'contribnet']:
		# Predict mAP, pair top-k accuracy, pair-attr top-k accuracy, pair-obj top-k accuracy
		closed_map, closed_acc, closed_pair_attr_acc, closed_pair_obj_acc, discr_attr_acc, discr_obj_acc, open_acc = evaluator.score_all_metrics(pred_list, discr_list)

		print('(test-topk) E: %d | CP: %s | CPA: %s | CPO: %s | mAP: %.1f | CDA: %.1f | CDO: %.1f | OP: %s' % (
			epoch, closed_acc, closed_pair_attr_acc, closed_pair_obj_acc, closed_map, discr_attr_acc, discr_obj_acc, open_acc))

#----------------------------------------------------------------#

testset = dset.CompositionDatasetActivations(root=os.path.join(args.base_dir, args.data_dir), pmodel=args.p_model, phase='test', split='compositional-split')
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

model = None
if args.model == 'visprodNN':
	model = models.VisualProductNN(testset, args)
elif args.model =='genmodel':
	model = models.GenModel(testset, args)
elif args.model =='contribnet':
	model = models.ContribNet(testset, args)
elif args.model == 'visprodcontribNN':
	model = models.VisProdContribNN(testset, args)

if args.model in ['visprodcontribNN', 'genmodel', 'contribnet']:
	evaluator = models.DecoupledEvaluator(testset, model)
else:
	evaluator = models.Evaluator(testset, model)

model.cuda()
# print(model)

checkpoint = torch.load(args.load)
model.load_state_dict(checkpoint['net'])
epoch = checkpoint['epoch']
print ('loaded model from', os.path.basename(args.load))

with torch.no_grad():
	test(epoch)


# ======================================================================================================================
