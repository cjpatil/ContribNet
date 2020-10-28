import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import math

def load_word_embeddings(emb_file, vocab):

	vocab = [v.lower() for v in vocab]

	embeds = {}
	for line in open(emb_file, 'r'):
		line = line.strip().split(' ')
		wvec = torch.FloatTensor(list(map(float, line[1:])))
		embeds[line[0]] = wvec

	# for zappos (should account for everything)
	custom_map = {'Faux.Fur':'fur', 'Faux.Leather':'leather', 'Full.grain.leather':'leather', 'Hair.Calf':'hair', 'Patent.Leather':'leather', 'Nubuck':'leather',
	              'Boots.Ankle':'boots', 'Boots.Knee.High':'knee-high', 'Boots.Mid-Calf':'midcalf', 'Shoes.Boat.Shoes':'shoes', 'Shoes.Clogs.and.Mules':'clogs',
	              'Shoes.Flats':'flats', 'Shoes.Heels':'heels', 'Shoes.Loafers':'loafers', 'Shoes.Oxfords':'oxfords', 'Shoes.Sneakers.and.Athletic.Shoes':'sneakers'}
	for k in custom_map:
		embeds[k.lower()] = embeds[custom_map[k]]

	embeds = [embeds[k] for k in vocab]
	embeds = torch.stack(embeds)
	print ('loaded embeddings', embeds.size())

	return embeds

#--------------------------------------------------------------------------------#

class MLP(nn.Module):
	def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
		super().__init__()
		mod = []
		for L in range(num_layers-1):
			mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
			mod.append(nn.ReLU(True))

		mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
		if relu:
			mod.append(nn.ReLU(True))

		self.mod = nn.Sequential(*mod)

	def forward(self, x):
		output = self.mod(x)
		return output


class Evaluator:

	def __init__(self, dset, model):

		self.dset = dset

		# convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
		pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
		self.pairs = torch.LongTensor(pairs)

		# mask over pairs that occur in closed world
		test_pair_set = set(dset.test_pairs)
		mask = [1 if pair in test_pair_set else 0 for pair in dset.pairs]
		self.closed_mask = torch.BoolTensor(mask)

		# object specific mask over which pairs occur in the object oracle setting
		oracle_obj_mask = []
		for _obj in dset.objs:
			mask = [1 if _obj==obj else 0 for attr, obj in dset.pairs]
			oracle_obj_mask.append(torch.BoolTensor(mask))
		self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

		# decide if the model being evaluated is a manifold model or not
		mname = model.__class__.__name__
		if mname in ['VisualProductNN'] :
			self.score_model = self.score_clf_model
		else:
			self.score_model = self.score_manifold_model


	# generate masks for each setting, mask scores, and get prediction labels
	def generate_predictions(self, scores, obj_truth): # (B, #pairs)

		def get_pred_from_scores(_scores):
			_, pair_pred = _scores.max(1)
			attr_pred, obj_pred = self.pairs[pair_pred][:,0], self.pairs[pair_pred][:,1]
			return (attr_pred, obj_pred)

		results = {}

		# open world setting -- no mask
		results.update({'open': get_pred_from_scores(scores)})

		# closed world setting - set the score for all NON test-pairs to -1e10
		mask = self.closed_mask.repeat(scores.shape[0], 1)
		closed_scores = scores.clone()
		closed_scores[mask.logical_not()] = -1e10
		results.update({'closed': get_pred_from_scores(closed_scores)})

		# object_oracle setting - set the score to -1e10 for all pairs where the true object does NOT participate
		mask = self.oracle_obj_mask[obj_truth]
		oracle_obj_scores = scores.clone()
		oracle_obj_scores[mask.logical_not()] = -1e10
		results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores)})

		return results

	def score_clf_model(self, scores, obj_truth):

		attr_pred, obj_pred = scores

		# put everything on CPU
		attr_pred, obj_pred, obj_truth = attr_pred.cpu(), obj_pred.cpu(), obj_truth.cpu()

		# - gather scores (P(a), P(o)) for all relevant (a,o) pairs
		# - multiply P(a)*P(o) to get P(pair)
		attr_subset = attr_pred.index_select(1, self.pairs[:,0])
		obj_subset = obj_pred.index_select(1, self.pairs[:,1])
		scores = (attr_subset*obj_subset) # (B, #pairs)

		results = self.generate_predictions(scores, obj_truth)
		return results

	def score_manifold_model(self, scores, obj_truth):

		# put everything on CPU
		scores = {k:v.cpu() for k,v in scores.items()}
		obj_truth = obj_truth.cpu()

		# gather scores for all relevant (a,o) pairs
		scores = torch.stack([scores[(attr, obj)] for attr, obj in self.dset.pairs], 1) # (B, #pairs)
		results = self.generate_predictions(scores, obj_truth)
		return results

	def evaluate_predictions(self, predictions, attr_truth, obj_truth):

		# put everything on cpu
		attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()

		# top 1 pair accuracy
		# open world: attribute, object and pair
		attr_match = (attr_truth==predictions['open'][0]).float()
		obj_match = (obj_truth==predictions['open'][1]).float()
		open_match = attr_match*obj_match

		# closed world, obj_oracle: pair
		closed_match = (attr_truth==predictions['closed'][0]).float() * (obj_truth==predictions['closed'][1]).float()
		obj_oracle_match = (attr_truth==predictions['object_oracle'][0]).float() * (obj_truth==predictions['object_oracle'][1]).float()

		return attr_match, obj_match, closed_match, open_match, obj_oracle_match


class VisualProductNN(nn.Module):
	def __init__(self, dset, args):
		super(VisualProductNN, self).__init__()
		self.attr_clf = MLP(dset.feat_dim, len(dset.attrs), 2, relu=False)
		self.obj_clf = MLP(dset.feat_dim, len(dset.objs), 2, relu=False)
		self.dset = dset

	def train_forward(self, x):
		img, attrs, objs = x[0], x[1], x[2]

		attr_pred = self.attr_clf(img)
		obj_pred = self.obj_clf(img)

		attr_loss = F.cross_entropy(attr_pred, attrs)
		obj_loss = F.cross_entropy(obj_pred, objs)
		loss = attr_loss + obj_loss

		return loss, None

	def val_forward(self, x):
		img = x[0]
		attr_pred = F.softmax(self.attr_clf(img), dim=1)
		obj_pred = F.softmax(self.obj_clf(img), dim=1)
		return None, [attr_pred, obj_pred]

	def forward(self, x):
		if self.training:
			loss, pred = self.train_forward(x)
		else:
			with torch.no_grad():
				loss, pred = self.val_forward(x)
		return loss, pred


# -------------------------------------------------------------------------------------------------------------- #


class Projector(nn.Module):

	def __init__(self, features_d, proj_d, num_layers=1, dropout=0.0, n_slope=0.1):
		super().__init__()

		modules = list()
		start_d = features_d
		for i in range(num_layers-1):
			layer_d = features_d - math.floor((i + 1) * (features_d - proj_d) / num_layers)
			modules.append(nn.Linear(start_d, layer_d))
			# modules.append(nn.ReLU(True))
			modules.append(nn.LeakyReLU(negative_slope=n_slope))
			if dropout>0.0:
				modules.append(nn.Dropout(p=dropout))
			start_d = layer_d
		modules.append(nn.Linear(start_d, proj_d))

		self.project = nn.Sequential(*modules)

		self.init_weights()

	def init_weights(self):
		"""
		Initializes some parameters with values from the uniform distribution, for easier convergence.
		"""
		# print("inside init_weights")

		# # Another way of initialising
		# torch.nn.init.uniform_(self.fc[0].weight, -0.1, 0.1)

		for layer in self.project.children():
			if isinstance(layer, nn.Linear):
				layer.weight.data.uniform_(-0.01, 0.01)
				layer.bias.data.fill_(0.0)

	def forward(self, features):
		"""

		@param features:
		@return:
		"""

		proj_f = self.project(features)
		return proj_f


class DecoupledEvaluator:

	def __init__(self, dset, model):

		self.dset = dset
		self.k = 5

		# convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
		pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
		self.pairs = torch.LongTensor(pairs)

		# mask over pairs that occur in closed world
		test_pair_set = set(dset.test_pairs)
		mask = [1 if pair in test_pair_set else 0 for pair in dset.pairs]
		self.closed_mask = torch.BoolTensor(mask)

		# object specific mask over which pairs occur in the object oracle setting
		oracle_obj_mask = []
		for _obj in dset.objs:
			mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
			oracle_obj_mask.append(torch.BoolTensor(mask))
		self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

		# decide if the model being evaluated is a manifold model or not
		mname = model.__class__.__name__
		if mname in ['VisProdContribNN']:
			self.score_model = self.score_clf_model
		else:
			self.score_model = self.score_manifold_model

	# generate masks for each setting, mask scores, and get prediction labels
	def generate_predictions(self, scores, obj_truth):  # (B, #pairs)

		def get_pred_from_scores(_scores): # pair predictions for 1962 pairs
			_, pair_pred = _scores.max(1)
			attr_pred, obj_pred = self.pairs[pair_pred][:, 0], self.pairs[pair_pred][:, 1]
			return (attr_pred, obj_pred)

		results = {}

		# open world setting -- no mask
		results.update({'open': get_pred_from_scores(scores)})

		# closed world setting - set the score for all NON test-pairs to -1e10
		mask = self.closed_mask.repeat(scores.shape[0], 1)
		closed_scores = scores.clone()
		closed_scores[mask.logical_not()] = -1e10
		results.update({'closed': get_pred_from_scores(closed_scores)})

		# object_oracle setting - set the score to -1e10 for all pairs where the true object does NOT participate
		mask = self.oracle_obj_mask[obj_truth]
		oracle_obj_scores = scores.clone()
		oracle_obj_scores[mask.logical_not()] = -1e10
		results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores)})

		return results

	def score_clf_model(self, scores, obj_truth):

		attr_pred, obj_pred = scores

		# put everything on CPU
		attr_pred, obj_pred, obj_truth = attr_pred.cpu(), obj_pred.cpu(), obj_truth.cpu()

		# - gather scores (P(a), P(o)) for all relevant (a,o) pairs
		# - multiply P(a)*P(o) to get P(pair)
		attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
		obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
		scores = (attr_subset * obj_subset)  # (B, #pairs)

		results = self.generate_predictions(scores, obj_truth)
		return results

	def score_manifold_model(self, scores, obj_truth):

		# put everything on CPU
		scores = {k: v.cpu() for k, v in scores.items()}
		obj_truth = obj_truth.cpu()

		# gather scores for all relevant (a,o) pairs
		scores = torch.stack([scores[(attr, obj)] for attr, obj in self.dset.pairs], 1)  # (B, #pairs)
		results = self.generate_predictions(scores, obj_truth)
		return results

	def evaluate_predictions(self, predictions, attr_truth, obj_truth):

		# put everything on cpu
		attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()

		# top 1 pair accuracy
		# open world: attribute, object and pair
		attr_match = (attr_truth == predictions['open'][0]).float()
		obj_match = (obj_truth == predictions['open'][1]).float()
		open_match = attr_match * obj_match

		# closed world, obj_oracle: pair
		closed_match = (attr_truth == predictions['closed'][0]).float() * (
				obj_truth == predictions['closed'][1]).float()
		obj_oracle_match = (attr_truth == predictions['object_oracle'][0]).float() * (
				obj_truth == predictions['object_oracle'][1]).float()

		return attr_match, obj_match, closed_match, open_match, obj_oracle_match

	def score_all_metrics(self, pred_list, discr_list):

		predictions = None

		for pred in pred_list:
			# put everything on CPU
			scores = {k: v.cpu() for k, v in pred.items()}

			# gather scores for all relevant (a,o) pairs
			scores = torch.stack([scores[(attr, obj)] for attr, obj in self.dset.pairs], 1)  # (B, #pairs)
			predictions = scores if predictions is None else torch.cat([predictions, scores], dim=0)

		num_samples = len(predictions)
		# _, open_topk = predictions.topk(self.k, dim=1, largest=True, sorted=True)

		closed_predictions = predictions.clone()
		closed_predictions[(self.closed_mask.logical_not()).repeat((num_samples, 1))] = -1e10
		# _, closed_topk = closed_predictions.topk(self.k, dim=1, largest=True, sorted=True)

		test_pair_truth = torch.tensor([self.dset.pair2idx[(self.dset.test_data[i][1], self.dset.test_data[i][2])] for i in range(num_samples)], dtype=torch.long)

		open_acc = self.get_accuracy_upto_topk(predictions, test_pair_truth)
		closed_acc = self.get_accuracy_upto_topk(closed_predictions, test_pair_truth)

		# open_map = self.get_mean_average_precision(predictions, test_pair_truth)
		closed_map = self.get_mean_average_precision(closed_predictions, test_pair_truth)

		attr_truth = torch.tensor([self.dset.attr2idx[self.dset.test_data[i][1]] for i in range(num_samples)], dtype=torch.long)
		obj_truth = torch.tensor([self.dset.obj2idx[self.dset.test_data[i][2]] for i in range(num_samples)], dtype=torch.long)

		closed_pair_attr_acc = self.extract_accuracy_from_pair("attr", closed_predictions, attr_truth)
		closed_pair_obj_acc = self.extract_accuracy_from_pair("obj", closed_predictions, obj_truth)

		attr_pred = torch.cat([discr[0].cpu() for discr in discr_list], dim=0)
		obj_pred = torch.cat([discr[1].cpu() for discr in discr_list], dim=0)

		attr_pred_acc = self.extract_accuracy_from_discr("attr", attr_pred, attr_truth)
		obj_pred_acc = self.extract_accuracy_from_discr("obj", obj_pred, obj_truth)

		return closed_map, closed_acc, closed_pair_attr_acc, closed_pair_obj_acc, attr_pred_acc, obj_pred_acc, open_acc

	def get_accuracy_upto_topk(self, pred, gt):
		"""
		To get the accuracy upto k values
		@param pred: 2-D
		@param gt: 1-D
		@return: list of accuracies
		"""
		total_samples = pred.shape[0]

		k_acc = list()
		for k in range(1, self.k + 1):
			_, ind = pred.topk(k, dim=1, largest=True, sorted=True)  # Get the indices of topk entries

			correct = ind.eq(gt.view(-1, 1).expand_as(ind))  # Find how many predictions are correct
			correct_total = correct.view(-1).int().sum().item()  # 0D tensor

			acc = 100.0 * (correct_total / total_samples)
			k_acc.append(round(acc, 1))
		return k_acc

	def get_mean_average_precision(self, pred, gt):
		"""

		@param pred: 1D tensor for predictions
		@param gt: 1D tensor for ground-truth
		@return:
		"""
		# Find the distinct classes from gt
		gt_labels = set(gt.tolist())

		total_precision = 0

		# Get the argmax
		pred = pred.argmax(dim=1)

		# Find precision for each class
		for label in gt_labels:
			precision = self.get_precision(pred, gt, label)
			total_precision += precision

		# Average it
		ap = total_precision / len(gt_labels)
		return ap

	def get_precision(self, pred, gt, label):
		"""

		@param pred: 1D tensor for predictions
		@param gt: 1D tensor for ground-truth
		@param label:
		@return:
		"""
		total_samples = len(pred)

		tp, tn, fp, fn = 0, 0, 0, 0
		for i in range(total_samples):
			if pred[i].item() == label:
				if pred[i].item() == gt[i].item():
					tp += 1  # True positive
				else:
					fp += 1  # False positive

		precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0

		return precision

	def extract_accuracy_from_pair(self, acc_for, pred, gt):
		"""
		To get the accuracy upto k values
		@param acc_for: "attr" or "obj"
		@param upto_k: int
		@param pred: 2-D
		@param gt: 1-D
		@return: list of accuracies
		"""
		total_samples = len(pred)

		k_acc = list()
		for k in range(1, self.k + 1):
			_, ind = pred.topk(k, dim=1, largest=True, sorted=True)  # Get the indices of topk entries

			# Get the attribute / object for the top-k test pairs
			for i in range(total_samples):
				for j in range(k):
					ind[i][j] = self.pairs[ind[i][j]][0] if acc_for == "attr" else self.pairs[ind[i][j]][1]

			correct_total = 0

			for i in range(total_samples):
				if gt[i] in ind[i].tolist():
					correct_total += 1

			acc = 100.0 * (correct_total / total_samples)
			k_acc.append(round(acc, 1))
		return k_acc

	def extract_accuracy_from_discr(self, acc_for, pred, gt):
		total_samples = pred.shape[0]
		ind = pred.argmax(dim=1)

		if pred.shape[1] == len(self.dset.pairs):
			extract = torch.tensor([self.pairs[ind[i]][0] if acc_for=="attr" else self.pairs[ind[i]][1] for i in range(total_samples)])
			correct = extract.eq(gt)  # Find how many predictions are correct
		else:
			correct = ind.eq(gt)  # Find how many predictions are correct
		correct_total = correct.view(-1).int().sum().item()  # 0D tensor
		return round(100.0 * (correct_total / total_samples), 1)


class DecoupledManifoldModel(nn.Module):

	def __init__(self, dset, args):
		super().__init__()
		self.args = args
		self.dset = dset

		self.margin = .9

		# precompute validation pairs
		attrs, objs = zip(*self.dset.pairs)
		attrs = [dset.attr2idx[attr] for attr in attrs]
		objs = [dset.obj2idx[obj] for obj in objs]
		self.val_attrs = torch.LongTensor(attrs).cuda()
		self.val_objs = torch.LongTensor(objs).cuda()

		input_dim = dset.feat_dim if args.clf_init else args.emb_dim

		self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
		self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

		# init with word embeddings
		if args.glove_init:
			pretrained_weight = load_word_embeddings(os.path.join(args.base_dir, 'data/glove/glove.6B.300d.txt'), dset.attrs)
			self.attr_embedder.weight.data.copy_(pretrained_weight)
			pretrained_weight = load_word_embeddings(os.path.join(args.base_dir, 'data/glove/glove.6B.300d.txt'), dset.objs)
			self.obj_embedder.weight.data.copy_(pretrained_weight)

		# # implement these in subclasses
		# self.compare_metric = lambda img_embed, pair_embed: None
		# self.image_attr_embedder = lambda img: None
		# self.image_obj_embedder = lambda img: None
		# self.attr_sim = lambda img_attr_embed, attr_embed: None
		# self.obj_sim = lambda img_obj_embed, obj_embed: None
		# self.train_forward = lambda x: None

	def cosine_triplet_loss(self, anchor, positive, negative, margin):
		distance_positive = F.cosine_embedding_loss(anchor, positive, torch.tensor(1).to('cuda')) # should be close to 0
		distance_negative = F.cosine_embedding_loss(anchor, negative, torch.tensor(1).to('cuda')) # should be close to 1

		losses = torch.relu(distance_positive - distance_negative + margin)
		return losses.mean()

	def forward(self, x):
		if self.training:
			return self.train_forward(x)
		else:
			with torch.no_grad():
				return self.val_forward(x)


class ContribNet(DecoupledManifoldModel):
	def __init__(self, dset, args):
		super().__init__(dset, args)

		proj_d = 1024

		self.image_attr_embedder = nn.Sequential(
			Projector(dset.feat_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.image_obj_embedder = nn.Sequential(
			Projector(dset.feat_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.embed_decoder = Projector(proj_d, dset.feat_dim, num_layers=2, dropout=args.dropout)

		self.word_attr_embedder = nn.Sequential(
			Projector(args.emb_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.word_obj_embedder = nn.Sequential(
			Projector(args.emb_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.compare_metric = lambda img_feats, pair_embed: F.cosine_similarity(img_feats, pair_embed)

		self.obj_discr = Projector(proj_d, len(dset.objs), num_layers=2, dropout=args.dropout)
		self.attr_discr = Projector(proj_d, len(dset.attrs), num_layers=2, dropout=args.dropout)

		self.contrib_d = 300
		self.obj_contrib = Projector(proj_d, self.contrib_d, num_layers=2, dropout=args.dropout)
		self.attr_contrib = Projector(proj_d, self.contrib_d, num_layers=2, dropout=args.dropout)

		self.obj_contrib_recon = Projector(self.contrib_d, proj_d, num_layers=2, dropout=args.dropout)
		self.attr_contrib_recon = Projector(self.contrib_d, proj_d, num_layers=2, dropout=args.dropout)

		with_contrib_d = proj_d+self.contrib_d
		intermediate_d = with_contrib_d
		self.obj_contrib_discr = nn.Sequential(
				nn.Linear(with_contrib_d, intermediate_d),
				nn.LeakyReLU(negative_slope=0.1),
				nn.Dropout(args.dropout),
				nn.Linear(intermediate_d, len(dset.objs))
		)

		self.attr_contrib_discr = nn.Sequential(
				nn.Linear(with_contrib_d, intermediate_d),
				nn.LeakyReLU(negative_slope=0.1),
				nn.Dropout(args.dropout),
				nn.Linear(intermediate_d, len(dset.attrs))
		)

	def compose(self, attrs, objs):
		output = self.word_attr_embedder(self.attr_embedder(attrs)) + self.word_obj_embedder(self.obj_embedder(objs))

		return output

	def compose_image_feat(self, img_attr_feats, img_obj_feats):
		img_feats = img_attr_feats + img_obj_feats

		return img_feats

	def predict_attr_label(self, attr_features, attr_pred, contrib_from_obj, obj_features, obj_pred):
		contrib_pred = self.attr_contrib_discr(torch.cat([attr_features, contrib_from_obj], dim=1))

		if self.training:
			return contrib_pred

		return contrib_pred

	def predict_obj_label(self, obj_features, obj_pred, contrib_from_attr, attr_features, attr_pred):
		contrib_pred = self.obj_contrib_discr(torch.cat([obj_features, contrib_from_attr], dim=1))

		if self.training:
			return contrib_pred

		return contrib_pred

	def train_forward(self, x):

		img, attrs, objs = x[0], x[1], x[2]
		neg_attrs, neg_objs = x[4], x[5]

		img_obj_feats = self.image_obj_embedder(img)
		img_attr_feats = self.image_attr_embedder(img)

		composed_feats = self.compose_image_feat(img_attr_feats, img_obj_feats)

		positive = self.compose(attrs, objs)
		negative = self.compose(neg_attrs, neg_objs)

		# Encoding loss
		enc_loss = F.cosine_embedding_loss(composed_feats, positive, torch.tensor(1).cuda())

		# Encoding triplet loss
		trip_loss = self.cosine_triplet_loss(composed_feats, positive, negative, self.margin)

		discr_loss = 0.
		recon_loss = 0.
		autoenc_loss = 0.

		# Discrimination loss
		if self.args.lambda_discr > 0.:

			obj_pred = self.obj_discr(img_obj_feats)
			attr_pred = self.attr_discr(img_attr_feats)

			# Prediction considering contribution
			contrib_from_attr = self.attr_contrib(img_attr_feats)
			contrib_from_obj = self.obj_contrib(img_obj_feats)
			# Recontruction of contribution features, autoencoder
			attr_contrib_recon = self.attr_contrib_recon(contrib_from_attr)
			obj_contrib_recon = self.obj_contrib_recon(contrib_from_obj)
			autoenc_loss += F.mse_loss(attr_contrib_recon, img_attr_feats)
			autoenc_loss += F.mse_loss(obj_contrib_recon, img_obj_feats)

			obj_pred = self.predict_obj_label(img_obj_feats, obj_pred, contrib_from_attr, img_attr_feats, attr_pred)
			attr_pred = self.predict_attr_label(img_attr_feats, attr_pred, contrib_from_obj, img_obj_feats, obj_pred)

			discr_loss += F.cross_entropy(attr_pred, attrs)
			discr_loss += F.cross_entropy(obj_pred, objs)

		# Reconstruction loss
		if self.args.lambda_recon > 0.:
			img_dec = self.embed_decoder(composed_feats)
			recon_loss += F.mse_loss(img_dec, img)

		# Calculate the loss
		loss = self.args.lambda_enc * enc_loss + self.args.lambda_triplet * trip_loss + self.args.lambda_autoenc * autoenc_loss + \
		       self.args.lambda_discr * discr_loss + self.args.lambda_recon * recon_loss

		return loss, None

	def val_forward(self, x):
		img, attrs, objs = x[0], x[1], x[2]
		batch_size = img.shape[0]

		img_obj_feats = self.image_obj_embedder(img)
		img_attr_feats = self.image_attr_embedder(img)

		composed_feats = self.compose_image_feat(img_attr_feats, img_obj_feats)

		pair_embeds = self.compose(self.val_attrs, self.val_objs)

		scores = {}
		for i, (attr, obj) in enumerate(self.dset.pairs):
			pair_embed = pair_embeds[i, None].expand(batch_size, pair_embeds.size(1))
			score = self.compare_metric(composed_feats, pair_embed)
			scores[(attr, obj)] = score

		obj_pred = self.obj_discr(img_obj_feats)
		attr_pred = self.attr_discr(img_attr_feats)

		# Prediction considering contribution
		contrib_from_attr = self.attr_contrib(img_attr_feats)
		contrib_from_obj = self.obj_contrib(img_obj_feats)

		obj_pred = self.predict_obj_label(img_obj_feats, obj_pred, contrib_from_attr, img_attr_feats, attr_pred)
		attr_pred = self.predict_attr_label(img_attr_feats, attr_pred, contrib_from_obj, img_obj_feats, obj_pred)

		attr_pred = F.softmax(attr_pred, dim=1)
		obj_pred = F.softmax(obj_pred, dim=1)

		return None, scores, [attr_pred, obj_pred]


class GenModel(DecoupledManifoldModel):
	def __init__(self, dset, args):
		super().__init__(dset, args)

		proj_d = 1024

		self.image_attr_embedder = nn.Sequential(
			Projector(dset.feat_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.image_obj_embedder = nn.Sequential(
			Projector(dset.feat_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.embed_decoder = Projector(proj_d, dset.feat_dim, num_layers=2, dropout=args.dropout)

		self.word_attr_embedder = nn.Sequential(
			Projector(args.emb_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.word_obj_embedder = nn.Sequential(
			Projector(args.emb_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.attr_clf = Projector(proj_d, len(dset.attrs), num_layers=2, dropout=args.dropout)
		self.obj_clf = Projector(proj_d, len(dset.objs), num_layers=2, dropout=args.dropout)

		self.compare_metric = lambda img_feats, pair_embed: F.cosine_similarity(img_feats, pair_embed)

	def compose(self, attrs, objs):
		output = self.word_attr_embedder(self.attr_embedder(attrs)) + self.word_obj_embedder(self.obj_embedder(objs))

		return output

	def compose_image_feat(self, img_attr_feats, img_obj_feats):
		img_feats = img_attr_feats + img_obj_feats

		return img_feats

	def train_forward(self, x):

		img, attrs, objs = x[0], x[1], x[2]
		neg_attrs, neg_objs = x[4], x[5]

		img_attr_feats = self.image_attr_embedder(img)
		img_obj_feats = self.image_obj_embedder(img)

		img_feats = self.compose_image_feat(img_attr_feats, img_obj_feats)

		positive = self.compose(attrs, objs)
		negative = self.compose(neg_attrs, neg_objs)

		enc_loss = F.cosine_embedding_loss(img_feats, positive, torch.tensor(1).cuda())

		trip_loss = self.cosine_triplet_loss(img_feats, positive, negative, self.margin)

		discr_loss = 0.
		recon_loss = 0.

		if self.args.lambda_discr > 0.:
			attr_pred = self.attr_clf(img_attr_feats)
			discr_loss += F.cross_entropy(attr_pred, attrs)

			obj_pred = self.obj_clf(img_obj_feats)
			discr_loss += F.cross_entropy(obj_pred, objs)

		if self.args.lambda_recon > 0.:
			img_dec = self.embed_decoder(img_feats)
			recon_loss += F.mse_loss(img_dec, img)

		loss = enc_loss + self.args.lambda_triplet * trip_loss + self.args.lambda_discr * discr_loss + self.args.lambda_recon * recon_loss
		return loss, None

	def val_forward(self, x):
		img, attrs, objs = x[0], x[1], x[2]
		batch_size = img.shape[0]

		img_attr_feats = self.image_attr_embedder(img)
		img_obj_feats = self.image_obj_embedder(img)

		img_feats = self.compose_image_feat(img_attr_feats, img_obj_feats)

		pair_embeds = self.compose(self.val_attrs, self.val_objs)

		scores = {}
		for i, (attr, obj) in enumerate(self.dset.pairs):
			pair_embed = pair_embeds[i, None].expand(batch_size, pair_embeds.size(1))
			score = self.compare_metric(img_feats, pair_embed)
			scores[(attr, obj)] = score

		attr_pred = F.softmax(self.attr_clf(img_attr_feats), dim=1)
		obj_pred = F.softmax(self.obj_clf(img_obj_feats), dim=1)

		return None, scores, [attr_pred, obj_pred]


class VisProdContribNN(VisualProductNN):
	def __init__(self, dset, args):
		super(VisualProductNN, self).__init__()
		self.dset = dset
		self.args = args

		proj_d = 1024

		self.image_attr_embedder = nn.Sequential(
			Projector(dset.feat_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.image_obj_embedder = nn.Sequential(
			Projector(dset.feat_dim, proj_d),
			nn.Dropout(args.dropout)
		)

		self.contrib_d = 300
		self.obj_contrib = Projector(proj_d, self.contrib_d, num_layers=2, dropout=args.dropout)
		self.attr_contrib = Projector(proj_d, self.contrib_d, num_layers=2, dropout=args.dropout)

		self.obj_contrib_recon = Projector(self.contrib_d, proj_d, num_layers=2, dropout=args.dropout)
		self.attr_contrib_recon = Projector(self.contrib_d, proj_d, num_layers=2, dropout=args.dropout)

		with_contrib_d = proj_d + self.contrib_d
		intermediate_d = with_contrib_d
		self.obj_contrib_discr = nn.Sequential(
			nn.Linear(with_contrib_d, intermediate_d),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Dropout(args.dropout),
			nn.Linear(intermediate_d, len(dset.objs))
		)

		self.attr_contrib_discr = nn.Sequential(
			nn.Linear(with_contrib_d, intermediate_d),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Dropout(args.dropout),
			nn.Linear(intermediate_d, len(dset.attrs))
		)

	def train_forward(self, x):
		img, attrs, objs = x[0], x[1], x[2]

		loss = 0.0
		img_obj_feats = self.image_obj_embedder(img)
		img_attr_feats = self.image_attr_embedder(img)

		# Prediction considering contribution
		contrib_from_attr = self.attr_contrib(img_attr_feats)
		contrib_from_obj = self.obj_contrib(img_obj_feats)
		# Recontruction of contribution features, autoencoder
		attr_contrib_recon = self.attr_contrib_recon(contrib_from_attr)
		obj_contrib_recon = self.obj_contrib_recon(contrib_from_obj)
		loss += self.args.lambda_autoenc * F.mse_loss(attr_contrib_recon, img_attr_feats)
		loss += self.args.lambda_autoenc * F.mse_loss(obj_contrib_recon, img_obj_feats)

		obj_pred = self.obj_contrib_discr(torch.cat([img_obj_feats, contrib_from_attr], dim=1))
		attr_pred = self.attr_contrib_discr(torch.cat([img_attr_feats, contrib_from_obj], dim=1))

		loss += self.args.lambda_discr * F.cross_entropy(attr_pred, attrs)
		loss += self.args.lambda_discr * F.cross_entropy(obj_pred, objs)

		return loss, None

	def val_forward(self, x):
		img = x[0]
		img_obj_feats = self.image_obj_embedder(img)
		img_attr_feats = self.image_attr_embedder(img)

		# Prediction considering contribution
		contrib_from_attr = self.attr_contrib(img_attr_feats)
		contrib_from_obj = self.obj_contrib(img_obj_feats)

		obj_pred = self.obj_contrib_discr(torch.cat([img_obj_feats, contrib_from_attr], dim=1))
		attr_pred = self.attr_contrib_discr(torch.cat([img_attr_feats, contrib_from_obj], dim=1))

		attr_pred = F.softmax(attr_pred, dim=1)
		obj_pred = F.softmax(obj_pred, dim=1)
		return None, [attr_pred, obj_pred]

	def forward(self, x):
		if self.training:
			loss, pred = self.train_forward(x)
		else:
			with torch.no_grad():
				loss, pred = self.val_forward(x)
		return loss, pred