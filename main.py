from bert_score import BERTScorer
import io
import os
import csv
import numpy as np
from scipy.stats import spearmanr, pearsonr

stsb_fpath = '/home/hurunyi/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv'
sts_fpath = '/home/hurunyi/SentEval/data/downstream/STS'
sick_fpath = '/home/hurunyi/SentEval/data/downstream/SICK/SICK_test_annotated.txt'
afs_fpath = '/home/hurunyi/SentEval/data/downstream/AFS'


def read_stsb_data(fpath):
	sent1 = [line.split("\t")[5] for line in io.open(fpath, encoding='utf8').read().splitlines()]
	sent2 = [line.split("\t")[6] for line in io.open(fpath, encoding='utf8').read().splitlines()]
	raw_scores = np.array([line.split("\t")[4] for line in io.open(fpath, encoding='utf8').read().splitlines()])

	not_empty_idx = raw_scores != ''
	gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
	sent1 = list(np.array(sent1)[not_empty_idx])
	sent2 = list(np.array(sent2)[not_empty_idx])

	return sent1, sent2, gs_scores


def read_sts_data(fpath, sts_type):
	datasets = []
	sent1 = []
	sent2 = []
	raw_scores = []

	if sts_type == 'STS12':
		datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
					'surprise.OnWN', 'surprise.SMTnews']
	elif sts_type == 'STS13':
		datasets = ['FNWN', 'headlines', 'OnWN']
	elif sts_type == 'STS14':
		datasets = ['deft-forum', 'deft-news', 'headlines',
					'images', 'OnWN', 'tweet-news']
	elif sts_type == 'STS15':
		datasets = ['answers-forums', 'answers-students',
					'belief', 'headlines', 'images']
	elif sts_type == 'STS16':
		datasets = ['answer-answer', 'headlines', 'plagiarism',
					'postediting', 'question-question']

	for dataset in datasets:
		sent1_tmp, sent2_tmp = zip(*[l.split("\t") for l in io.open(fpath + '/STS.input.%s.txt' % dataset,
									 encoding='utf8').read().splitlines()])
		raw_scores_tmp = np.array([x for x in io.open(fpath + '/STS.gs.%s.txt' % dataset,
									   encoding='utf8').read().splitlines()])
		not_empty_idx = raw_scores_tmp != ''
		sent1.extend(np.array(sent1_tmp)[not_empty_idx])
		sent2.extend(np.array(sent2_tmp)[not_empty_idx])
		raw_scores.extend(raw_scores_tmp[not_empty_idx])

	gs_scores = [float(x) for x in raw_scores]

	return sent1, sent2, gs_scores


def read_sick_data(fpath):
	sent1 = []
	sent2 = []
	raw_scores = []

	skipFirstLine = True
	with io.open(fpath, 'r', encoding='utf-8') as f:
		for line in f:
			if skipFirstLine:
				skipFirstLine = False
			else:
				text = line.strip().split('\t')
				sent1.append(text[1])
				sent2.append(text[2])
				raw_scores.append(text[3])

	gs_scores = [float(s) for s in raw_scores]

	return sent1, sent2, gs_scores


def read_afs_data(fpath):
	datasets = ['ArgPairs_DP', 'ArgPairs_GC', 'ArgPairs_GM']
	sent1 = []
	sent2 = []
	raw_scores = []

	for dataset in datasets:
		with open(fpath + '/%s.csv' % dataset, 'r', encoding='utf-8', errors='ignore') as f:
			skipFirstLine = True
			reader = csv.reader(f)
			for text in reader:
				if skipFirstLine:
					skipFirstLine = False
				else:
					sent1.append(text[9])
					sent2.append(text[10])
					raw_scores.append(text[0])

	gs_scores = [float(x) for x in raw_scores]

	return sent1, sent2, gs_scores


def main():
	task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSb', 'SICK-R', 'AFS']
	scorer = BERTScorer(model_type="bert_mnli_outputs/checkpoint-6000", num_layers=12, lang="en")

	sent1 = []
	sent2 = []
	gs_scores = []

	for task in task_list:
		if task == 'STS12':
			sent1, sent2, gs_scores = read_sts_data(os.path.join(sts_fpath, 'STS12-en-test'), task)
		elif task == 'STS13':
			sent1, sent2, gs_scores = read_sts_data(os.path.join(sts_fpath, 'STS13-en-test'), task)
		elif task == 'STS14':
			sent1, sent2, gs_scores = read_sts_data(os.path.join(sts_fpath, 'STS14-en-test'), task)
		elif task == 'STS15':
			sent1, sent2, gs_scores = read_sts_data(os.path.join(sts_fpath, 'STS15-en-test'), task)
		elif task == 'STS16':
			sent1, sent2, gs_scores = read_sts_data(os.path.join(sts_fpath, 'STS16-en-test'), task)
		elif task == 'STSb':
			sent1, sent2, gs_scores = read_stsb_data(stsb_fpath)
		elif task == 'SICK-R':
			sent1, sent2, gs_scores = read_sick_data(sick_fpath)
		elif task == 'AFS':
			sent1, sent2, gs_scores = read_afs_data(afs_fpath)

		P, R, F1 = scorer.score(sent1, sent2)
		sys_scores = F1.numpy()

		pearson = pearsonr(sys_scores, gs_scores)
		spearman_score = spearmanr(sys_scores, gs_scores)

		print(f"Task: {task}, Pearson: {pearson[0]}, Spearman: {spearman_score[0]}")


if __name__ == "__main__":
	main()
