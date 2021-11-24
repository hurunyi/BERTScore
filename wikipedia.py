from bert_score import BERTScorer
import io
import os
import csv
import numpy as np
from tqdm import tqdm


wikipedia_path = "/data/hurunyi/wikipedia"


def read_data(fpath):
	sent1 = []
	sent2 = []
	sent3 = []

	with open(os.path.join(fpath, "test.csv"), "r") as f:
		skipFirstLine = True
		reader = csv.reader(f, delimiter=',', quotechar='"')
		print("Loading data...")
		for text in tqdm(reader):
			if skipFirstLine:
				skipFirstLine = False
			else:
				sent1.append(text[1])
				sent2.append(text[2])
				sent3.append(text[3])
		print("Successfully loaded!")
	return sent1, sent2, sent3

def main():
	scorer = BERTScorer(model_type="bert-base-uncased", lang="en")

	sent1, sent2, sent3 = read_data(wikipedia_path)
	_, _, F1_2 = scorer.score(sent1, sent2)
	_, _, F1_3 = scorer.score(sent1, sent3)
	score2 = F1_2.numpy()
	score3 = F1_3.numpy()
	right_cot = np.sum(score2 > score3)
	print(f"Acc: {right_cot} / {len(score2)} = {right_cot/len(score2)}")


if __name__ == "__main__":
	main()
