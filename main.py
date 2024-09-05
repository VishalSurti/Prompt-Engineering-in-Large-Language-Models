import os
import requests
import pandas as pd
import time

df_train = pd.read_csv('dataset/gsm8k_train.csv')
df_test = pd.read_csv('dataset/gsm8k_test.csv')

ds = {'train': df_train, 'test': df_test}

#prompting_technique = 'zero_shot'
#prompting_technique = 'chain_of_thought'
prompting_technique = 'least_to_most'

BASE_URL = "http://127.0.0.1:11434"
GEN_URL = BASE_URL + "/api/generate"

class Timer:
	def __init__(self):
		self.start_time = None
		self.end_time = None

	def start(self):
		self.start_time = time.time()
		return self.start_time
	
	def end(self):
		self.end_time = time.time()
		return self.elapsed()

	def elapsed(self):
		return f"{(self.end_time - self.start_time):.3f}s"

def get_response(model, question, stream=False):
    params = {"model": model, "prompt": question, "stream": stream}
    res = requests.post(GEN_URL, json=params)
    return res.json()['response']

def chain_of_thought(question):
	chain_question=question+"\nLets think about it step by step."
	return chain_question

def least_to_most(question):
	return question+"\n What subproblems must be solved before answering the question?"

def save_to_file_test(filename, question, answer, dataset_ans):
	if os.path.isfile(os.path.join(os.getcwd(), filename)):
		df = pd.read_csv(filename)
		df.loc[len(df), df.columns] = question, answer, dataset_ans
		df.to_csv(filename, index=False)
	else:
		print(f"File does not exist! Creating {filename}")
		df = pd.DataFrame(columns=['Question', 'Answer', 'Dataset Answer'])
		df.loc[len(df), df.columns] = question, answer, dataset_ans
		df.to_csv(filename, index=False)

timer = Timer()

if prompting_technique == 'zero_shot':
	for type in ['train', 'test']:
		checkpoint = 0
		if os.path.isfile(f'dataset_question_{type}.csv'):
			df = pd.read_csv(f'cataset_question_{type}.csv')
			checkpoint = len(df)
		for i in range(checkpoint, len(ds[type])):
			q = ds[type].loc[i]['question']
			dataset_ans=ds[type].loc[i]['answer']
			print(f"Prompting Question {i + 1} of {len(ds[type])} ({((i+1)/len(ds[type]))*100:.3f}%)")
			timer.start()
			a = get_response('llama3', q)
			timer.end()
			print(f'Response received in {timer.elapsed()}! Saving response.')
			save_to_file_test(f'chain_of_thought_output_{type}.csv', q, a, dataset_ans)

if prompting_technique == 'chain_of_thought':
	for type in ['train', 'test']:
		checkpoint = 0
		if os.path.isfile(f'chain_of_thought_output_{type}.csv'):
			df = pd.read_csv(f'chain_of_thought_output_{type}.csv')
			checkpoint = len(df)
		for i in range(checkpoint, len(ds[type])):
			q = ds[type].loc[i]['question']
			dataset_ans=ds[type].loc[i]['answer']
			print(f"Prompting Question {i + 1} of {len(ds[type])} ({((i+1)/len(ds[type]))*100:.3f}%)")
			timer.start()
			cq=chain_of_thought(q)
			a = get_response('llama3', cq)
			timer.end()
			print(f'Response received in {timer.elapsed()}! Saving response.')
			save_to_file_test(f'chain_of_thought_output_{type}.csv', q, a, dataset_ans)

if prompting_technique == 'least_to_most':
	for type in ['train', 'test']:
		checkpoint = 0
		if os.path.isfile(f'least_to_most_output_{type}.csv'):
			df = pd.read_csv(f'least_to_most_output_{type}.csv')
			checkpoint = len(df)
		for i in range(checkpoint, len(ds[type])):
			q = ds[type].loc[i]['question']
			dataset_ans=ds[type].loc[i]['answer']
			print(f"Prompting Question {i + 1} of {len(ds[type])} ({((i+1)/len(ds[type]))*100:.3f}%)")
			timer.start()
			ltm = least_to_most(q)
			a = get_response('llama3', ltm)
			timer.end()
			print(f'Response received in {timer.elapsed()}! Saving response.')
			save_to_file_test(f'least_to_most_output_{type}.csv', q, a, dataset_ans)