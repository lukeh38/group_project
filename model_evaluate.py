from trainer import predicter
import numpy as np
import os
import time 
def evaluate():

	directory = "new_training"
	correct = 0
	total = 0

	for filename in os.listdir(directory):
		if filename.endswith(".jpg"):
			name = os.path.join(directory,filename)
			time.sleep(1)
			output = predicter(0.2, name)
			if len(output)>1:
				label =output[0]
			else:
				label = "NA"
			label = convert(label)
			txt_name = filename[0:-3]
			txt_name = txt_name+'txt'
			txt_name = os.path.join(directory,txt_name)
			f = open(txt_name,'r')
			content = f.read()
			corr = content[0]
			# if label == 'hand':
			# 	label = '0'
			# elif label == 'thumbs_up':
			# 	label = '1'
			# elif label == 'thumbs_down':
			# 	label = '2'
			
			total += 1
			if corr == label:
				correct = correct +1
	print((correct/total)*100)


def convert(label):
	if label == 'hand':
		return '0'
	elif label == 'thumbs_up':
		return '1'
	elif label == 'thumbs_down':
		return '2'

evaluate()