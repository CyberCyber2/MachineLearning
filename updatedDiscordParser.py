#!/usr/bin/python
import os
import argparse
import sys
import json
import re
import itertools


# Concat all files in directory into 1 html file
# Unused
def concatFiles(dir):
	files = os.listdir(dir)
	with open('output_concatFile.txt', 'w') as fOut:
		for infile in files:
			with open(os.path.join(dir, infile)) as fIn:
				for line in fIn:
					fOut.write(line)


def writeToDataset(string, dataset):
	newString = ("".join(c for c in string if ord(c) < 128) + "\n").strip()  # remove non ascii characters
	try:
		if len(newString) > 2 and newString[0] != ":" and newString[-1] != ":":
			dataset.write(newString+"\n")  # writes user and message to dataset file
			#Future: replace non ascii names with default
			#combine lastUser/currUser if user in middle deleted
	except:
		print(newString)
		raise Exception("Error Writing To File")


def contentToAscii(content):
	stripped = (c for c in content if 0 < ord(c) < 127)
	return ''.join(stripped)


def getFirstUsername(path):
	with open(path + '\\' + os.listdir(path)[0], 'r', errors='ignore') as f:
		data = json.load(f)
		for message in data['messages']:
			firstUser = str(message['author']['name'])
			return firstUser

# Extract Names with Text only
def extractData():
	Path = os.getcwd() + '\\input'
	msg_path = Path + '\\messages'
	files = os.listdir(msg_path)
	print(Path)
	print(files)
	lastUser = getFirstUsername(msg_path)
	with open(Path + '\\dataset.txt', 'w') as dataset:
		for x in files:
			with open(msg_path + '\\' + x, 'r', errors='ignore') as \
					json_file:
				data = json.load(json_file)
				totalString = []  # all messages by user at once
				for message in data['messages']:
					currUser = str(message['author']['name'])
					if (currUser != lastUser):
						combinedString = ''.join(totalString)
						writeToDataset((lastUser) + ": " + combinedString, dataset)
						totalString.clear()
					content = str(message['content']).replace("\n", " ")  # user and message
					totalString.append(content + " ")
					lastUser = currUser


###########
# arguments
# parser = argparse.ArgumentParser(description='Discord Parser')
# parser.add_argument('--html_dir', type=str, required=True)
# Mode--->user based(make all text by user into 1 file), server based(train off all all server), feed into next user
# args = parser.parse_args()

# inputDir = args.html_dir
# concatFiles(inputDir)
# extractData(inputDir)
extractData()
