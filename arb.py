'''for i in range(10):
	print(i, end=", ")
def runf(root):
	i = 1
	for path, dirs, files in os.walk(root):
		print path
		print dirs
		print files
		print "---"
		i += 1
		if i >= 4:
	            break
	http://blog.outcome.io/pytorch-quick-start-classifying-an-image/
	https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call-in-python
	http://pillow.readthedocs.io/en/3.4.x/reference/Image.html
	https://stackoverflow.com/questions/17893542/why-do-os-path-isfile-return-false
	http://pythoncentral.io/how-to-sort-a-list-tuple-or-object-with-sorted-in-python/
	https://stackoverflow.com/questions/18424899/print-range-of-numbers-on-same-line
	https://docs.python.org/2/tutorial/inputoutput.html
	https://stackoverflow.com/questions/32796452/python-printing-out-list-separated-with-comma
	https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/
	http://pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/	



root = "/home/george/py_programs/CDATA/notMNIST_small/train"

#	Creates a dictionary for class name to index/label conversion
def class_to_index(root):
	class_list = sorted([directory for directory in os.listdir(root)])
	class_to_labels = {class_list[i]: i for i in range(len(class_list))}
	return class_to_labels

# 	Creates a list of image file path and label pairs
def create_dataset(root, class_to_labels):
	dataset = []
	for label in sorted(class_to_labels.keys()):
		path = os.path.join(root, label)
		for image_file in os.listdir(path):
			image_file = os.path.join(path, image_file)
			if os.path.isfile(image_file):
				dataset.append((image_file, class_to_labels[label]))
	return dataset


'''

#count = 0
#for folder, subfolder, f in os.walk("/home/george/py_programs/CDATA/notMNIST_small"):
#	count += len(f)

my_list = ['apple', 'banana', 'grapes', 'pear']
for count, value in enumerate(my_list, 3):
    print(value)










