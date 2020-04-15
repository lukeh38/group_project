def m(filename,arr):
	txt=[]
	name = filename[0:-4]+'.txt'
	height = 720
	width = 1280
	x1,y1,x2,y2= converter(height,width,arr)
	l = label_to_num(arr)

	txt.append([name,l,x1,y1,x2,y2])
	for i in range(len(txt)):
	  f = open(name,"w")
	  f.write(' '.join(repr(e) for e in txt[i][1:]))
	  f.close()


def label_to_num(arr):
  if arr[0] == 'hand':
    return 0
  elif arr[0] == 'thumbs_up':
    return 1
  elif arr[0] == 'thumbs_down':
    return 2

def converter(height,width, arr):
	xmin = arr[1]
	ymin = arr[2]
	xmax = arr[3]
	ymax = arr[4]
	x = (float((xmin + xmax)) / 2) / width
	y = (float((ymin + ymax)) / 2) / height

	w = float((xmax - xmin)) / width
	h = float((ymax - ymin)) / height
	return x,y,w,h

