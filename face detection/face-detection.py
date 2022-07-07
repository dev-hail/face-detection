from cv2 import CascadeClassifier, imread, rectangle, imshow, waitKey, destroyAllWindows
import sys

classifier = CascadeClassifier('./haarcascade_frontalface_default.xml')

image = imread(sys.argv[1])

bounding_box = classifier.detectMultiScale(image)
for box in bounding_box:
	x, y, width, height = box
	x2, y2 = x + width, y + height
	rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

print(f" found {len(bounding_box)} faces. \n press any key to exit")

imshow('face detection', image)
waitKey(0)
destroyAllWindows()