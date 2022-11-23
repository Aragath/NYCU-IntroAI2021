import face_recognition
import os
import cv2
import numpy as np
import sys

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.4
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # 'hog'(on CPU) or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

###########################################################################################

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

############################################################################################

print("loading known faces")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        temp_encoding = face_recognition.face_encodings(image)
        if len(temp_encoding) >0 :
        	encoding = temp_encoding[0]
        else:
        	print("no face found in", name, filename)
        	quit()
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):
    # Load image
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    temp_image = image
    image = cv2.resize(image, (0, 0), None, 0.9, 0.9) # Scaling the image to try to speed up the computation
    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)
    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)
    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        # Returns array of values in order of passed known_faces
        faceDis = face_recognition.face_distance(known_faces, face_encoding)
        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of the least distance
        	matchIndex = np.argmin(faceDis)
        	match = known_names[matchIndex]
        	print(f' - {match} from {results}')
        else:  # If there's none similar faces, name he/she 'unknown'
        	match = str("unknown")
        # print distance of the unknown_faces between each known_faces
        print(faceDis)
        # Each location contains positions in order: top, right, bottom, left
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        # Get color by name using our fancy function
        color = name_to_color(match)
        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
        # Now we need smaller, filled grame below for a name
        # This time we use bottom in both corners - to start from bottom and move 50 pixels down
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)
        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        # Wite a name
        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), FONT_THICKNESS)
    # Show image
    #imageS = cv2.resize(image, (0, 0), None, 0.75, 0.75) # Scaling the image b/c some of the images are too large to fit in the screen
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL) # Let the window be adjustable
    cv2.imshow(filename, image)
    # Overlooking if there's key pressed 
    key = cv2.waitKey(0)
    if key == ord('q') or key == 27: # Esc
        print('halting face_rec')
        sys.exit()
    cv2.destroyWindow(filename)