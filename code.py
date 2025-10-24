import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import pyttsx3

vid = cv2.VideoCapture(2)
hand = mp.solutions.hands
draw = mp.solutions.drawing_utils
model = hand.Hands(static_image_mode=False,min_tracking_confidence=0.8,min_detection_confidence=0.9,max_num_hands=1)

columns = [f"{i}" for i in range(63)] + ["Label"]
df = pd.DataFrame(columns=columns)
registrations = int(input("How many registrations: "))
count = 0

for _ in range(registrations):
    name = input("Label Name: ")

    while True:
        s , frame = vid.read()
        if not s:
            print("Failed to grab frame.")
            break
        frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        output = model.process(frame1)
        
        if output.multi_hand_landmarks:
            draw.draw_landmarks(frame,landmark_list=output.multi_hand_landmarks[0],connections=hand.HAND_CONNECTIONS)
            
        cv2.putText(frame, f"{name}",(50, 50),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Window", frame)

        key = cv2.waitKey(1) & 255
        if key == ord('s'):
            print(f"Starting capture for {name}...")
            samples_collected = 0
            
            while samples_collected < 50:
                ret, frame = vid.read()
                if not ret:
                    break

                frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                output = model.process(frame1)
            
                if output.multi_hand_landmarks:
                    draw.draw_landmarks(frame,landmark_list=output.multi_hand_landmarks[0],connections=hand.HAND_CONNECTIONS)
                                
                    face = []

                    for idx in range(21):
                        lm = output.multi_hand_landmarks[0].landmark[idx]
                        face.extend([lm.x, lm.y, lm.z])
                    face.append(name) 

                    df.loc[count] = face
                    count += 1
                    samples_collected += 1

                    cv2.putText(frame, f"Recording {name}: {samples_collected}/2000",(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                else:
                    cv2.putText(frame, "Hand not detected", (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Window", frame)
                if cv2.waitKey(1) & 255 == ord('q'):
                    break

                time.sleep(0.001)

            print(f"Done collecting 2000 samples for {name}")

        elif key == ord('q'):
            print("Exiting registration loop.")
            break
            
vid.release()
cv2.destroyAllWindows()
import sklearn
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.ensemble import RandomForestClassifier

import optuna
from optuna.samplers import GridSampler,RandomSampler
df = pd.read_csv(r"C:\Users\User\Downloads\Alphabets_Data_.csv")
fv = df.drop('Label',axis=1)
cv = df['Label']
fv.shape , cv.shape
x_train, x_test, y_train, y_test = train_test_split(fv,cv,test_size=0.2,random_state=40,stratify=cv)
# x_train.shape,x_test.shape,y_train.shape,y_test.shape
def objective(trial):
    
    # define the hyper parameters with its range of values
    n_estimators = trial.suggest_int("n_estimators",10,20)
    max_samples = trial.suggest_float("max_samples",0.5,0.8)

    # define algorithm
    rf = RandomForestClassifier(n_estimators=n_estimators,max_samples=max_samples)

    # training by using k-fold
    dict_ = cross_validate(estimator=rf,X=x_train,y=y_train,cv=4,scoring='accuracy',return_train_score= True) 
    cv_acc = dict_['test_score'].mean()
    train_acc = dict_['train_score'].mean()

    # return an any additional values
    trial.set_user_attr('train_acc',train_acc)

    return cv_acc
space = {'n_estimators':range(10,20),'max_samples':[0.5,0.6,0.7,0.8]}
study = optuna.create_study(direction="maximize",sampler=GridSampler(search_space=space))
study.optimize(objective)
study.best_params
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "zira" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
df = pd.read_csv(r"C:\Users\User\Downloads\Alphabets_Data_.csv")
fv = df.drop('Label',axis=1)
cv = df['Label']

l = []
for i in fv.values:
    s = np.array(i).reshape(21,3)
    center = s - s[0]
    dist = np.linalg.norm(s[0] - s[12])
    fpd = center / (dist + 1e-6)
    l.append(fpd.flatten())
fv = l

rf = RandomForestClassifier(n_estimators=18,max_samples=0.8)
rf_model = rf.fit(fv,cv)

hand = mp.solutions.hands
draw = mp.solutions.drawing_utils
model = hand.Hands( static_image_mode=False,
    min_tracking_confidence=0.8,
    min_detection_confidence=0.9,
    max_num_hands=1)  

start_prediction = False  
vid = cv2.VideoCapture(2)
sentence = ""
text = ""
last_prediction_time = 0
prediction_interval = 3
while True:
    s, frame = vid.read()
    if not s:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = model.process(frame_rgb)

    if output.multi_hand_landmarks:
        draw.draw_landmarks(
            frame, 
            landmark_list=output.multi_hand_landmarks[0], 
            connections=hand.HAND_CONNECTIONS)

    key = cv2.waitKey(1) & 255
    current_time = time.time()

    if output.multi_hand_landmarks and (current_time - last_prediction_time) >= prediction_interval:
        last_prediction_time = current_time

        data = []
        for han in output.multi_hand_landmarks:
            for lm in han.landmark:
                data.extend([lm.x, lm.y, lm.z])

        s = np.array(data).reshape(21, 3)
        center = s - s[0]
        dist = np.linalg.norm(s[0] - s[12])
        fpd = center / (dist + 1e-6)

        prediction = rf_model.predict([fpd.flatten()])
        predicted_char = prediction[0]
        print("Prediction:", predicted_char)

        if not start_prediction:
            if predicted_char == "Start":
                start_prediction = True
                engine.say("Prediction Started")
                engine.runAndWait()
        else:
            if predicted_char not in ["Back Space", "Start", "_"]:
                engine.say(" Letter ")
                engine.say(predicted_char)
                engine.runAndWait()
                text += predicted_char
                sentence += predicted_char
            elif predicted_char == "Back Space":
                engine.say(".")
                engine.say("Backspace")
                engine.runAndWait()
                sentence = sentence[:-1]
                text = text[:-1]
            elif predicted_char == "_":
                engine.say(".")
                engine.say("Space")
                engine.say("Word")
                engine.say(text)
                engine.runAndWait()
                sentence += " "
                text = ""
            elif predicted_char == "Start": 
                sentence = sentence.replace("_", " ")
                engine.say("Stop")
                engine.say(str(sentence))
                engine.runAndWait()
                start_prediction = False

    if key == ord('c'):
        sentence = ""
        text = ""
        start_prediction = False
        engine.say(".")
        engine.say("Reset")
        engine.runAndWait()

    cv2.putText(frame, str(sentence), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 10), 1)
    cv2.putText(frame, "Show 'Start' gesture to begin", (10, 440), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 100), 1)
    cv2.putText(frame, "Press 'c' to clear and 'q' to quit", (10, 465), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1)
    cv2.imshow("Hand Gesture to Voice", frame)

    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
