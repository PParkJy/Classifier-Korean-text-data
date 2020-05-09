import os, re, glob
import cv2
import numpy as np
from keras.models import model_from_json
import smtplib
import json
from sklearn.metrics import classification_report, confusion_matrix
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
'''
jsonPath = "../../Results/experiment2/base/_json/4_base.json"
weightsPath = "../../Results/experiment2/base/_h5/4_base.h5"

categories = ["bacteria", "healthy", "lateblight", "targetspot", "yellowleafcurl"]

with open(jsonPath,'r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(weightsPath)
print("loaded model from disk")

def Dataization(img_path):
    image_w = 32
    image_h = 32
    img=cv2.imread(img_path)
    return (img/256)

src=[]
name=[]
test_X=[]
test_Y=[]
image_dir= "../../Dataset/DCGAN_4000/bacteria/"


label = [1,0,0,0,0]
image_dir = "../../Dataset/DCGAN_4000/bacteria/"
for top_train, dir_train, f_train in os.walk(image_dir):
    for filename in f_train:
        img = cv2.imread(image_dir + filename)
        test_X.append(img / 256)
        test_Y.append(label)

test_X = np.array(test_X)
test_Y = np.array(test_Y)

test_predictions = model.predict(test_X)
test_predictions = np.round(test_predictions)
accuracy = accuracy_score(test_Y, test_predictions)
confmat = confusion_matrix(test_Y.argmax(axis=1), test_predictions.argmax(axis=1))  # argmax 미사용시 에러
model.summary()

print(confmat)
print(classification_report(test_Y.argmax(axis=1), test_predictions.argmax(axis=1),
                            target_names=["bacteria", "healthy", "lateblight", "targetspot", "yellowleafcurl"]))
f = open('../../Results/experiment2/DCGAN_4000/detected.txt', 'a')
f.write('------------------' + str(i) + 'DCGAN result------------------\n')
f.write(str(confmat) + '\n')
f.write(str(classification_report(test_Y.argmax(axis=1), test_predictions.argmax(axis=1),
                                  target_names=["bacteria", "healthy", "lateblight", "targetspot",
                                                "yellowleafcurl"]) + '\n'))
for i in range(len(categories)):
    axis_sum = 0
    for j in range(len(categories)):
        axis_sum = axis_sum + confmat[i, j]
    answer = confmat[i, i] / axis_sum
    answer = str(answer)
    print(categories[i] + " accuracy : ", end='')
    print(answer)
    f.write(str(categories[i] + " accuracy : " + answer + '\n'))
f.close()

predict = model.predict_classes(test)

cnt_bacteria = 0
cnt_healthy = 0
cnt_lateblight = 0
cnt_targetspot = 0
cnt_yellowleafcurl = 0


for i in range(len(test)):
    #print(name[i] + " : , predict: "+str(categories[predict[i]]))
    if str(categories[predict[i]]) == 'healthy':
        cnt_healthy += 1
    elif str(categories[predict[i]]) == 'bacteria':
        cnt_bacteria += 1

#f = open('.txt','w')
print("test: ",len(test))
print("Healthy: ",cnt_healthy)
print(disease, cnt_bacteria)
print("accuracy: ",len(test)/cnt_bacteria)

#f.write()
f.close()

'''
import os, re, glob
import cv2
import numpy as np
from keras.models import model_from_json
import smtplib
import json
from sklearn.metrics import classification_report, confusion_matrix

disease = input('disease: ')
jsonPath = "../../Results/experiment2/base/_json/4_base.json"
weightsPath = "../../Results/experiment2/base/_h5/4_base.h5"

categories = ["bacteria", "healthy", "lateblight", "targetspot", "yellowleafcurl"]

with open(jsonPath,'r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(weightsPath)
print("loaded model from disk")

def Dataization(img_path):
    image_w = 32
    image_h = 32
    img=cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
    return (img/256)

src=[]
name=[]
test=[]

image_dir = "../../Dataset/GAU_DCGAN/"+disease+"_1/" #"../../Dataset/DCGAN_4000/"+disease+"/"

for file in os.listdir(image_dir):
    if (file.find('.png') is not -1):
        src.append(image_dir+file)
        name.append(file)
        test.append(Dataization(image_dir+file))

test = np.array(test)
predict = model.predict_classes(test)

cnt_bacteria = 0
cnt_healthy = 0
cnt_lateblight = 0
cnt_targetspot = 0
cnt_yellowleafcurl = 0

for i in range(len(test)):
    print(name[i] + " : , predict: "+str(categories[predict[i]]))
    if str(categories[predict[i]]) == 'healthy':
        cnt_healthy += 1
    elif str(categories[predict[i]]) == 'bacteria':
        cnt_bacteria += 1
    elif str(categories[predict[i]]) == 'lateblight':
        cnt_lateblight += 1
    elif str(categories[predict[i]]) == 'targetspot':
        cnt_targetspot += 1
    elif str(categories[predict[i]]) == 'yellowleafcurl':
        cnt_yellowleafcurl += 1

f = open('../../Results/experiment2/256_DCGAN/_result/'+disease+'_GAN_check.txt','a')
cnt = 0
if disease == "bacteria":
    cnt = cnt_bacteria
elif disease == "lateblight":
    cnt = cnt_lateblight
elif disease == "targetspot":
    cnt = cnt_targetspot
elif disease == "yellowleafcurl":
    cnt = cnt_yellowleafcurl
print("test: ",len(test))
print("Healthy: ",cnt_healthy)
print("bacteria: ", cnt_bacteria)
print("lateblight: ", cnt_lateblight)
print("targetspot: ", cnt_targetspot)
print("yellowleafcurl: ", cnt_yellowleafcurl)
print(disease+" precision: ",cnt/cnt)
print(disease+" recall: ",cnt/len(test))
print(disease+" f1-score: ",2*((cnt/cnt) * (cnt/len(test)))/((cnt/cnt)+(cnt/len(test))))
print(disease+" accuracy: ",cnt/len(test))
f.write("test: " + str(len(test))+'\n')
f.write("Healthy: "+str(cnt_healthy)+'\n')
f.write("bacteria: "+ str(cnt_bacteria)+'\n')
f.write("lateblight: "+ str(cnt_lateblight)+'\n')
f.write("targetspot: "+ str(cnt_targetspot)+'\n')
f.write("yellowleafcurl: "+ str(cnt_yellowleafcurl)+'\n')
f.write(disease+" precision: "+str(cnt/cnt)+'\n')
f.write(disease+" recall: "+str(cnt/len(test))+'\n')
f.write(disease+" f1-score: "+str(2*((cnt/cnt) * (cnt/len(test)))/((cnt/cnt)+(cnt/len(test))))+'\n')
f.write(disease+" accuracy: "+str(cnt/len(test))+'\n\n')
f.close()
