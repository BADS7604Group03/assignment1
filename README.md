# BADS7604 Groupwork Assignment 1 : ML vs. MLP
# Member 
1) ณัฐภณ อัศวเหม 6310422052 (% contribution in this homework: 16.67%)
2) วิชิต ชำนาญนาวา 6310422055 (% contribution in this homework: 16.67%)
3) ดวงธิดา แซ่แต้ 6310422056 (% contribution in this homework: 16.67%)
4) เมธี ประเสริฐกิจพันธุ์ 6310422053 (% contribution in this homework: 16.67%)
5) ไตรทิพย์ ศุภศิริวัฒนา 6310422009 (% contribution in this homework: 16.67%)
6) พีรพัทธ ตั้งไพบูลย 6310422024 (% contribution in this homework: 16.67%)

# Data
เราใช้ข้อมูลรายการบัตรเครดิตที่เกิดขึ้นใรยูโรปเมื่อเดือนกันยายน 2013 โดยข้อมูลนี้มีรายการที่ฉ้อโกง(fraud) อยู่ 492 รายการจากทั้งหมด 284,807 รายการ (ข้อมูลมีปัญหา imbalance สูง) และข้อมูลนี้มีตัวแปรด้วยกัน 2 กลุ่มคือ 
1) ตัวแปรที่ผ่านขั้นตอน pca :  V1,V2,V3,...,V28
2) ตัวแปรที่ไม่ผ่านขั้นตอน pca : Time ,Amount 

# Data Preparation
เราใช้ข้อมูลตัวแปร V1,V2,V3,...,V28 และ Amount เพื่อทำนายรายการที่ฉ้อโกง(fraud) โดยก่อนนำข้อมูลไปใช้เราได้ทำการปรับขอบเขตของข้อมูล (Features Scaling) ด้วยวิธี min - max scaling หลังจากนั้นก็แบ่งข้อมูลออกเป็น 2 ชุดคือ 80% traning set สำหรับสร้างโมเดล และ 20% test set สำหรับทดสอบโมเดล โดยเราจะทำการ Over-Sampling ข้อมูลใน traning set ด้วยเพื่อแก้ปัญหา imbalance สูง

# Experiment
เราทำการทดลองสร้างโมเดลทำนายรายการที่ฉ้อโกง(fraud) ด้วยวิธี Deep Learning Multi-Layer Perceptron (MLP) ที่กำหนด loss function เป็น binary crossentropy และ Activation function ที่ output layer เป็น softmax โดยเราออกแบบลักษณะ MLP network ทั้งหมด 3 แบบดังนี้ 

**แบบที่ 1 : Network 1 Hidden Layer**

![image](https://user-images.githubusercontent.com/87576892/152275279-55868795-20b9-4bc9-9b46-bc3af68f2a6f.png)

**แบบที่ 2 : Network 2 Hidden Layer**

![image](https://user-images.githubusercontent.com/87576892/152275245-67e2defb-37c8-476e-af14-2d728e880fa8.png)

**แบบที่ 3 : Network 3 Hidden Layer**

![image](https://user-images.githubusercontent.com/87576892/152275205-82681b9d-1b26-4515-be84-23b878906fe1.png)

และทำการทดสอบปรับค่า Parameter ทั้งหมด 6 ตัวดังนี้
1) Number of neurous in each hidden layer : 1, 28, 56
2) Activation function in hidden layer : relu, sigmoid
3) Optimizer : sgd, adam
4) Learning Rate : 0.001, 0.005
5) Batch size : 256, 521, 1024
6) Number of epoch : 10, 25, 50



# Reference 
https://www.kaggle.com/mlg-ulb/creditcardfraud
https://www.kaggle.com/gpreda/credit-card-fraud-detection-predictive-models
