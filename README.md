# BADS7604 Groupwork Assignment 1 : ML vs. MLP
# Member 
1) ณัฐภณ อัศวเหม 6310422052 (% contribution in this homework: 16.67%)</br> Hypertuing for 1 layer, write recommendation, Plot Acc/Loss for 1 Layer
2) ดวงธิดา แซ่แต้ 6310422056 (% contribution in this homework: 16.67%)</br> Hypertuing for 1 layer, write summary result for 1 layer, Plot Acc/Loss for 2 Layer
3) เมธี ประเสริฐกิจพันธุ์ 6310422053 (% contribution in this homework: 16.67%)</br>Hypertuing for 2 layer, write comparison Traditional-ML and MLP, Plot Acc/Loss for 3 Layer
4) พีรพัทธ ตั้งไพบูลย 6310422024 (% contribution in this homework: 16.67%)</br>Hypertuing for 2 layer, write summary result for 2 layer, write discussion and conclussion
5) วิชิต ชำนาญนาวา 6310422055 (% contribution in this homework: 16.67%)</br>Hypertuing for 3 layer, write summary result for 3 layer, write reference
6) ไตรทิพย์ ศุภศิริวัฒนา 6310422009 (% contribution in this homework: 16.67%)</br>Preparing a code for experiment, Hypertuing for 3 layer, write experimental design

## Objective 
งานชิ้นนี้มีวัตถุงประสงค์เพื่อทดลองสร้างโมเดลจากข้อมูลชุดหนึ่งด้วยวิธี Traditional-ML และ Multiple layer perceptron (MLP) เพื่อเปรียบเทียบผลโดยทางทีมมีสมมติฐานว่าการใช้ Model แบบ Traditional-ML จะใช้เวลาในการ training น้อยกว่าการสร้างโมเดลด้วยวิธี MLP (Multiple layer perceptron) แต่ก็จะมีความแม่นยำน้อยกว่าด้วยเช่นกัน

## Quick Links
- [x] [Experimental Design Summary](#Experimental-Design-Summary)
- [x] [Compare performance of MLP](#Compare-performance-of-MLP)
- [x] [Compare between MLP and Traditional-ML](#Compare-between-MLP-and-Traditional-ML)
- [x] [Conclusion](#Conclusion)
- [x] [Reference](#Reference)

## Experimental Design Summary
### Data
เราใช้ข้อมูลรายการบัตรเครดิตที่เกิดขึ้นใรยูโรปเมื่อเดือนกันยายน 2013 โดยข้อมูลนี้มีรายการที่ฉ้อโกง(fraud) อยู่ 492 รายการจากทั้งหมด 284,807 รายการ (ข้อมูลมีปัญหา imbalance สูง) และข้อมูลนี้มีตัวแปรด้วยกัน 2 กลุ่มคือ 
1) ตัวแปรที่ผ่านขั้นตอน pca :  V1,V2,V3,...,V28
2) ตัวแปรที่ไม่ผ่านขั้นตอน pca : Time ,Amount 

### Data Preparation
เราใช้ข้อมูลตัวแปร V1,V2,V3,...,V28 และ Amount เพื่อทำนายรายการที่ฉ้อโกง(fraud) โดยก่อนนำข้อมูลไปใช้เราได้ทำการปรับขอบเขตของข้อมูล (Features Scaling) ด้วยวิธี min - max scaling หลังจากนั้นก็แบ่งข้อมูลออกเป็น 2 ชุดคือ 80% traning set สำหรับสร้างโมเดล และ 20% test set สำหรับทดสอบโมเดล โดยเราจะทำการ Over-Sampling ข้อมูลด้วยเทคนิค SMOTE ใน traning set ด้วยเพื่อแก้ปัญหา imbalance สูง

### Network Architecture Design
เราทำการทดลองสร้างโมเดลทำนายรายการที่ฉ้อโกง(fraud) ด้วยวิธี Deep Learning Multi-Layer Perceptron (MLP) ที่กำหนด loss function เป็น binary crossentropy และ Activation function ที่ output layer เป็น softmax โดยเราออกแบบลักษณะ MLP network ทั้งหมด 3 แบบดังนี้ 

**แบบที่ 1 : Network 1 Hidden Layer**

![image](https://user-images.githubusercontent.com/87576892/152275279-55868795-20b9-4bc9-9b46-bc3af68f2a6f.png)

**แบบที่ 2 : Network 2 Hidden Layer**

![image](https://user-images.githubusercontent.com/87576892/152275245-67e2defb-37c8-476e-af14-2d728e880fa8.png)

**แบบที่ 3 : Network 3 Hidden Layer**

![image](https://user-images.githubusercontent.com/87576892/152275205-82681b9d-1b26-4515-be84-23b878906fe1.png)

### Parameter tuning 
ทางทีมทำการทดสอบปรับค่า Parameter ทั้งหมด 6 ตัวดังนี้
**1) Number of neurous in each hidden layer** : 1, 28, 56

**2) Activation function in hidden layer** : relu, sigmoid

**3) Optimizer** : sgd, adam

**4) Learning Rate** : 0.001, 0.005

**5) Batch size** : 256, 521, 1024

**6) Number of epoch** : 10, 25, 50

ทั้งนี้ในการทดลองทางทีมได้กำหนดในจำนวน neurous หรือ node ในแต่ละ Layer มีจำนวนเท่ากันทั้งหมดเช่นที่การทดลอง 2 layer จำนวน 28 node หมายถึงมีจำนวน node ที่ hidden layer 1 จำนวน 28 node และที่ hidden layer 2 จำนวน 28 node เป็นต้น

### Training
โดยในการทดลองจะมีการกำหนด validation set ไว้ 20% (validation_split=0.2) และจะมีการคำนวณค่า auc,	f1score,	precision,	recall,	loss,	accuracy ไว้เพื่อใช้ในการเปรียบเทียบกับโมเดลอื่นๆ 

## Compare performance of MLP
## เปรียบเทียบผลลัพธ์จากการทดลองด้วย Hidden layers จำนวน 1 layers

![image](https://user-images.githubusercontent.com/87576892/152378755-2f06a96c-ec56-4d3a-a9a9-e41394123fb1.png)

•	เมื่อดูจาก AUC พบว่าผลการทดสอบที่ดีที่สุดได้ค่า AUC = 0.9827 โดยมีทั้งหมด 512 batch size, 25 epoch ใช้ learning rate 0.001 มีจำนวน node = 1 ใช้วิธี ADAM ในการ Optimize และใช้เวลาในการรันทั้งหมด 45.5192 วินาที 

•	Epoch: จากผลการทดลองพบว่าการTrain ด้วยจำนวน Epoch ที่ 5, 10 และ 15 ให้ผลลัพธ์ค่า AUC เฉลี่ยที่ไม่แตกต่างกันมาก แม้ว่าจำนวน 25 Epoch จะทำให้ได้ AUC เฉลี่ยสูงสุดที่ 0.943558333 และจำนวน 15 Epoch จะได้ค่า Loss เฉลี่ยน้อยที่สุดที่ 0.1135383 

•	Learning rate: จากผลการทดลองพบว่าการ Train ด้วย Learning rate เท่ากับ 0.005 สามาถทำค่าเฉลี่ย AUC และเวลาการ Train ได้ดีกว่าการTrain ด้วย Learning rate เท่ากับ 0.001 เล็กน้อย 

•	Optimizer: จากผลการทดลองพบว่าการใช้ Optimizer เป็น sgd ได้ผลเฉลี่ยทั้ง AUC และ loss ดีกว่าการใช้ Optimizer เป็น adam ค่อนข้างมาก โดย sgd ได้ AUC เฉลี่ย 0.934502963 และ Loss เฉลี่ย 0.11341467 ในขณะที่ adam ได้ Loss เฉลี่ยสูงถึง 0.336807 และได้ AUC เฉลี่ยเพียง 0.922517568 เท่านั้น

•	Num_node: จากผลการทดลอง พบว่าจำนวน Node ที่ 26 ให้ค่าเฉลี่ย AUC ที่สูงที่สุดที่ 0.971134167 ในขณะที่จำนวน 1 Node ให้ค่า AUC ที่ต่ำที่สุด

•	Activation: จากผลการทดลองพบว่า การใช้ Sigmoid ให้ค่าเฉลี่ย AUC สูงที่สุดที่ 0.970262202 แต่มีค่า average loss ต่ำกว่าการใช้ Relu ในขณะที่ running times สูงกว่าเล็กน้อย

## เปรียบเทียบผลลัพธ์จากการทดลองด้วย Hidden layers จำนวน 2 layers

![Table1](https://user-images.githubusercontent.com/87868128/152371396-17d5e6a9-90fa-40d6-859b-531735fc89ce.png)
- Batch size: จากผลการทดลองพบว่า Batch size จำนวน 256 สามารถทำ AUC เฉลี่ยได้สูงสุดที่ *0.948821817* และมีค่า Loss เฉลี่ยน้อยที่สุดเท่ากับ *0.114028904*
- Epoch: จากผลการทดลองพบว่าการ Train ด้วยจำนวน Epoch ที่ 10, 25 และ 50 ให้ผลลัพธ์ค่า AUC เฉลี่ยที่ไม่แตกต่างกันมาก แม้ว่าจำนวน 25 Epoch จะทำให้ได้ AUC เฉลี่ยสูงสุดที่ *0.934642138* และจำนวน 50 Epoch จะได้ค่า Loss เฉลี่ยน้อยที่สุดที่ *0.117220471* แต่การ Train ด้วยจำนวนเพียง 10 Epoch ก็สามารถทำค่าเฉลี่ย AUC และ Loss ได้ใกล้เคียงกับจำนวน 25 และ 50 Epoch ที่ *0.933079333* และ *0.134150465* ในขณะที่ใช้เวลาเฉลี่ยในการ Train น้อยกว่า 1-2 เท่า
- Learning rate: จากผลการทดลองพบว่าการ Train ด้วย Learning rate เท่ากับ 0.001 สามาถทำค่าเฉลี่ย AUC, Loss และเวลาการ Train ได้ดีกว่าการ Train ด้วย Learning rate เท่ากับ 0.005 เพียงเล็กน้อย โดยได้ AUC เฉลี่ย *0.935565655* และ Loss เฉลี่ย *0.118655164*
- Optimizer: จากผลการทดลองพบว่าการใช้ Optimizer เป็น sgd ได้ผลเฉลี่ยทั้ง AUC, Loss และเวลาการ Train ดีกว่าการใช้ Optimizer เป็น adum ค่อนข้างมาก โดย sgd ได้ AUC เฉลี่ย *0.940881566* และ Loss เฉลี่ย *0.095590263* ในขณะที่ adum ได้ Loss เฉลี่ยสูงถึง *0.20048599* และได้ AUC เฉลี่ยเพียง *0.926370741* เท่านั้น
- Number of nodes: จากผลการทดลองพบว่าการใช้จำนวน nodes ใน Hidden layer เท่ากับ 28 และ 56 ทำให้ได้ค่าเฉลี่ย AUC และ Loss ดีกว่าจำนวน nodes เท่ากับ 1 อย่างชัดเจนแม้ว่าจำนวน nodes เท่ากับ 56 จะทำให้ได้ AUC และ Loss เฉลี่ยดีกว่าที่ *0.963544647* และ *0.0452666192* แต่จำนวน nodes เท่ากับ 28 ก็สามารถทำได้ใกล้เคียงกันที่ *0.963367028* และ *0.052666192* ในขณะที่ใช้เวลาในการ Train น้อยกว่าประมาณ 10%
- Activation: จากผลการทดลองพบว่าการใช้ Activation เป็น sigmoid ให้ค่าเฉลี่ย AUC ดีกว่าการใช้ Activation เป็น relu อย่างมากโดย sigmoid ได้ค่าเฉลี่ย AUC ที่ *0.968878344* ในขณะที่ relu ได้เพียง *0.898373962* เท่านั้น ในขณะที่ใช้ระยะเวลาในการ Train เท่าๆ กัน

### สรุปผลการทดลอง
- จากผลการทดลองจำนวน 216 ผลการทดลองพบว่า การทดลองที่ให้ผล AUC, Loss และระยะเวลาการ Train ดีที่สุดเป็นดังนี้
- AUC มากที่สุดเท่ากับ *0.982462277*

![Table2](https://user-images.githubusercontent.com/87868128/152372337-40d6b979-d019-40fb-988b-de4e350e3b6e.png)
- Loss น้อยที่สุดเท่ากับ *0.008670322*

![Table3](https://user-images.githubusercontent.com/87868128/152372409-108da93f-1552-48b1-8151-f4b1ed178742.png)
- ระยะเวลาในการ Train น้อยที่สุดเท่ากับ *16.52333045* วินาที

![Table4](https://user-images.githubusercontent.com/87868128/152372450-912a816f-cd2d-4ef4-929f-639fc3f1c196.png)

## เปรียบเทียบผลลัพธ์จากการทดลองด้วย Hidden layers จำนวน 3 layers
![image](https://user-images.githubusercontent.com/83268624/152385817-d8c6a0c5-f552-4034-b867-26c16e4e5a05.png)
 Batch size: พบว่า ยิ่ง Batch ยิ่งมาก ค่า Loss % และ AUC จะไม่ต่างกันมากแต่พบว่า ที่ Batch size ที่สูงขึ้นจะทำให้เวลาเฉลี่ยในการคำนวณลดลงอย่างเห็นได้ชัดอีกทั้งโดยที่ Batch Size 1024 จะให้ค่าเฉลี่ยของ % AUC สูงสุดที่ 93% ค่า Loss เฉลี่ยที่ 0.170 และเวลาในการประมวลผลเฉลี่ยของการ Train อยู่ที่ 43.127 วินาทีซึ่งเป็นเวลาที่น้อยที่สุด
- Epoch: จากการวิเคราะห์พบว่า จำนวน Epoch มี   % AUC และ Loss ที่ไม่ต่างกัน แต่ Running Time ที่ใช้กลับมากขึ้นเมื่อเพิ่มจำนวณ Epoch โดยที่  25 Epoch จะให้ % AUC สูงสุดที่ 93 % ในขณะที่ 10 Epoch ให้ค่าเฉลี่ย %AUC ที่ 91.7 % แต่ค่าเฉลี่ยเวลาในการประมวลผลการ Train อยู่ที่ 42.966 วินาทีซึ่งเร็วกว่า  25 Epoch อยู่ถึง 2.3 เท่า
- Learning rate: % AUC และ Loss,  Running Time ที่ไม่ต่างกัน แต่ในการทดลองพบว่าที่ ใช้ Learning 0.001 จะให้ % AUC 92.1%   และค่า Loss 0.155 ซึ่งดีกว่า Learning Rate ที่0.005
- Optimizer: จากผลการทดลองพบว่าการใช้ Optimizer เป็น sgd ได้ผลเฉลี่ยทั้ง AUC, Loss และเวลาการ Train ดีกว่าการใช้ Optimizer เป็น adum ค่อนข้างมาก โดย sgd ได้ AUC เฉลี่ย 92.5 % และ Loss เฉลี่ย 0.289 ในขณะที่ adum มี % AUC เฉลี่ยเพียง 91.13%  และ Loss ที่ 0.289
- Number of nodes: จากผลการทดลองพบว่าการใช้จำนวน nodes ใน Hidden layer เท่ากับ 28 และ 56 ทำให้ ได้ค่าเฉลี่ย% AUC และ Loss ดีกว่าจำนวน nodes เท่ากับ 1 อย่างชัดเจนแม้ว่าจานวน nodes เท่ากับ 56 จะทำให้ได้ AUC และ Loss เฉลี่ยดีกว่าที่ 96.3 % และ 0.029 ในขณะที่จำนวน nodes เท่ากับ 28 ก็สามารถทำได้ใกล้เคียงกันที่ 96.1% และ 0.037 
- Activation: จากผลการทดลองพบว่าการใช้ Activation เป็น sigmoid ให้ค่าเฉลี่ย AUC ดีกว่าการใช้ Activation เป็น relu อย่างมากโดย sigmoid ได้ค่าเฉลี่ย AUC ที่ 96.7 % ในขณะที่ relu ได้เพียง 87.5% เท่านั้น 
จากผลการทดลองจานวน 216 ผลการทดลองพบว่า การทดลองที่ให้ผล AUC, Loss และระยะเวลาการ Train ดี ที่สุดเป็นดังนี้
![image](https://user-images.githubusercontent.com/83268624/152386737-dc0e9534-31fc-4841-9c53-6414b87c9e61.png)

## Compare between MLP and Traditional-ML
![image](https://user-images.githubusercontent.com/71161635/152391947-2aaa5df9-48aa-49ca-95de-1fdc93ca5d72.png)

•	จากการทดลองจะเห็นว่าถ้าเรา focus ดูที่ AUC score ของ traditional machine learning กับ deep learning เราจะพบว่า ค่าของ AUC score ทุก scenarios ของ deep learning นั้น ดีกว่าทุก scenarios ของ traditional models รวมถึง running time ที่ใช้นั้นใช้เวลาน้อยกว่าของ traditional machine learning ยกเว้น เพียงแค่ scenarios ของ 1 Hidden layer ที่ใช้ running time มากกว่า LightGBM แต่ทว่า AUC score ของ scenarios ของ 1 Hidden layer มีค่ามากกว่า 1.1% (0.9836 > 0.9729234)
•	จากการทดลองจะเห็นว่า performance ของ deep learning นั้นยิ่งมี hidden layers เยอะจะยิ่งมี AUC score ที่ดีขึ้นและใช้ running time น้อยลง

## Conclusion
### Discussion
**จากการทดลองทางกลุ่มมีข้อคิดเห็นตามสมมติฐานที่ได้ตั้งขึ้น ดังนี้**  
การใช้ Model ทั้งแบบ Traditional-ML และ MLP สามารถให้ผลความแม่นยำ (AUC Score) ในการทำนายได้ไม่แตกต่างกันอย่างมีนัยสำคัญ แต่ระยะเวลาในการ Train ให้ได้ Model ที่สามารถทำนายได้แม่นยำนั้นแตกต่างกันอย่างมีนัยสำคัญ เนื่องจาก MLP Model มีค่า Parameter ที่ต้องกำหนดอยู่หลายค่า การจะกำหนด Parameter ที่เหมาะสมที่สุดสำหรับ Model ให้ได้ในครั้งเดียวเป็นเรื่องที่ทำได้ยาก ทางกลุ่มจึงต้องมีการทดลองซ้ำหลายๆ ครั้งรวมทั้งหมดมากกว่า 600 ครั้ง และใช้เวลารวมมากกว่า 17 ชั่วโมง ในการทดสอบทุกความเป็นไปได้สำหรับ Parameter ที่ทางกลุ่มคัดเลือกมา เพื่อให้ได้ชุด Parameter ที่ทำนายได้แม่นยำที่สุดเพียงชุดเดียว ในขณะที่ใน Traditional-ML Model ทางกลุ่มใช้เพียงค่า Parameter เริ่มต้นไม่มีการปรับปรุงค่า Parameter (Fine-Tuning) แต่อย่างใด การ Train ใช้ระยะเวลาน้อยกว่า MLP Model หลายเท่า ก็สามารถได้ค่าความแม่นยำที่เกือบใกล้เคียงกับ MLP Model ถึงแม้จะน้อยกว่าแต่ก็ให้ผลลัพธ์ในระดับที่น่าพึงพอใจ ทางกลุ่มจึงสรุปผลการทดลองในครั้งนี้ว่า หากชุดข้อมูลเป็น Structure เช่นเดียวกับในการทดลองนี้ และมี Features ไม่มาก การใช้ Traditional-ML ก็เพียงพอให้ได้ผลในการทำนายที่แม่นยำ อีกทั้งยังใช้เวลาและทรัพยากรในการ Train ที่น้อยกว่าการใช้ MLP Model เป็นอย่างมาก
### Recommend MLP VS Traditional-ML
จากการทดลอง สรุปได้ว่าใน Datasets ที่นำมาทดลองนี้ การใช้ Deep Learning สามารถให้ผลที่เเม่นยำกว่า (เมื่อเทียบจากการใช้ AUC Score, F1-Score เป็นหลัก)
โดยส่วนหนึงมาจากการทดลง fine-tune ค่า parameter ของโมเดลจนได้โมเดลที่ให้ผลลัพท์ที่ดีที่สุดในเเต่ละ Case (1,2 เเละ 3 layers)
เเต่หากพิจารณาจากตัวเเบบของ Traditional ML ที่เรานำมาใช้ พบว่าความเเม่นยำก็อยู่ในระดับที่ค่อนข้างน่าพอใจ โดยหากทำการ Fine-tune parameter ของ
Traditional ML มาใช้ดีๆเเล้ว ก็อาจทำให้โมเดลมีความเเม่นยำขึ้นได้อีก
### Pros and Cons of MLP versus Traditional ML
จากการทดลอง เราพอจะสรุปได้ว่า ประโยชน์ของ MLP หรือ Deep Learning นั้น สามารถนำมาใช้งาน Classification ในเเบบที่ Traditional ML สามารถนำไปทำได้
โดยจุดได้เปรียบของ Deep Learning Algorithm อยู่ที่การจัดการข้อมูลที่มี Feature ค่อนข้างมาก เเละมีขนาดข้อมูลที่ใหญ่ ซึ่งจะให้ประสิทธิภาพที่สูงกว่า Traditional ML
เเต่หากข้อมูลมี Features ที่ไม่ได้เยอะ หรือ ซับซ้อนมาก การใช้ Traditional ML ก็เป็นตัวเลือกที่ไม่ได้เเย่เเต่อย่างใด รวมถึงการอธิบาย ที่มา-ที่ไปของ Model นั้น
ในส่วนของ Traditional ML Model นั้นอาจจะสามารถอธิบายได้ง่ายกว่าของ MLP 

## Reference
- https://www.kaggle.com/mlg-ulb/creditcardfraud
- https://www.kaggle.com/gpreda/credit-card-fraud-detection-predictive-models
## Reference: Library ที่สำคัญที่ใช้ใน Project ได้แก่ 
   lightgbm : เป็น Library ที่ใช้สำหรับสร้าง Traditional ML แบบ LightGBM
   
   
   seaborn  : เป็น Library ใช้สำหรับแสดงผลเชิง visualization พวกการฟต่างๆ
   
   
   xgboost  : เป็น Library ใช้สำหรับสร้าง Traditional ML (ensemble) แบบ xgboost
   
   
   sklearn  : เป็น Libraryที่ใช้งานหลากหลายมาก เริ่มตั้งแต่การจัดแบ่ง data สำหรับ train และ test ,ทำ cross validation ,สร้าง model ,วัดประสิทธิภาพ model  etc.
   
   
   numpy    : เป็น Libraryที่สำคัญมากที่ใช้ในจัดการและคำนวณข้อมูลแบบ Array ในหลายๆมิติ
   
   
   imblearn : เรานำ Libraryนี้มาช่วยในการจัดการเรื่อง Imbalance data 
   
   
   matplotlib : เป็น Libraryที่ใช้สำหรับการแสดงผลเชิง visualization จำพวกกราฟและเส้นต่างๆ
   
   
   tensorflow : เป็น Library สำคัญสำหรับการสร้าง Model ทางด้าน MLP 
   
   
   pandas   : เป็น Library หนึ่งที่สำคัญสำหรับใช้จัดการข้อมูลแบบ Array ในหลายๆมิติ
   
   
   ![image](https://user-images.githubusercontent.com/83268624/152651097-61320dae-7309-4272-a43f-4c55f3506b4d.png)

