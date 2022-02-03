## Compare between MLP and Traditional-ML


![image](https://user-images.githubusercontent.com/71161635/152391947-2aaa5df9-48aa-49ca-95de-1fdc93ca5d72.png)


•	จากการทดลองจะเห็นว่าถ้าเรา focus ดูที่ AUC score ของ traditional machine learning กับ deep learning เราจะพบว่า ค่าของ AUC score ทุก scenarios ของ deep learning นั้น ดีกว่าทุก scenarios ของ traditional models รวมถึง running time ที่ใช้นั้นใช้เวลาน้อยกว่าของ traditional machine learning ยกเว้น เพียงแค่ scenarios ของ 1 Hidden layer ที่ใช้ running time มากกว่า LightGBM แต่ทว่า AUC score ของ scenarios ของ 1 Hidden layer มีค่ามากกว่า 1.1% (0.9836 > 0.9729234)


•	จากการทดลองจะเห็นว่า performance ของ deep learning นั้นยิ่งมี hidden layers เยอะจะยิ่งมี AUC score ที่ดีขึ้นและใช้ running time น้อยลง
