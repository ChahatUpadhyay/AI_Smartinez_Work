# AI_Smartinez_Work
# Jelly fish Classification Model

**1. Data Collection and processing:**
* Collected data from Kaggle where we have 6 different types of jellyfish having 150 of 224*224 pixels each.
   * As images were less in numbers
   * We augmented the data thorugh horizontal flip, little zoom and change in brightness etc all wrapped up in Image Generator funciton.
   * After this we got 600 images per class or types.
   * We have 3600 images total.
     
**2. Loading and training the Model:**
* Sucessfully loaded the VGG16 Model and add custom top 5 layers to make the model more generalize and highly accurate
* we trained the model on augmented data with 20 epochs.
* we got 97% accuracy on test data.

**3. Optimising and Fine Tuning the model:**
* We aslo did the fine tunnning work first by frezzing last 10 layers.
*  used Adama and reduceOnPleatue for best optimization.
*  Also added early-stopping if the test loss does not improving.
*  Got 98 % accuracy

**4. Saving and pretesting model:**
* Saved the fine tuned model in .h5 format.
* Tested the model against unseen images of jelly fish.
* The model predicted the image accurately.

**5. Built App code using python:**
* We used flask to easy and fast maintainance and scalability.
* Also created the structure and template using html and css files.
* Integerated the saved model as back-end to predict user input images.
* Hosted the app locally sucessfully.

**Sample Output of Web-App**
![image](https://github.com/user-attachments/assets/19d226ae-216f-4dea-b969-e6040e636f3b)

