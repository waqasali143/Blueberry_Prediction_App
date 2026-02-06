# Blueberry_Prediction_App

 # App Wild Blueberry Yield
#### [My app link](https://blueberry-predict-app.streamlit.app/)
The app is developed using Streamlit.We set the value of the given input parameter, <br>and this app give us the predicted value in the output.
##  Basic Build App Steps
- ### create the virtual environment.
- ### install and import essential library.
- ### import model.
- ### write the app body syntax.
## Note:<br>
### When we app online deploy then very important to correctly write the model location path<br>
### Wrong location pass<br>
Code<br>``` Model_name = pickle.load(open("D:/folder/new folder/w_rfm_model.pkl", 'rb'))```
### Right location pass<br>
Code<br>``` Model_name = pickle.load(open("./w_Xgbm_model.pkl", 'rb'))```
