# Ted, The Deep-Learning Chatbot

## About this Project
Ted is a multipurpose chatbot made using Python3, who can chat with you and help in performing daily tasks. It uses NLP and Deep-Learning to analyse the user's message, classify it into the a broader category and then reply with a suitable message or the required information. It is hosted using flask and is available on heroku at the link specified above. 

## Project UI
Home Page:

![image](https://raw.githubusercontent.com/Karan-Malik/Chatbot/master/UI/main_screen.PNG?token=AKGUW5C2TMM37OQE5FSPXLS66D55O)

To run it locally on your system, follow these steps:
1. Clone this repository onto your system. On Command Prompt, run the following command:

```
git clone https://github.com/Karan-Malik/Chatbot.git
```
2. Change your directory to Chatbot:
```
cd Chatbot
```
3. Make sure you have all the required libraries listed in requirements.txt. In case any of the libraries are missing, install them using pip. Type this command into your Command Prompt, replacing 'Your-library-name' by the required library name:
```
pip install Your-library-name 
```
4. Then run the follwing commands to run the application:
```
set FLASK_APP=chatbot.py
flask run
```

5. Enter the url provided after running the previous commands into your web browser

Ted is now ready to chat!

#### I would like to thank [Ashutosh Varma](https://github.com/ashutoshvarma) and [Manorit Chawdhry](https://github.com/manorit2001) for their help and contribution to this project. Do check out their Github accounts!

##### Copyright (c) 2020 Karan-Malik

## My Changes in This Fork

This fork is based on **Ted, The Deep-Learning Chatbot**, originally created by [Karan Malik](https://github.com/Karan-Malik). The original MIT License is preserved.

### My change

I added a new `study_tips` intent to `chatbot_codes/intents.json`, with:

- 10 example patterns, such as “Give me study tips” and “Help me make a study plan”
- 3 practical responses about focused study sessions, active practice, and making a simple study plan

### What I am learning from this change

- How a chatbot organizes training examples by intent
- How patterns and responses are stored in JSON
- Why changing the training data means the model must be retrained before it can use the new intent

### Retraining note

The saved model files must be regenerated and tested before the running chatbot can answer the new study-tips patterns. This section will be updated after that test passes.
