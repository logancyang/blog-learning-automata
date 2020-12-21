---
toc: true
layout: post
description: Alexa skill development guide
categories: [note, alexa]
title: "How to Setup Local Debugging for Alexa Skills"
comments: true
---

- Make sure you open the root directory of your Alexa skill in the workspace
- Make sure you set the right interpreter in VS Code
- Add the following code to your Lambda function where `...` is your original code:


```python
# Dev
from flask import Flask
from flask_ask_sdk.skill_adapter import SkillAdapter

app = Flask(__name__)
sb = SkillBuilder()

# ...

# Dev
@app.route("/")
def home():
    return "Hello, Flask!"


skill_adapter = SkillAdapter(
                    skill=sb.create(),
                    skill_id=1,
                    app=app)

skill_adapter.register(app=app, route="/")


if __name__ == '__main__':
    app.run()
```


- Use `ngrok` and paste its url in **Endpoint->HTTPS->default region** in Alexa console
- In debugging tab on the left panel, setup the `launch.json` for flask.
- Click the green play button in debug panel to start the flask server in VS Code and start the debugger
- Go to the Alexa Simulator in VS Code's Alexa Skills Toolkit, start the conversation, it will communicate with the Alexa HTTPS endpoint in AWS via the local flask server and ngrok, and enable breakpoints in my code.
