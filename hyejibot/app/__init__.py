from flask import Flask, jsonify, request
from hyejibot.classifier import Classifier
app = Flask(__name__)
c = Classifier()

@app.route("/keyboard")
def main_key():
    target_json = {
        "type" : "buttons",
        "buttons" : ["대화를 시작해봐요!"]
    }
    return jsonify(target_json)


@app.route("/message", methods = ['POST'])
def return_func():
    target_json = {}
    msg_content = request.get_json() # user_key, type, content
    intent = c.classification(msg_content)
    answer = c.get_answer(intent)
    target_json["message"] = {
        "text" : answer
    }
    return jsonify(target_json)
