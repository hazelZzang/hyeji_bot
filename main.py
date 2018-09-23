from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/keyboard")
def main_key():
	target_json = {
		"type" : "buttons",
		"buttons" : ["자기소개"]
	}
	return jsonify(target_json)

@app.route("/message", methods = ['POST'])
def return_func():
	target_json = {}
	msg_content = request.get_json() # user_key, type, content
	if msg_content["content"].startswith("혜지야"):
		target_json["message"] = {
			"text" : "안뇽 방가루"
		}
	else :
		target_json["message"] = {
			"text" : "한국어를 배우고 있다 가나다라마..바!"
		}

	return jsonify(target_json)

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=7700,debug=True)