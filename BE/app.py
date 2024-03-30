from flask import Flask, request, jsonify
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://mongodb:27017/myDatabase"  # Docker service name for MongoDB
mongo = PyMongo(app)

@app.route('/submit', methods=['POST'])
def submit():
    if not request.json or not 'name' in request.json or not 'email' in request.json:
        return jsonify({'error': 'Missing name or email'}), 400

    name = request.json['name']
    email = request.json['email']

    mongo.db.users.insert_one({'name': name, 'email': email})
    return jsonify({'message': 'User added successfully'}), 201

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
