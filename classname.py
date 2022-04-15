import json


p = 'GTSRB/className.json'
with open(p) as json_file:
    data = json.load(json_file)

print("Type:", type(data))
print("People:", data['0'])