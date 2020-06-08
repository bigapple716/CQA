from py2neo import Graph
import os
addr = 'http://10.1.1.28:7979'
username = 'neo4j'
password = '123456'

graph = Graph(addr,username = username,password = password)

files = os.listdir()
files = [f for f in files if f.endswith('.txt')]
for file in files:
    name = file.split('.')[0]
    with open(file) as f:
        lines = f.readlines()
        answers = '|'.join(lines[2:])
        update="MATCH (n {{name:\"{0}\"}}) SET n.textqa答案=\"{1}\" RETURN n"
        text = update.format(name,answers)
        print(text)
        graph.run(text)