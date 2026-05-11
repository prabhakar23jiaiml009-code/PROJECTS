#with open("students.txt","w") as f:
#   f.write("prabhakar")
#   f.write("sushant")
#with open("students.txt","r") as f:
#   data = f.read()
#   print("file content:\n",data)
#with open("students.txt", "a") as f:
#   f.write("yashika\n")
#   f.write("sushant\n")
#with open("student.txt","r") as f:
#   lines = f.readline()
#   print(f"total lines:{len(lines)}") 
#with open("students.txt", "r") as f:
#    text = f.read()
#    words = text.split()
#    print(f"Total Words: {len(words)}") 


import csv
headers = ['id','name','marks']
rows = [
    
    [100, 'prabhakar', 56],
    [102, 'Rahul',     89],
    [103, 'prabhakar', 78],
    [104, 'harish',   100],
    
]

with open("student_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerow(rows)