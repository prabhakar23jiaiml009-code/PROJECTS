
#def my_deco(funtion):
#    def wrapper():
#       print("phele chai baad me sb kuch:") 
#       funtion()
#       print("chai ho jaye:") 
#    return wrapper
#@my_deco
#def say_hello():
#    print("hello")            

#say_hello()



#def decorator(funtion):
#    def wrapper():
#        print("or bhai kese hai decorator ek acha topic hai:")
#        funtion()
#        print("kaam ke log kaam ke hote hai :")
#    return wrapper

#@decorator
#def sayhello():
#        print("Hello")
        
#sayhello()        


#def announce(funtion):
#    def pay():
#        print("bhai ye decorator hai :")
#       funtion()
#        print("jai ho.........")
#    return pay
#@announce  
#def greet():
#    print("jai shree shyam")
#greet() 

def res(funtion):
    def peek(*args, **kwargs):
        answer = funtion(*args, **kwargs)
        return answer * 2
    return peek

@res
def add(a, b):
    return a + b
print(add(5, 5))


