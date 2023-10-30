

### TUTORIAL
# ## Data Types
# x = str("Hello World")

# x = int(20)

# x = float(20.5)

# x = complex(1j)

# x = list(("apple", "banana", "cherry"))

# x = tuple(("apple", "banana", "cherry"))

# x = range(6)

# x = dict(name="John", age = 36)

# x = set(("apple", "banana", "cherry"))

# x = frozenset(("apple", "banana", "cherry"))
             
# x = bool(5)
              
# x = bytes(5)

# x = bytearray(5)

# x = memoryview(bytes(5))

# print(x)


# # Casting

# x = int(1)
# y = int(2.8)
# z = int("3")

# print(x, y, z)


# x = float(1)  
# y = float(2.8)   
# z = float("3")   
# w = float("4.2") 

# print(x, y, z, w)

# x = str("s1")
# y = str(2) 
# z = str(3.0)

# print(x, y, z)


# # Strings

# a = "Hello World"
# print(a[1])

# for x in a:
#     print(x)

# print("Hello" in a)

# a = "Hello, World!"
# print(a.upper())


# a = "Hello, World!"
# print(a.lower())


# a = "Hello, World "
# print(a.strip()) #Remove space

# print(a.replace("H", "J"))

# b = a.split(",")
# print(b)


# txt = "Test Test Test {}"
# age = 29

# print(txt.format(age))


# txt = "Hello, welcome to my Home"
# x = txt.find("Hello")
# print(x)


# txt = "python is FUN!"
# x = txt.capitalize()
# print (x)



# x = 5
# y = 2
# print(x % y)


# x = 5
# x%=3
# print(x)


# # List
# # List items can be of any data type
# myList = ["apple", "banana", "cherry"]
# print(myList)
# print(len(myList))
# print(type(myList))

# list1 = ["abc", 34, True, 40, "male"]
# print(list1)

# list2 = list(("apple", "banana", "cherry")) # note the double round-brackets
# print(list2)
# print(list2[1])
# print(list2[-1])

# if "apple" in list2:
#     print("the apple is existing")

# list2.insert(3, "waterlemon")
# print(list2)


# # To add an item to the end of the list, use the append() method:
# list2.append("orange")
# print(list2)


# # To append elements from another list to the current list, use the extend() method.
# # The extend() method does not have to append lists, you can add any iterable object (tuples, sets, dictionaries etc.).
# list3 = ["mango", "pineapple", "papaya"]
# print("list3", list3)
# list2.extend(list3)
# print(list2)


# list2.remove("banana")
# print(list2)

# list2.pop(2)
# print(list2)

# # If you do not specify the index, the pop() method removes the last item.
# list2.pop()
# print(list2)

# del list3[0]
# print(list3)


# list3.clear()
# print(list3)


# for x in list2:
#     print(x)


# for i in range(len(list2)):
#     print(list2[i])

# [print(x) for x in list2]


# newlist = [x.upper() for x in list2]
# print(newlist)

# newlist = ["apple" for x in list2]
# print(newlist)



# list4 = [100, 50, 20, 70, 10]
# list4.sort()
# print(list4)

# list4.sort(reverse=True)
# print(list4)


# list5 = list2.copy()
# print(list5)


# Tuple
# Tuples are used to store multiple items in a single variable.
# A tuple is a collection which is ordered and unchangeable.

# tupleTest = ("apple", "banana", "cherry")
# print(tupleTest)

# Tuple items are ordered, unchangeable, and allow duplicate values.
# print(len(tupleTest))


# tupleTest = ("apple",)
# print(type(tupleTest))

# #NOT a tuple
# tupleTest = ("apple")
# print(type(tupleTest))

# tupleTest = ("abc", 1234, True, "test", 49)
# print(tupleTest)


# tupleTest = tuple(("apple", "banana", "cherry")) # note the double round-brackets
# print(tupleTest)


# print(tupleTest[1])
# print(tupleTest[0:2])

# Once a tuple is created, you cannot change its values. Tuples are unchangeable, 
# or immutable as it also is called.
# x = ("apple", "banana", "cherry")
# y = list(x)
# y[1] = "kiwi"
# x = tuple(y)
# print(x)

# Since tuples are immutable, they do not have a built-in append() method, 
# but there are other ways to add items to a tuple.
# x = ("apple", "banana", "cherry")
# y = list(x)
# y.append("orange")
# x = tuple(y)
# print(x)


# Add tuple to a tuple. You are allowed to add tuples to tuples, 
# so if you want to add one item, (or many), create a new tuple with the item(s), 
# and add it to the existing tuple
# x = ("apple", "banana", "cherry")
# y =("orrange",)
# x += y
# print(x)


# Unpacking a Tuple
# fruits = ("apple", "banana", "cherry")
# (green, yellow, red) = fruits
# print(green, yellow, red)


# fruits = ("apple", "banana", "cherry", "strawberry", "raspberry")
# (green, yellow, *red) = fruits
# print(green)
# print(yellow)
# print(red)



# # Loop Through a Tuple
# testTuple = ("apple", "banana", "cherry")
# for x in testTuple:
#   print(x)


# # Join Tuples
# tuple1 = ("a", "b" , "c")
# tuple2 = (1, 2, 3)

# tuple3 = tuple1 + tuple2
# print(tuple3)



# # Set
# thisset = {"apple", "banana", "cherry"}
# print(thisset)


# Dioctionarys
# Dictionaries are used to store data values in key:value pairs.
# A dictionary is a collection which is ordered*, changeable and do not allow duplicates.

# testDict = {
#     "brand": "ford",
#     "model": "Mustang",
#     "year": 1964
#     }

# print(testDict["brand"])
# print(testDict)

# Dictionaries are changeable
# Duplicates Not Allowed
# testDict = {
#     "brand": "ford",
#     "model": "Mustang",
#     "year": 1964,
#     "year": 1975
#     }

# print(len(testDict))



# testDict = {
#     "brand": "ford",
#     "model": "Mustang",
#     "year": 1964,
#     "colors": ["red", "black", "white"]
#     }
# print(testDict["colors"][0])


# testDict = dict(name = "Dimitris", age = 30, country = "Norway")
# print(testDict)

# print(testDict.get("name"))

# print(testDict.keys())
# print(testDict.values())
# print(testDict.items())

# Update Item
# testDict["age"] = 29
# print(testDict)


# if "name" in testDict:
#     print("There is exist the name key in the Dictionary Test")


# testDict.update({"country": "Greece"})
# print(testDict)

# Add item
# testDict["color"] = "white"
# print(testDict)

# testDict.update({"color": "Black"})
# print(testDict)

# #Remove item
# testDict.pop("country")
# print(testDict)


# The popitem() method removes the last inserted item (in versions before 3.7, a random item is removed instead):
# testDict.popitem()
# print(testDict)


# del testDict["country"]
# print(testDict)

# del testDict
# print(testDict)


# testDict.clear()
# print(testDict)


# testDict = dict(name = "Dimitris", age = 30, country = "Norway")
# for x in testDict:
#     print(x)
    
# for x in testDict:
#     print(testDict[x])


# for x in testDict.values():
#     print(x)

# for x in testDict.keys():
#     print(x)

# for x,y in testDict.items():
#     print(x, y)


# Copy Dictionaries
# testDict2 = testDict.copy()
# print(testDict2)

# testDict2 = dict(testDict)
# print(testDict2)





# If statement
# a = 12
# b = 11
# if a > b: print("a is greater than b")
# print("A") if a > b else print("B")

# a = 330
# b = 330
# print("A") if a > b else print("=") if a == b else print("B")

# if a == b:
#     pass

# While Loop
# i = 1
# while i < 6:
#   print(i)
#   i += 1



# For loop
# for x in range(2, 6):
#   print(x)

# for x in range(2, 30, 3):
#   print(x)

# adj = ["red", "big", "tasty"]
# fruits = ["apple", "banana", "cherry"]

# for x in adj:
#   for y in fruits:
#     print(x, y)

# for x in [0, 1, 2]:
#   pass


# Functions
# def test_func(*kids):
#     print("The youngest child is " + kids[2])

# test_func("Emil", "Tobias", 'Linus')



# Passing a list
# def my_function(food):
#   for x in food:
#     print(x)

# fruits = ["apple", "banana", "cherry"]

# my_function(fruits)

# function definitions cannot be empty, but if you for some reason have a function definition with no content, put in the pass statement to avoid getting an error.
# def myfunction():
#   pass


# Lambda
# A lambda function is a small anonymous function.
# A lambda function can take any number of arguments, but can only have one expression.
# lambda arguments : expression

# x = lambda a : a + 10
# print(x(5))

# x = lambda a, b : a*b
# print(x(5, 6))


# def myfunc(n):
#   return lambda a : a * n

# mydoubler = myfunc(2)

# print(mydoubler(11))


# Array
# cars = ["Ford", "Volvo", "BMW"]
# x = cars[0]
# print(x)
# print(len(cars))
# cars[1] = "Seat"
# print(cars)

# for x in cars:
#     print(x)

# cars.append("Honda")
# print(cars)

# cars.pop(1)
# print(cars)

# cars.remove("Volvo")
# print(cars)
 

# Add the elements of a list (or any iterable), to the end of the current list
# listCars = ["Mercedes"]
# cars.extend(listCars)
# print(cars)








# An object is essential to work with the class attributes.
# The object is created using the class name. When we create an object of the class, 
# it is called instantiation. The object is also called the instance of a class.

# A constructor is a special method used to create and initialize an object of a class. 
# This method is defined in the class.

# In Python, Object creation is divided into two parts in Object Creation 
# and Object initialization

# Internally, the __new__ is the method that creates the object
# And, using the __init__() method we can implement constructor to initialize the object.

# The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.

# class Person:
#     # constructor
#     def __init__(self, name, sex, profession):
#         self.name = name
#         self.sex = sex
#         self.profession = profession

    
#     def show(self):
#         print('Name:', self.name, 'Sex:', self.sex, 'Profession:', self.profession)

#     def work(self):
#         print(self.name, 'working as a : ', self.profession)




# Dimitris = Person('Dimitris', 'Male', 'Developer')
# Dimitris.show()



# Instance variables: The instance variables are attributes attached to an instance of a class. 
# We define instance variables in the constructor ( the __init__() method of a class).
# Class Variables: A class variable is a variable that is declared inside of class, 
# but outside of any instance method or __init__() method.


# class Student:
#     school_name = 'ABC School'

#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
        
        
        
# student1 = Student("Harry", 12)
# print("Student 1", student1.name, student1.age)


# print("School name:", student1.school_name)



# student2 = Student("Harry", 12)
# print("Student 2", student2.name, student2.age)

# student2.school_name = "DEH School"
# print("School name:", student2.school_name)




# In Object-oriented programming, Inside a Class, we can define the following three types of 
# methods.

# Instance method: Used to access or modify the object state.
# If we use instance variables inside a method, such methods are called instance methods.
# Class method: Used to access or modify the class state. In method implementation, 
# if we use only class variables, then such type of methods we should declare as a class method.
# Static method: It is a general utility method that performs a task in isolation. 
# Inside this method, we don’t use instance or class variable because this static method 
# doesn’t have access to the class attributes.



# class Student:
#     # class variable
#     school_name = 'ABC School'
    
#     # constructor
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
        
#     # instance method
#     def show(self):
#         # access instance variables and class variables
#         print('Student:', self.name, self.age, Student.school_name)

#     # instance method
#     def change_age(self, new_age):
#         self.age = new_age

    
#     # class method
#     @classmethod
#     def modify_school_name(cls, new_name):
#         cls.shool_name = new_name


# # call instance methods
# student1 = Student("Dimitris", 28)
# student1.show()
# student1.change_age(29)


# # call class method
# Student.modify_school_name('XZY School')
# # call instance methods
# student1.show()





# class Fruit:
#     def __init__(self, name, color):
#         self.name = name
#         self.color = color
        
#     def show(self):
#          print("Fruit is", self.name, "and Color is", self.color) 

    
# obj = Fruit("Apple", "Red")
# obj.name = "Straberry"
# obj.show()



# # Deleting Object Properties
# del obj.name

# # Accessing object properties after deleting
# print(obj.name)
# # Output: AttributeError: 'Fruit' object has no attribute 'name'


# # delete object
# del obj




# Inheritance
# Parent class is the class being inherited from, also called base class.

# Child class is the class that inherits from another class, also called derived class.



# class Person:
#     def __init__(self, fname, lname):
#         self.firstname = fname
#         self.lastname = lname
        
#     def printname(self):
#         print(self.firstname, self.lastname)

# x = Person("Dimitris", "Taskoudis")
# x.printname()


# Create a class named Student, which will inherit the properties and methods 
# from the Person class:
# Note: Use the pass keyword when you do not want to add any other properties or methods to the class.
# class Student(Person):
#     pass

# x = Student("Mike", "Olsen")
# x.printname()


# When you add the __init__() function, the child class will no longer inherit 
# the parent's __init__() function.
# To keep the inheritance of the parent's __init__() function, 
# add a call to the parent's __init__() function:
# class Student(Person):
#   def __init__(self, fname, lname):
#     Person.__init__(self, fname, lname)
        

# super() function
# Python also has a super() function that will make the child class inherit 
# all the methods and properties from its parent:
# class Student(Person):
#   def __init__(self, fname, lname, year):
#     super().__init__(fname, lname)
#     self.graduationyear = year

#   def welcome(self):
#     print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)
    
    
# x = Student("Mike", "Olsen", 2019)
# x.welcome()
    
    
# Iterators 
# Lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.
# All these objects have a iter() method which is used to get an iterator:

# mytuple = ("apple", "banana", "cherry")
# myit = iter(mytuple)

# print(next(myit))
# print(next(myit))
# print(next(myit))
    
    
# The __iter__() method acts similar, you can do operations (initializing etc.), but must always return the iterator object itself.
# The __next__() method also allows you to do operations, and must return the next item in the sequence.
# class MyNumbers:
#   def __iter__(self):
#     self.a = 1
#     return self

#   def __next__(self):
#     x = self.a
#     self.a += 1
#     return x

# myclass = MyNumbers()
# myiter = iter(myclass)

# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
    
    
# Inheritance 
# class Car:
#   def __init__(self, brand, model):
#     self.brand = brand
#     self.model = model

#   def move(self):
#     print("Drive!")

# class Boat:
#   def __init__(self, brand, model):
#     self.brand = brand
#     self.model = model

#   def move(self):
#     print("Sail!")

# class Plane:
#   def __init__(self, brand, model):
#     self.brand = brand
#     self.model = model

#   def move(self):
#     print("Fly!")

# car1 = Car("Ford", "Mustang")       #Create a Car class
# boat1 = Boat("Ibiza", "Touring 20") #Create a Boat class
# plane1 = Plane("Boeing", "747")     #Create a Plane class

# for x in (car1, boat1, plane1):
#   x.move()



# A variable is only available from inside the region it is created. This is called scope.
# x = 300
# def myfunc():
#   x = 200
#   print(x)

# myfunc()
# print(x)


# If you need to create a global variable, but are stuck in the local scope, you can use the global keyword.
# x = 300
# def myfunc():
#   global x
#   x = 200

# myfunc()
# print(x)

# Modules
# import Module
# Module.greetings("Dimitris")

# import Module as mx
# a = mx.person1["age"]
# print(a)

# import platform
# x = platform.system()
# print(x)

# import platform
# x = dir(platform)
# print(x)



# Dates
# import datetime
# x = datetime.datetime.now()
# print(x)


# print(x.year)
# print(x.strftime("%A"))

# Create a date
# x = datetime.datetime(2023, 10, 20)
# print(x)

# The datetime object has a method for formatting date objects into readable strings.
# The method is called strftime(), and takes one parameter, format, to specify the format of the returned string:
# x = datetime.datetime(2018, 6, 1)
# print(x.strftime("%B"))



#Maths
# The pow(x, y) function returns the value of x to the power of y (xy).
# x = pow(4,3)  # 4 * 4 * 4
# print(x)


# import math

# x = math.sqrt(64)
# print(x)


# # The math.ceil() method rounds a number upwards to its nearest integer, and the math.floor() method rounds a number downwards to its nearest integer, and returns the result:
# x = math.ceil(1.4)
# y = math.floor(1.4)

# print(x)
# print(y)



# Json
# import json 

# Parse JSON - Convert from JSON to Python
# x =  '{ "name":"John", "age":30, "city":"New York"}'
# y = json.loads(x)
# print(y["age"])


# Convert from Python to JSON
# x = {
#   "name": "John",
#   "age": 30,
#   "city": "New York"
# }
# y = json.dumps(x)
# print(y)



# # RegEx
# import re
# txt = "The rain in Spain"
# # x = re.search("^The.*Spain$", txt)


# x = re.findall("ai", txt)
# print(x)


# x = re.search("\s", txt)
# print(x)


# x = re.split("\s", txt)
# print(x)

# # Try Except
# x = 10
# try:
#     raise Exception("Sorry, no numbers below zero")
#     print(x)
# except:
#     print("Error")





# Files
# The key function for working with files in Python is the open() function.
# The open() function takes two parameters; filename, and mode.
# "r" - Read - Default value. Opens a file for reading, error if the file does not exist
# "a" - Append - Opens a file for appending, creates the file if it does not exist
# "w" - Write - Opens a file for writing, creates the file if it does not exist
# "x" - Create - Creates the specified file, returns an error if the file exists

# "t" - Text - Default value. Text mode
# "b" - Binary - Binary mode (e.g. images)



# file = open("E:/Test.txt", "r")
# print(file.read())



# Return the 5 first characters of the file:
# print(file.read(5))

# Read one line of the file
# print(file.readline())


# Close the file when you are finish with it
# print(file.readline())
# file.close()



# To write to an existing file, you must add a parameter to the open() function:
# "a" - Append - will append to the end of the file
# "w" - Write - will overwrite any existing content

# file = open("E:/Test.txt", "w")
# file.write("I have deleted the content")
# file.close()

# file = open("E:/Test.txt", "r")
# print(file.read())



# Create a New File
# f = open("E:/myfile.txt", "x")



# Delete File
# import os
# os.remove("Test.txt")

# import os
# if os.path.exists("demofile.txt"):
#   os.remove("demofile.txt")
# else:
#   print("The file does not exist")

# Delete Folder
# import os 
# os.rmdir("NameOfFolder")




# import os  
    
# # Function to Get the current   
# # working directory  
# def current_path():  
#     print("Current working directory before")  
#     print(os.getcwd())  
#     print()  
    
    
# # Driver's code  
# # Printing CWD before  
# current_path()  
    
# # Changing the CWD  
# os.chdir('../')  
    
# # Printing CWD after  
# current_path()  

# # Directory  
# directory = "GeeksforGeeks"
  
# # Parent Directory path  
# parent_dir = "D:/Pycharm projects/"
  
# # Path  
# path = os.path.join(parent_dir, directory)  
  
# # Create the directory  
# # 'GeeksForGeeks' in  
# # '/home / User / Documents'  
# os.mkdir(path)  
# print("Directory '% s' created" % directory)  
  
# # Directory  
# directory = "Geeks"
  
# # Parent Directory path  
# parent_dir = "D:/Pycharm projects"
  
# # mode  
# mode = 0o666
  
# # Path  
# path = os.path.join(parent_dir, directory)  
  
# # Create the directory  
# # 'GeeksForGeeks' in  
# # '/home / User / Documents'  
# # with mode 0o666  
# os.mkdir(path, mode)  
# print("Directory '% s' created" % directory)  


# path = "/"
# dir_list = os.listdir(path)  
  
# print("Files and directories in '", path, "' :")  
  
# # print the list  
# print(dir_list)  










#################################
# Pandas
# import pandas as pd

# data = [1, 2, 7]

# Series
# var = pd.Series(data)
# print(var)


# Labels
# var = pd.Series(data, index=["x", "y", "z"])
# print(var)
# print(var["y"])


# # Key/Value Objects as Series
# calories = {"day1": 420, "day2": 380, "day3": 390}
# var = pd.Series(calories)
# print(var)


# calories = {"day1": 420, "day2": 380, "day3": 390}
# myvar = pd.Series(calories, index = ["day1", "day2"])
# print(myvar)




# DataFrames
# data = {
#         "calories": [420, 380, 390],
#         "duration": [50, 40, 45]
#         }
# print(data)
# df = pd.DataFrame(data)
# print(df)


# Locate Row
#refer to the row index:
# print(df.loc[0])


# Return row 0 and 1:
# print(df.loc[[0, 1]])



# #refer to the named index:
# df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
# print(df.loc["day1"])


# Read CSV
# df = pd.read_csv('data.csv')
# use to_string() to print the entire DataFrame.
# print(df.to_string())
# print(df)

# print(pd.options.display.max_rows)



# Read JSON
# df = pd.read_json('data.json')
# print(df)

# Get a quick overview by printing the first 10 rows of the DataFrame:
# print(df.head(10))

# Print the first 5 rows of the DataFrame:
# print(df.head())
# The tail() method returns the headers and a specified number of rows, starting from the bottom.
# print(df.tail())



# Cleaning Empty Cells
# df = pd.read_csv('data.csv')
# print("The initial df data", df.size)

# Return a new Data Frame with no empty cells:
# new_df = df.dropna()
# print("New df", new_df.size)


# Remove all rows with NULL values:
# If you want to change the original DataFrame, use the inplace = True argument:
# df.dropna(inplace=True)
# print(df.size)


# The fillna() method allows us to replace empty cells with a value:
# Replace NULL values with the number 130:
# df.fillna(130, inplace=True)


# Replace NULL values in the "Calories" columns with the number 130:
# df["Calories"].fillna(130, inplace=True)


# Mean = the average value (the sum of all values divided by number of values).
# Calculate the MEAN, and replace any empty values with it:
# x = df["Calories"].mean()
# df["Calories"].fillna(x, inplace=True)


# Median = the value in the middle, after you have sorted all values ascending.
# Calculate the MEDIAN, and replace any empty values with it:
# y = df["Calories"].median()
# df["Calories"].fillna(y, inplace=True)



# Mode = the value that appears most frequently.
# Calculate the MODE, and replace any empty values with it:
# z = df["Calories"].mode()[0]
# df["Calories"].fillna(z, inplace=True)


# Set "Duration" = 45 in row 7:
# df.loc[7, 'Duration'] = 45



# To replace wrong data for larger data sets you can create some rules,
# e.g. set some boundaries for legal values, and replace any values that are outside of the boundaries.
# for x in df.index:
#     if df.loc[x, "Duration"] > 120:
#         df.loc[x, "Duration"] = 120


# for x in df.index:
#   if df.loc[x, "Duration"] > 120:
#     df.drop(x, inplace = True)



# Discovering Duplicates
# print(df.duplicated())

# df.drop_duplicates(inplace=True)



# Finding Relationships
# The corr() method calculates the relationship between each column in your data set.
# print(df.corr())

# The Result of the corr() method is a table with a lot of numbers that represents how well the relationship is between two columns.
# The number varies from -1 to 1.
# 1 means that there is a 1 to 1 relationship (a perfect correlation), and for this data set, each time a value went up in the first column, 
# the other one went up as well.
# 0.9 is also a good relationship, and if you increase one value, the other will probably increase as well.
# -0.9 would be just as good relationship as 0.9, but if you increase one value, the other will probably go down.
# 0.2 means NOT a good relationship, meaning that if one value goes up does not mean that the other will.






# import pandas as pd

# data = [[50, True], [40, False], [30, False]]
# df = pd.DataFrame(data)

# Return the the value of the second [1] row of the first [0] column:
# print(df.iloc[1, 0])


# Definition and Usage
# The iloc property gets, or sets, the value(s) of the specified indexes.

# Specify both row and column with an index.


# Specify columns by including their indexes in another list:
# print(df.iloc[[0,2], [0,1]])

# print(df.iloc[0:2])









