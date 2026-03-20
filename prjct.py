print ("are you willing to join goa")
answer = input ("yes or no: ")
if answer == "no":
    print ("have a great day")
    exit()
while answer!= "yes":
    print ("pick yes or no")
    answer1 = input ("yes or no: ")
print("you have joined goa academy")

choice = input ("Goa teaches both MMA and programming, which one will you pick: ")

if choice =="MMA":
    print ("you have joined MMA")
    print ("here you will learn wrestling,boxing and judo ")
    exit()
elif choice == "programming":
    print ("here you will learn web development, graphic design and game development")

print ("which one are you going to pick?")
print ("graphic design course")
print ("web development course")
print("or")
print ("game development course")

choice1= input()
if choice1== "web development course":
    print ("youve successfuly joined web development course")
elif choice1== "game development course":
    print ("youve successfuly joined game development course")
elif choice1== "graphic design course":
    print ("youve successfuly joined graphic design course")
print ("pick your plan")
print ("once a week for $100")
print ("twice a week for $200")
plan= input()
while plan != "once a week" and plan!= "twice a week":
    print("options are only once a week and twice a week")
    plan2 = input()
    if plan2 == "once a week":
        print ("you will have lessons only on friday")
        exit()
    if plan2 == "twice a week":
        print ("you will have lessons on monday and friday")
        exit()










