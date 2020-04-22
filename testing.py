import os
import random

training=[]
validation=[]

weights=[]


def sign(i):
    if (i>0):
        return 1
    else:
        return -1

def test(epoch_num):
    out=[]

    for i in range(30):
        out.append([])
        out[i]=[x for x in weights[i]] #edit this later

    file=open(os.getcwd()+"\\outputs\\epoch"+str(epoch_num)+".txt","w")
    for i in range(30):
        for j in range(51):
            file.write(str(out[i][j])+" ")
        file.write("\n")

    file.close()

    correct=0

    for k in range(len(validation)):
        layer1=[0]*30

        for i in range(30):
            for j in range(51):
                layer1[i]+=out[i][j]*validation[k][j]

        layer2=[0]*2

        for i in range(30):
            layer2[i//15]+=sign(layer1[i])

        best=-20
        for i in range(2):
            best=max(layer2[i],best)

        for i in range(2):
            if (best==layer2[i]):
                if (i==validation[k][51]):
                    correct+=1
                break

    print("epoch "+str(epoch_num)+": "+str(correct)+"/"+str(len(validation)))
    

def main():
    global training,validation
    
    file=open("train.txt","r")
    training=[i.split(" ") for i in file.read().split("\n")]
    file.close()
    
    file=open("validation.txt","r")
    validation=[i.split(" ") for i in file.read().split("\n")]
    file.close()

    for i in range(len(training)):
        training[i]=[int(x) for x in training[i]]

    for i in range(len(validation)):
        validation[i]=[int(x) for x in validation[i]]
    
    for i in range(30):
        weights.append([])
        for j in range(51):
            weights[i].append(random.uniform(-1,1)) #initialization

    test(0)

    #for each output node the goal is to reach GOAL or -GOAL depending on answer

    BATCH_SIZE=10000 #make this divisible by 10000
    LEARNING_RATE=0.00000000001
    GOAL=10
    
    for epoch_num in range(1,1000):
        for batch in range(len(training)//BATCH_SIZE):
            cost=0
            
            for k in range(BATCH_SIZE*batch,BATCH_SIZE*(batch+1)):
                out=[0]*30
                for i in range(30):
                    for j in range(51):
                        out[i]+=weights[i][j]*training[k][j]
                
                for i in range(30):
                    if (i//15==training[k][51]): #correct answer
                        cost+=(GOAL-out[i])**2
                    else:
                        cost+=(-GOAL-out[i])**2

            for k in range(BATCH_SIZE*batch,BATCH_SIZE*(batch+1)):
                out=[0]*30
                for i in range(30):
                    for j in range(51):
                        out[i]+=weights[i][j]*training[k][j]

                for i in range(30):
                    for j in range(51):
                        if (i//15==training[k][51]): 
                            weights[i][j]+=cost*(GOAL-out[i])*training[k][j]*LEARNING_RATE
                        else:
                            weights[i][j]+=cost*(-GOAL-out[i])*training[k][j]*LEARNING_RATE

            #print(cost)
            
        test(epoch_num)
        

if __name__=="__main__":
    main()
