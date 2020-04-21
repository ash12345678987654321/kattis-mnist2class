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

    for i in range(150):
        out.append([])
        out[i]=[x for x in weights[i]] #edit this later

    file=open(os.getcwd()+"\\outputs\\epoch"+str(epoch_num)+".txt","w")
    for i in range(150):
        for j in range(51):
            file.write(str(out[i][j])+" ")
        file.write("\n")

    file.close()

    correct=0

    for k in range(len(validation)):
        layer1=[0]*150

        for i in range(150):
            for j in range(51):
                layer1[i]+=out[i][j]*validation[k][j]

        layer2=[0]*10

        for i in range(150):
            layer2[i//15]+=sign(layer1[i])

        best=-20
        for i in range(10):
            best=max(layer2[i],best)

        for i in range(10):
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
    
    for i in range(150):
        weights.append([])
        for j in range(51):
            weights[i].append(random.uniform(-1,1)) #initialization

    test(0)

    #for each output node the goal is to reach 15 or -15 depending on answer

    BATCH_SIZE=1000 #make this divisible by 50000
    LEARNING_RATE=0.0000000000001
    THRESHOLD=0.5
    
    for epoch_num in range(1,100):
        for batch in range(len(training)//BATCH_SIZE):
            cost=0
            
            for k in range(BATCH_SIZE*batch,BATCH_SIZE*(batch+1)):
                out=[0]*150
                for i in range(150):
                    for j in range(51):
                        out[i]+=weights[i][j]*training[k][j]
                
                for i in range(150):
                    if (i//15==training[k][51]): #correct answer
                        cost+=(15-out[i])**2
                    else:
                        cost+=(-15-out[i])**2

            for k in range(BATCH_SIZE*batch,BATCH_SIZE*(batch+1)):
                out=[0]*150
                for i in range(150):
                    for j in range(51):
                        out[i]+=weights[i][j]*training[k][j]

                for i in range(150):
                    for j in range(51):
                        if (i//15==training[k][51]): 
                            weights[i][j]+=cost*(15-out[i])*training[k][j]*LEARNING_RATE
                        else:
                            weights[i][j]+=cost*(-15-out[i])*training[k][j]*LEARNING_RATE

            print(cost)
            
        test(epoch_num)
        

if __name__=="__main__":
    main()
