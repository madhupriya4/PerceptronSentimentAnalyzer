import json,sys

#get the model parameters from the trained output
def getInput(fileName):
    wordtext = open(fileName, "r").read().split("\n")

    weights = json.loads(wordtext[0])

    bias1=float(wordtext[1])
    bias2=float(wordtext[2])

    return weights,bias1,bias2

#classifies the test lines into two classes using two binary classifications
def classify(fileName, weights,bias1,bias2):
    outputFile = open("percepoutput.txt", "w")

    inputFile = open(fileName).read()

    #remove punctuations
    puncList = [".", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "$", "&", ")", "(", "\""]
    for punc in puncList:
        inputFile = inputFile.replace(punc, '')

    fileLines = inputFile.split("\n")

    #iterate for each line
    for line in fileLines:
        if line == "":
            continue

        key = line.split(" ")[0]
        text = line.lower().split(" ")[1:]

        class1Name = "True"
        class2Name = "Pos"

        #compute feature counts for test lines
        x=dict()

        for word in text:
            word=''.join([i for i in word if i.isalpha()])
            print word
            if word=="":
                continue
            if word in weights:
                if word in x:
                    x[word] +=1
                else:
                    x[word] =1

        #compute activation for the line and decide the label of the output class
        a=0.0
        for word in x:
            a += weights[word][0]*x[word]
        a +=bias1

        if a<=0:
            class1Name="Fake"
        a = 0.0
        for word in x:
            a += weights[word][1]*x[word]
        a +=bias2
        if a<=0:
            class2Name="Neg"


        outputFile.write(key + " " + class1Name + " " + class2Name + "\n")

    outputFile.close()

def main():

    #fileName=sys.argv[1]
    weights,bias1,bias2=getInput("averagedmodel.txt")

    classify(sys.argv[2], weights, bias1, bias2)

main()
