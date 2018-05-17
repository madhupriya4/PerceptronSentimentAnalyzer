import sys, json


# get the input set of lines in lower case
def getInput():
    inputFile = open(sys.argv[1], "r").read().lower()

    #remove punctuations
    puncList = [".", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "$", "&", ")", "(", "\""]
    for punc in puncList:
        inputFile = inputFile.replace(punc, '')

    fileLines = inputFile.split("\n")


    # remove last new line
    if (fileLines[len(fileLines) - 1]) == "":
        fileLines = fileLines[:-1]

    return fileLines

#compute the activation and adjust parameters if the class is wrong
def adjustParams(x, y, weight, cweight, bias, cbias, count, classNum):
    a = 0.0
    for word in x:
        a += weight[word][classNum] * x[word]
    a += bias

    if a * y <= 0: #if class is wrong
        for word in x:
            weight[word][classNum] += x[word] * y
            cweight[word][classNum] += x[word] * y * count
        bias += y
        cbias += y * count

    return weight, cweight, bias, cbias

#train the perceptron model where the features are the words in the vocabulary

def Train(fileLines, stopwords):
    #initialize the weight and bias params
    weight = dict()
    cweights = dict()
    b1 = 0.0
    b2 = 0.0
    cb1 = 0.0
    cb2 = 0.0
    c = 1.0

    maxIter = 25

    #iterate to tune the parameters
    for i in xrange(maxIter):

        #update the weights and bias for each sentence
        for line in fileLines:
            c1 = line.split(" ")[1]
            c2 = line.split(" ")[2]
            words = line.split(" ")[3:]
            y1 = 1
            y2 = 1
            if c1 == "fake":
                y1 = -1
            if c2 == "neg":
                y2 = -1

            #compute the feature counts for each sentence
            x = dict()
            for word in words:
                word = ''.join([i for i in word if i.isalpha()])
                if word in stopwords:
                    continue
                if word == "":
                    continue
                if word not in weight:
                    weight[word] = [0.0, 0.0]
                    cweights[word] = [0.0, 0.0]
                if word not in x:
                    x[word] = 1
                else:
                    x[word] += 1

            #adjust the params for each of the two classes
            weight, cweights, b1, cb1 = adjustParams(x, y1, weight, cweights, b1, cb1, c, 0)
            weight, cweights, b2, cb2 = adjustParams(x, y2, weight, cweights, b2, cb2, c, 1)
            c += 1

    writeModel("vanillamodel.txt", weight, b1, b2)

    #subtract the cached weights for the average model
    for word in weight:
        weight[word][0] -= (1 / c) * cweights[word][0]
        weight[word][1] -= (1 / c) * cweights[word][1]

    b1 -= cb1 / c
    b2 -= cb2 / c

    writeModel("averagedmodel.txt", weight, b1, b2)

#write the model params to the output file
def writeModel(fileName, weight, bias1, bias2):
    with open(fileName, 'w') as outputFile:
        outputFile.write(json.dumps(weight) + "\n")
        outputFile.write(str(bias1) + "\n")
        outputFile.write(str(bias2))


def main():

    #remove standard English stopwords
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                 'now']

    fileLines = getInput()
    Train(fileLines, stopwords)


main()
