import itertools
import collections


def morseDecode(inputStringList):
    """
    This method should take a list of strings as input. Each string is equivalent to one letter
    (i.e. one morse code string). The entire list of strings represents a word.

    This method should convert the strings from morse code into english, and return the word as a string.

    """
    # Please complete this method to perform the above described function
    morseLetters = {'.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '--.': 'G', '..-.': 'F', '....': 'H'
        , '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P'
        , '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y'
        , '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5', '-....': '6'
        , '--...': '7', '---..': '8', '----.': '9', '-----': '0'}

    decoded_word = ''
    for letter in inputStringList:
        decoded_word += morseLetters[letter]

    return decoded_word



def partialDecode(inputStringList, words):
    if len(inputStringList) == 0:
        return words
    out = []
    for w in words:
        out.append(w + morseDecode(['-' + inputStringList[0][1:]]))
        out.append(w + morseDecode(['.' + inputStringList[0][1:]]))
    return partialDecode(inputStringList[1:], out)


def morsePartialDecode(inputStringList):
    """
    This method should take a list of strings as input. Each string is equivalent to one letter
    (i.e. one morse code string). The entire list of strings represents a word.

    However, the first character of every morse code string is unknown (represented by an 'x' (lowercase))
    For example, if the word was originally TEST, then the morse code list string would normally be:
    ['-','.','...','-']

    However, with the first characters missing, I would receive:
    ['x','x','x..','x']

    With the x unknown, this word could be TEST, but it could also be EESE or ETSE or ETST or EEDT or other permutations.

    We define a valid words as one that exists within the dictionary file provided on the website, dictionary.txt
    When using this file, please always use the location './dictionary.txt' and place it in the same directory as
    the python script.

    This function should find and return a list of strings of all possible VALID words.
    """
    # Please complete this method to perform the above described function

    dictionaryFileLoc = './dictionary.txt'
    dictWords = {}
    with open(dictionaryFileLoc) as f:
        for line in f:
            dictWords[line.strip().upper()] = 1

    words = []
    if len(inputStringList) < 1:
        return words

    words = ['']

    '''
    # Iterative approach
    words.append('')
    for i in inputStringList:
        tmp = []
        for w in words:
            i = '.' + i[1:]
            tmp.append(w + morseDecode([i]))
            i = '-' + i[1:]
            tmp.append(w + morseDecode([i]))
        words = tmp'''

    # Recursive approach
    words = partialDecode(inputStringList, words)

    out = []
    for w in words:
        if w in dictWords:  # Selecting valid words
            out.append(w)
    return out


class Maze:
    def __init__(self):
        """
        Constructor - You may modify this, but please do not add any extra parameters
        """
        self.maxHeight = 0
        self.maxWidth = 0
        self.grid = {}

    def addCoordinate(self, x, y, blockType):
        """
        Add information about a coordinate on the maze grid
        x is the x coordinate
        y is the y coordinate
        blockType should be 0 (for an open space) of 1 (for a wall)
        """
        # Please complete this method to perform the above described function
        if x > self.maxWidth:
            self.maxWidth = x + 1

        if y > self.maxHeight:
            self.maxHeight = y + 1

        self.grid[(x, y)] = blockType

    def printMaze(self):
        """
        Print out an ascii representation of the maze.
        A * indicates a wall and a empty space indicates an open space in the maze
        """
        printGrid = ''
        # Please complete this method to perform the above described function
        for y in range(0, self.maxHeight):
            for x in range(0, self.maxWidth):
                if (x,y) not in self.grid or self.grid[(x,y)] == 1:
                    printGrid += '*'
                else:
                    printGrid += ' '
            printGrid += '\n'
        print(printGrid)

    def findRoute(self, x1, y1, x2 ,y2):
        """
        This method should find a route, traversing open spaces, from the coordinates (x1,y1) to (x2,y2)
        It should return the list of traversed coordinates followed along this route as a list of tuples (x,y),
        in the order in which the coordinates must be followed
        If no route is found, return an empty list
        """
        path = self.bfs((x1, y1), (x2, y2))
        if path is not None and len(path) > 0:
            return path
        else:
            return []

    def bfs(self, start, end):
        queue = collections.deque([[start]])  # A double-ended queue to store future nodes to visit
        seen = set([start])  # Set of visited notes
        while queue:  # While there are nodes to visit
            path = queue.popleft()  # Retrieve the first node to visit
            x, y = path[-1]  # Get the coordinates of the first node
            if y == end[1] and x == end[0]:  # Check whether the destination is reached
                return path
            for x, y in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):  # Loop through four adjacent neighbours (East, West, South, North)
                if (0 <= y <= self.maxHeight and 0 <= x <= self.maxWidth and (x,y) in self.grid and (x, y) not in seen
                        and self.grid[(x,y)] == 0):  # Check if the neighbour exists and is a hole
                        queue.append(path + [(x, y)])
                        seen.add((x, y))


def morseCodeTest():
    """
    This test program passes the morse code as a list of strings for the word
    HELLO to the decode method. It should receive a string "HELLO" in return.
    This is provided as a simple test example, but by no means covers all possibilities, and you should
    fulfill the methods as described in their comments.
    """

    hello = ['....', '.', '.-..', '.-..', '---']
    print(morseDecode(hello))


def partialMorseCodeTest():
    """
    This test program passes the partial morse code as a list of strings
    to the morsePartialDecode method. This is provided as a simple test example, but by
    no means covers all possibilities, and you should fulfill the methods as described in their comments.
    """

    # This is a partial representation of the word TEST, amongst other possible combinations
    test = ['x', 'x', 'x..', 'x']
    print(morsePartialDecode(test))

    # This is a partial representation of the word DANCE, amongst other possible combinations
    dance = ['x..','x-','x.','x.-.','x']
    print(morsePartialDecode(dance))


def mazeTest():
    """
    This sets the open space coordinates for the example
    maze in the assignment.
    The remainder of coordinates within the max bounds of these specified coordinates
    are assumed to be walls
    """
    myMaze = Maze()

    myMaze.addCoordinate(1,0,0)
    myMaze.addCoordinate(1,1,0)
    myMaze.addCoordinate(7,1,0)
    myMaze.addCoordinate(1,2,0)
    myMaze.addCoordinate(2,2,0)
    myMaze.addCoordinate(3,2,0)
    myMaze.addCoordinate(4,2,0)
    myMaze.addCoordinate(6,2,0)
    myMaze.addCoordinate(7,2,0)
    myMaze.addCoordinate(4,3,0)
    myMaze.addCoordinate(7,3,0)
    myMaze.addCoordinate(4,4,0)
    myMaze.addCoordinate(7,4,0)
    myMaze.addCoordinate(3,5,0)
    myMaze.addCoordinate(4,5,0)
    myMaze.addCoordinate(7,5,0)
    myMaze.addCoordinate(1,6,0)
    myMaze.addCoordinate(2,6,0)
    myMaze.addCoordinate(3,6,0)
    myMaze.addCoordinate(4,6,0)
    myMaze.addCoordinate(5,6,0)
    myMaze.addCoordinate(6,6,0)
    myMaze.addCoordinate(7,6,0)
    myMaze.addCoordinate(5,7,0)


    myMaze.printMaze()
    print(myMaze.findRoute(1,0,7,1))


def main():
    morseCodeTest()
    partialMorseCodeTest()
    mazeTest()


if(__name__ == "__main__"):
    main()
