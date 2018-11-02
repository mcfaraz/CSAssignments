"""
This module represents some classes for a simple word game.

There are a number of incomplete methods in the which you must implement to make fully functional.

About the game board!
The board's tiles are indexed from 1 to N, and the first square (1,1) is in the top left.
A tile may be replaced by another tile, hence only one tile may occupy a space at any one time.
"""


class LetterTile:
    """ This class is complete. You do not have to do anything to complete this class """

    def __init__(self, letter):
        self.letter = letter.lower()

    def get_letter(self):
        """ Returns the letter associated with this tile. """
        return self.letter

    def get_score(self):
        """ Returns the score associated with the letter tile """
        return {
            'a': 1,
            'b': 2,
            'c': 2,
            'd': 3,
            'e': 1,
            'f': 3,
            'g': 2,
            'h': 3,
            'i': 1,
            'j': 3,
            'k': 2,
            'l': 3,
            'm': 5,
            'n': 3,
            'o': 1,
            'p': 2,
            'q': 2,
            'r': 3,
            's': 1,
            't': 1,
            'u': 1,
            'v': 3,
            'w': 3,
            'x': 5,
            'y': 3,
            'z': 5
        }[self.letter]


class GameBoard:
    """ This class represents the gameboard itself.
        You are requried to complete this class.
    """

    def __init__(self, width, height):
        """ The constructor for setting up the gameboard """
        # Store the board size
        self.width = width
        self.height = height
        self.tiles = {}  # Initiate a dictionary of letters

    def set_tile(self, x, y, tile):
        """ Places a tile at a location on the board. """
        self.tiles[(x, y)] = tile

    def get_tile(self, x, y):
        """ Returns the tile at a location on the board """
        return self.tiles[(x, y)]

    def remove_tile(self, x, y):
        """ Removes the tile from the board and returns the tile"""
        return self.tiles.pop((x, y))

    def get_words(self):
        """ Retuns a list of the words on the board sorted in alphabetic order.

        """
        words = []
        for i in range(1, self.height+1):
            tmp = ''
            for j in range(1, self.width+1):
                if (i, j) not in self.tiles:
                    if len(tmp) > 1:  # Detect the end of a word
                        words.append(tmp)
                    tmp = ''  # Flush tmp
                if (i, j) in self.tiles:
                    tmp = tmp + self.get_tile(i, j).get_letter()
            if len(tmp) > 1:  # Detect a word when reaching end of a row
                words.append(tmp)

        for i in range(1, self.width+1):
            tmp = ''
            for j in range(1, self.height+1):
                if (j, i) not in self.tiles:
                    if len(tmp) > 1:  # Detect the end of a word
                        words.append(tmp)
                    tmp = ''
                if (j, i) in self.tiles:
                    tmp = tmp + self.get_tile(j, i).get_letter()
            if len(tmp) > 1:
                words.append(tmp)  # Detect a word when reaching end of a column

        words.sort()
        return words

    def top_scoring_words(self):
        """ Returns a list of the top scoring words.
            If there is a single word, then the function should return a single item list.
            If multilpe words share the highest score, then the list should contain the words sorted alphabetically.
        """
        words = self.get_words()
        word_score = lambda word: sum([LetterTile(c).get_score() for c in word])  # Calculate the score of a word
        scores = {word: word_score(word) for word in words}  # Find the score for all the words
        highest_score = max(scores.values())
        high_scores = [key for key in scores if scores[key] == highest_score]  # Find the words with the highest score
        if len(high_scores) > 1:
            high_scores.sort()
        return high_scores

    def print_board(self):
        """ Prints a visual representation of the board
            Please use the - character for unused spaces
        """
        for i in range(1, self.height+1):
            for j in range(1, self.width+1):
                if (i, j) in self.tiles:
                    print(self.get_tile(i, j).get_letter(), end='')
                else:
                    print('-', end='')
            print('\n')

    def letters_placed(self):
        """ Returns a count of all letters currently on the board """
        return len(self.tiles)


if __name__ == "__main__":
    """ This is just a sample for testing you might want to add your own tests here """
    board = GameBoard(10, 10)

    a = LetterTile("a")
    b = LetterTile("b")
    d = LetterTile("d")
    e = LetterTile("e")
    m = LetterTile("m")
    o = LetterTile("o")
    s = LetterTile("s")
    t = LetterTile("t")
    n = LetterTile("n")
    l = LetterTile("l")
    z = LetterTile("z")
    v = LetterTile("v")
    p = LetterTile("p")

    board.set_tile(1, 1, a)
    board.set_tile(1, 2, d)
    board.set_tile(1, 3, a)
    board.set_tile(1, 4, m)
    board.set_tile(1, 6, t)

    board.set_tile(2, 4, o)
    board.set_tile(2, 6, a)

    board.set_tile(3, 1, s)
    board.set_tile(3, 2, a)
    board.set_tile(3, 3, n)
    board.set_tile(3, 4, d)
    board.set_tile(3, 6, b)

    board.set_tile(4, 1, z)
    board.set_tile(4, 4, l)
    board.set_tile(4, 6, l)

    board.set_tile(5, 1, z)
    board.set_tile(5, 4, e)
    board.set_tile(5, 5, v)
    board.set_tile(5, 6, e)

    board.set_tile(6, 1, p)

    board.print_board()
    print("There are {} letters placed on the board.".format(board.letters_placed()))
    print(board.remove_tile(1, 1).get_letter())
    board.print_board()
    print(board.get_words())
    print(board.top_scoring_words())

    print("There are {} letters placed on the board.".format(board.letters_placed()))

    # Uncomment below once you have implemented get_words
    print ("=== Words ===")
    for word in board.get_words():
        print(word)

    # Uncomment below once you have implmented top_scoring_words
    print ("=== Top Scoring Words ===")
    for word in board.top_scoring_words():
        print(word)
