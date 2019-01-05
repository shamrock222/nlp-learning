# -*- coding: utf-8 -*-
"""
@refer:https://github.com/mammothb/symspellpy
@usage:word_segmentation
"""
import os
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module

def main():
    initial_capacity = 83000
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 0
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary, prefix_length)
    # load dictionary
    dictionary_path = os.path.join(os.path.dirname(__file__), "./data/frequency_dictionary_en_82_765.txt")
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

    edit_distance_max = 0
    prefix_length = 7
    sym_spell = SymSpell(83000, edit_distance_max, prefix_length)
    sym_spell.load_dictionary(dictionary_path, 0, 1)

    typo = "thequickbrownfoxjumpsoverthelazydog"
    correction = "the quick brown fox jumps over the lazy dog"
    result = sym_spell.word_segmentation(typo)  # create object

    # a sentence without any spaces
    input_term = "thequickbrownfoxjumpsoverthelazydog"
    result = sym_spell.word_segmentation(input_term)
    # display suggestion term, term frequency, and edit distance
    print("{}, {}, {}".format(result.corrected_string, result.distance_sum, result.log_prob_sum))

if __name__ == "__main__":
    main()