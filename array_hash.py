# Problem 1: Given an integer array nums, return true if any value appears more 
# than once in the array, otherwise return false.
from typing import List

class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        values = set()
        for num in nums:
            if num in values:
                return True
            values.add(num)  
        return False 
    
# Notes: set() is a hashSet that takes unique values, no keys, no order
# for x in xx is a advanced for loop 
# the indent directly below the loop is outside of the loop
# .add for hashSet 
# Captialize booleans

# Problem 2: Given two strings s and t, 
# return true if the two strings are anagrams of each other,
# otherwise return false.

# An anagram is a string that contains 
# the exact same characters as another string, 
# but the order of the characters can be different.

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        return countS == countT
    
# Notes: first check the length of each string, return false if unequal
# create two hashmaps
# loop through both strings, and for add a count for each letter
# add the , 0 so its not none but 0 when the count doesn't exist
# check if the hashmaps are the same