# Problem 1: Given an integer array nums, return true if any value appears more 
# than once in the array, otherwise return false.
from collections import defaultdict
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

# Problem 3: Given an array of integers nums and an integer target, 
# return the indices i and j such that nums[i] + nums[j] == target and i != j.

# You may assume that every input has exactly one pair of indices 
# i and j that satisfy the condition.
#Return the answer with the smaller index first.

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
       prevMap = {} # val : index

       for i , n in enumerate(nums):
            diff = target - n
            if diff in prevMap:
                return [prevMap[diff], i]
            prevMap[n] = i
            
# Notes: create a hashmap to track the values and the index
# then use enumerate which lets you for loop through the nums list with index and value
# then find the diff, find in prevMap return the two indexs
# else add the val and index to hashmap

#Problem 4: Given an array of strings strs, group all anagrams together into sublists.
# You may return the output in any order. 

# An anagram is a string that contains the exact same characters as another string,
# but the order of the characters can be different.

# Input: strs = ["act","pots","tops","cat","stop","hat"]
# Output: [["hat"],["act", "cat"],["stop", "pots", "tops"]]

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = defaultdict(list)

        for word in strs:
            key = ''.join(sorted(word))
            anagrams[key].append(word)
        return list(anagrams.values())
    
#Notes: create a defaultdict––a hashmap where if a key doesn't exist, a new default value automactially 
# loop through the word in strs list
# break down each word in a sorted char array for the key
# add the word to that key
# then take the values for the hashmap and return it as a list

# Problem 5: Given an integer array nums and an integer k, return the k most frequent elements within the array.

#Input: nums = [1,2,2,3,3,3], k = 2 Output: [2,3]
# Input: nums = [7,7], k = 1 Output: [7]

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {} 
        freq = [[] for i in range(len(nums) + 1)]
        
        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for n , c in count.items():
            freq[c].append(n)
        
        res = []
        for i in range(len(freq) - 1, 0 , -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res

# Notes: create a dictionary to story numbered apperance for each number
# loop through the array and add to the count with key: num and value being count or 0 + 1

# Create a frequency bucket by intialzing a array for length of nums + 1 

# fill in the freq bucket by looping through the count.items (tuples of the dict)
# take n = value c = count and append to the freq bucket as count as index and n and value 

# Collect the top k frequent numbers
# intialize the result list
#  range(start, stop, step) -> for the length of freq, iterate backwards
# for the values of n, add to result array, then return res


# Problem 6: Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

# Input: ["neet","code","love","you"] Output:["neet","code","love","you"]

class Solution:

    def encode(self, strs: List[str]) -> str:
        res = ""
        for word in strs:
            res += str(len(word) + '#' + word)
        return word

    def decode(self, s: str) -> List[str]:
       res = []
       i = 0
       while i < len(s):
           j = i
           while s[j] != '#':
               j += 1
           length = int(s[i:j])
           word =  s[j+1: j+1+length]
           res.append(word)
           i = j + 1 + length
       return res
   
   # Notes: for encode, intailize a string and then add all strs to it with cast on length
   # make sure to add the length of the word, # to place hold, and the word
   
   # for decode, intalize result as a list
   # iterate i while its less than length of str
   # make j = i to track when # is
   # loop until j = #
   # take the substring of i to j and cast to int for length
   # then the word is substring from j+ 1 to j+1 + length
   # add that word to result and move i to the end of word
   # at then end return result
   
   
   # Problem 7: Given an integer array nums, return an array output where output[i] is the product of all the elements of nums except nums[i].
   # Input: nums = [1,2,4,6] Output: [48,24,12,8]
   
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))

        # Build prefix product
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]

        # Build suffix product and multiply directly
        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= suffix
            suffix *= nums[i]
        return res
   
   # Notes: create a result array with length of nums filled with 1
   # use prefix and suffix techique:
   # prefix[i]: product of all elements before index i.
   # suffix[i]: product of all elements after index i.
   # iterate through the foward and backwards of nums and muplity each value 
   # by the suffix or prefix and return result
   
# You are given a a 9 x 9 Sudoku board board. A Sudoku board is valid if the following rules are followed:
# Each row must contain the digits 1-9 without duplicates.
# Each column must contain the digits 1-9 without duplicates.
# Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without duplicates.
# Return true if the Sudoku board is valid, otherwise return false
   
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = defaultdict(set)
        rows = defaultdict(set)
        squares = defaultdict(set) # key = (r/3, c/3)
        
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (board[r][c] in rows[r] or
                   board[r][c] in cols[c] or
                   board[r][c] in squares[(r//3, c//3)]):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r//3, c//3)].add(board[r][c])
            return True 
        
# Notes: create a defaultdict of sets for cols rows and squares
# iterate through rows and cols and then check each value 
# if board is . continue
# if board is in rows for r , in cols for c, or squares for r//3, c//3: return false
# else add that value to cols, rows, square 
# return true at the end


# Week 2:
# Problem 1: Given a string s, return true if it is a palindrome, otherwise return false.

# Input: s = "Was it a car or a cat I saw?"
# Output: true


class Solution:
    def isPalindrome(self, s: str) -> bool:
        start = 0
        end = len(s) - 1
        while start < end:
            
            while start < end and not s[start].isalnum():
                 start += 1
            while start < end and not s[end].isalnum():
                end -= 1

            if s[start].lower() != s[end].lower():
                return False
            
            start += 1
            end -= 1
        return True
    
    # Notes: Use the two pointer method: 
    # intialize a pointer the start and end of string and converge them
    # while start is less than end pointer
    # move the pointer if they aren't a letter
    # then compare values and if they are equal, if not return false
    # else move the pointers
    # return true at end loop 
    
    # Problem 2: Given an array of integers numbers that is sorted in non-decreasing order.
    # Return the indices (1-indexed) of two numbers, [index1, index2], such that they add up to a given target number target and index1 < index2. 
    # Note that index1 and index2 cannot be equal, therefore you may not use the same element twice.
    
    # Input: numbers = [1,2,3,4], target = 3
    # Output: [1,2]
    
    class Solution:
        def twoSum(self, nums: List[int], target: int) -> List[int]:
            l = 0
            r = len(nums) - 1
            while l < r:
                total = nums[l] + nums[r]
                if total > target:
                    r -= 1
                elif total < target:
                    l += 1
                else:
                    return [l + 1, r + 1]
                return []
    # Notes: Use the two pointer method: 
    # intialize a pointer the start and end of array and converge them
    # if the total of the integers is more than target, move the right pointer
    # if the total of the integers is less than target, move the left pointer
    # then return the correct index if equal

    # Problem 3: Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
    # where nums[i] + nums[j] + nums[k] == 0, and the indices i, j and k are all distinct.
    # The output should not contain any duplicate triplets. 
    # You may return the output and the triplets in any order.
    
    # Input: nums = [-1,0,1,2,-1,-4]
    # Output: [[-1,-1,2],[-1,0,1]]
    
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            res = []
            nums.sort()

            for i, a in enumerate(nums):
                if i > 0 and a == nums[i - 1]:
                    continue
                l, r = i + 1, len(nums) - 1
                while l < r:
                    threeSum = a + nums[l] + nums[r]
                    if threeSum > 0:
                        r -= 1
                    elif threeSum < 0:
                        l += 1
                    else:
                        res.append([a, nums[l], nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l - 1]: 
                            l += 1
            return res
        
    # Notes: Use the two pointer method: 
    # intialize a result list and sort the array to use dynamic pointers
    # use enumeration since you need the value and index
    # then l pointer to left of index and right is at end of array
    # for the loop of remaining array, find the threeSum total
    # if the total is more than target, move the right pointer
    # if the total is less than target, move the left pointer
    # else append the total parts
    # to move the pointers, its while l < r and num[l] has a duplicate
    # move l again / duplicate only matter from one direction
    # return result 

