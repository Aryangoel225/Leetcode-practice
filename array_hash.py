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
    
    # Problem 4: You are given an integer array heights where heights[i] represents the height of the [i] bar 
    # You may choose any two bars to form a container. 
    # Return the maximum amount of water a container can store.
   
    class Solution:
        def maxArea(self, heights: List[int]) -> int:
            l = 0
            r = len(heights) - 1
            maxArea = 0

            while l < r:
                area = min(heights[l] , heights[r]) * (r - l)
                maxArea = max(maxArea, area)
                if heights[l] <= heights[r]:
                    l += 1
                else:
                    r -= 1 
            return maxArea
    # Notes: intialize left and right pointers, and max area
    # while looping, find the area by taking the min height of the two pointers and multplying by width
    # compare to maxArea 
    # then whatever the lowest value of the pointers is shift that value, 
    # because you trying to find the largest bucket
    # return maxArea
    
    #Problem 5: You are given an array non-negative integers height which represent an elevation map.
    # Each value height[i] represents the height of a bar, which has a width of 1.
    # Return the maximum area of water that can be trapped between the bars.


    class Solution:
        def trap(self, height: List[int]) -> int:
            if not height: 
                return 0

            l , r = 0, len(height) - 1
            leftMax , rightMax = height[l], height[r]
            res = 0

            while l < r:
                if leftMax < rightMax:
                    l += 1
                    leftMax = max(leftMax, height[l])
                    res += leftMax - height[l]
                else:
                    r -= 1
                    rightMax = max(rightMax, height[r])
                    res += rightMax - height[r]
            return res
        
    #Notes: if height is null return 0
    # intialize two pointers left and right of array
    # initalize left and right max value of those pointers
    # result = 0
    # loop through and check if left or rightMax is less, which is need to calculate the side of the bucket
    # then shift the respective pointer, and find the new max, add the difference to the result
    # return result
    
    # Week 3: Problem 1: You are given a string s consisting of the following characters: '(', ')', '{', '}', '[' and ']'.
    # The input string s is valid if and only if:
        # Every open bracket is closed by the same type of close bracket.
        # Open brackets are closed in the correct order.
        # Every close bracket has a corresponding open bracket of the same type.
        # Return true if s is a valid string, and false otherwise.
    
    # Input: s = "([{}])"
    # Output: true
    
    class Solution:
        def isValid(self, s: str) -> bool:
            stack = []
            closeToOpen = {")" : "(", "}" : "{", "]" : "[" }

            for c in s:
                if c in closeToOpen:
                    if stack and stack[-1] == closeToOpen[c]:
                        stack.pop()
                    else:
                        return False
                else:
                    stack.append(c)

            return True if not stack else False
        
    # Notes: intialize a stack and a hashset
    # loop for each char in string, if char is a closing bracket key ->
    # if stack not empty and pop item from stack equals the value of the char for the key pop the stack
    # else return false
    # if its an opening bracket appended it to the stack 
    # Finally return true if stack is empty else return false
    
    # Problem 2: Design a stack class that supports the push, pop, top, and getMin operations.
    # int getMin() retrieves the minimum element in the stack.
    
    class MinStack:
        def __init__(self):
            self.stack = []
            self.minStack = []

        def push(self, val: int) -> None:
            self.stack.append(val)
            val = min(val, self.minStack[-1] if self.minStack else val)
            self.minStack.append(val)

        def pop(self) -> None:
            self.stack.pop()
            self.minStack.pop()

        def top(self) -> int:
            return self.stack[-1]
            
        def getMin(self) -> int:
            return self.minStack[-1]
        
    # Notes: create a stack for both tracking values and the min for each value parallel 
    # push: append to both stack the correct value and the min value which should be compared to the top
    # pop: use pop on both
    # top/ getMin: use get from the top of the stack
    
    # Problem 3: You are given an array of strings tokens that represents a valid arithmetic expression in Reverse Polish Notation.
    
    class Solution:
        def evalRPN(self, tokens: List[str]) -> int:
            stack = []
            for n in tokens:
                if n == "+":
                    stack.append(stack.pop() + stack.pop())
                elif n == "-":
                    a, b = stack.pop(), stack.pop()
                    stack.append(b - a)
                elif n == "*":
                    stack.append(stack.pop() * stack.pop())
                elif n == "/":
                    a, b = stack.pop(), stack.pop()
                    stack.append(int(float(b) / a))
                else:
                    stack.append(int(n))
            return stack[0]
    
    # Problem 4: You are given an integer n. 
    # Return all well-formed parentheses strings that you can generate with n pairs of parentheses.
    
    # Input: n = 3
    # Output: ["((()))","(()())","(())()","()(())","()()()"]
    
    class Solution:
        def generateParenthesis(self, n: int) -> List[str]:
            # only add open parathensis if open < n
            # only add closing parathensis if closed < open
            # valid IIF open == closed == n

            stack = []
            res = []

            def backtrack(openN, closedN):
                if openN == closedN == n:
                    res.append("".join(stack))
                    return
                
                if openN < n:
                    stack.append("(")
                    backtrack(openN + 1, closedN)
                    stack.pop()

                if closedN < openN:
                    stack.append(")")
                    backtrack(openN, closedN + 1)
                    stack.pop()
            backtrack(0,0)
            return res
    # Notes: This problem requires backtracking with a binary tree structure
    # in other words: there is a recursive loop with 2 different branches
    # initialize stack and result 
    # create a inside function with the num of Open and Close brackets
    # if open = closed = n : base case return 
    # if open is less than n, add open, recursive call with open + 1, 
    # and pop the stack to clear it for new combitnation 
    # if closed is less than open, add closed, recursive call with closed + 1, 
    # and pop the stack to clear it for new combitnation 
    # call backtrack in general func and return res
    
    
    # Problem 5: You are given an array of integers temperatures where temperatures[i] 
    # represents the daily temperatures on the ith day.
    
    # return a array where each index is the days after the corresponding 
    # index temperature has a tempature higher than it
    
    # Input: temperatures = [30,38,30,36,35,40,28]
    # Output: [1,4,1,2,1,0,0]
    
    class Solution:
        def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
            result = [0] * len(temperatures)
            stack = [] # pair: [temp, index]

            for i, t in enumerate(temperatures):
                while stack and t > stack[-1][0]:
                    stackT, stackInd = stack.pop()
                    result[stackInd] = i - stackInd
                
                stack.append([t, i])

            return result
    # Notes: create a result array with 0 and length of temps
    # then a empty stack with pair: [temp, index]
    # for a  enumerate for loop
    # create a while loop that as the stack not empty and the current temp
    # is greater than the top of the stack
    # pop from the stack and add the difference of the current index from the stack index 
    # to the corresponding result index array position 
    # if not true then add the current value and index pair to stack
    # return result
    
    
    
    # Car fleet problem 6: you are give an array of position and speed of n cars on a road
    # You are given the target and need to find how many fleets of cars (if a car catches up to another it stays at its speed and joins its fleet)
    # are left at the end til the target
    
    # Input: target = 10, position = [1,4], speed = [3,2]
    # Output: 1
    
    class Solution:
        def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
            pair = [[p, s] for p , s in zip(position, speed)]
            stack = []
            
            for p , s in sorted(pair)[::-1]: # Reverse Sorted Order
                stack.append((target - p) / s)
                if len(stack) >= 2 and stack[-1] <= stack[-2]:
                    stack.pop()

            return len(stack)
    
    # 
    