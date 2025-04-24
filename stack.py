# Week 3: Problem 1: You are given a string s consisting of the following characters: '(', ')', '{', '}', '[' and ']'.
# The input string s is valid if and only if:
# Every open bracket is closed by the same type of close bracket.
# Open brackets are closed in the correct order.
# Every close bracket has a corresponding open bracket of the same type.
# Return true if s is a valid string, and false otherwise.
    
# Input: s = "([{}])"
# Output: true

from typing import List

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
        # only add open parathesis if open < n
        # only add closing parathesis if closed < open
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

# Notes: create a array of pairs of the positions and speed arrays given
# create a stack
# create a reverse order sorted list for loop to iterate through with p for position and s for speed
# append in stack the time it takes for the car
# and if the lenght of the stack is greater than 2 and the top element smaller than element before it (meaning the earlier car will take less time and catch up)
# pop the value from the stack (the cars became a fleet)
# reutrn the length of stack 


# Problem 7: given an array of heights to represent bars. Find the max area rectangle
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        maxArea = 0
        stack = []  # pair (index, height)

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start, h))

        for i, h in stack:
            maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea

# Notes: 
# 1. Initialize `maxArea` to 0 and a stack to store pairs of (index, height).
# 2. Loop through the `heights` array with `i` (index) and `h` (height).
# 3. Save the current index in `start`.
# 4. While the stack is not empty and the top of the stack has a greater height than the current height:
#    - Pop the previous bar for its index (which it can extend back furthest to) and height.
#    - Calculate the area and update `maxArea` if the new area is larger.
#    - Set `start` equal to the index of the popped value because that is how far the current bar extends.
# 5. Append the current `start` and `h` to the stack.
# 6. After the loop, process the remaining values in the stack to calculate the area and update `maxArea`.
# 7. Return `maxArea`.