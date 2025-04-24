# Week 2:
# Problem 1: Given a string s, return true if it is a palindrome, otherwise return false.

# Input: s = "Was it a car or a cat I saw?"
# Output: true


from typing import List


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