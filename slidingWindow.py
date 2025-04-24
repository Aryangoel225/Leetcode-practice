# Problem 1: You are given an integer array prices where prices[i] is the price of NeetCoin on the ith day.
# You may choose a single day to buy one NeetCoin and choose a different day in the future to sell it.
# Return the maximum profit you can achieve. You may choose to not make any transactions, in which case the profit would be 0.

# Input: prices = [10,1,5,6,7,1]
# Output: 6

from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l = 0
        r = 1
        max_profit = 0

        while r < len(prices):
            if prices[l] < prices[r]:
                profit = prices[r] - prices[l]
                max_profit = max(profit, max_profit)
            else:
                l = r
            r += 1
        return max_profit

# Notes: sliding window problem with left pointer at the start
# the right pointer at 1 , maxprofit to track the max
# loop through the lenght of price array
# if the price of l is less than r (i.e profit) find the progit and replace max
# else move l to r 
# increment r
# return max profit


# Problem 2: Given a string s, find the length of the longest substring without duplicate characters.
# A substring is a contiguous sequence of characters within a string.

# Input: s = "zxyzxyz"
# Output: 3

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l = 0
        foundChar = set()
        max_length = 0

        for r in range(len(s)):
            while s[r] in foundChar:
                foundChar.remove(s[l])
                l += 1
            foundChar.add(s[r])
            max_length = max(max_length, r - l + 1)

        return max_length
    
# Notes: sliding window l and r are pointers. l start at 0
# foundChar is the count unique character put into a set()
# max length is the max substring length
# loop throug the string with r
# the not condition: if char in found in foundChar
# remove that character and move l by 1
# then at the current value of char to foundChar
# find max length
# return max length

# Problem 3: given a string of captail letters, return the longest substring that replace k letters 
# will give a substring of one type of letter

# Input: s = "XYYX", k = 2
# Output: 4

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        res = 0
        count = {}

        l = 0
        maxf = 0
         
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])

            while (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)

        return res
    
# Notes: need res to track the longest stirng
# count is to track the frequency of letters
# l is for a pointer a the start 
# maxf is to know the max frequency to be able to tell if it violates the k condition
# loop through the length of the string
# find the count for the value r pointer
# find max frequency
# while not the condition: the number of letters remaining from length - maxf > k
# subtract that count value of l
# move l by one 
# find max result
# return max result

