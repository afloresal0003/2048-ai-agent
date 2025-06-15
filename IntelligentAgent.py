#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 7 16:02:19 2024

UNI: AF3262
NAME: Anthony Flores-Alvarez
"""
import time
from BaseAI import BaseAI

"""
This AI agent plays 2048 framing it as an adversarial search with expectiminimax,
alpha-beta pruning, and custom heuristics. This strategy is pulled in part from the 
following paper examining 2048 strategies:
    - https://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf
    
Some run history for your convenience (group=10runs):
Top 5 Max Tiles for Group 1: [2048, 2048, 2048, 2048, 2048]
Top 5 Max Tiles for Group 2: [2048, 2048, 2048, 1024, 1024]
Top 5 Max Tiles for Group 3: [4096, 2048, 2048, 2048, 2048]
Top 5 Max Tiles for Group 4: [2048, 2048, 1024, 1024, 1024]
Top 5 Max Tiles for Group 5: [4096, 2048, 2048, 2048, 1024]
Top 5 Max Tiles for Group 6: [2048, 2048, 2048, 2048, 1024]
After reducing time limit (0.2->0.17)
Top 5 Max Tiles for Group 1: [2048, 2048, 2048, 1024, 1024]
Top 5 Max Tiles for Group 2: [2048, 2048, 2048, 2048, 2048]
Top 5 Max Tiles for Group 3: [2048, 2048, 2048, 1024, 1024]
"""
class IntelligentAgent(BaseAI):
    
    # From assignment description
    playerMoveTimeLimit = 0.2
    possibleTileProbabilities = [(2, 0.9), (4, 0.1)]

    # From the paper referenced above (used to put importance on certain tiles)
    snakeShapedWeights = [
        [4**15, 4**14, 4**13, 4**12],
        [4**8,  4**9,  4**10, 4**11],
        [4**7,  4**6,  4**5,  4**4],
        [4**0,  4**1,  4**2,  4**3],
    ]
    
    # Weights for each heuristic (used Bayesian optimization algo to find these)
    snakeWeight = 0.15966252220214197
    smoothnessWeight = 2.3089382562214906
    emptyCellsWeight = 2.410254660260118
    maxInCornerWeight = 6.832635188254583
    
    def getMove(self, grid):
        moveStartTime = time.time()
        depth = 1
        bestMove = None

        # Iterative deepening search for best move within time limit
        while time.time() - moveStartTime < self.playerMoveTimeLimit:
            try:
                _, move = self.maximize(grid, float('-inf'), float('inf'), depth, moveStartTime)
                if move is not None:
                    bestMove = move
                depth += 1
            except TimeoutError:
                break

        return bestMove

    def maximize(self, grid, alpha, beta, depth, moveStartTime):
        if depth == 0 or not grid.canMove():
            return self.evaluate(grid), None

        if time.time() - moveStartTime >= self.playerMoveTimeLimit:
            raise TimeoutError()

        maxChildMove, maxStateUtility = None, float('-inf')

        for move, child in grid.getAvailableMoves():
            minChanceUtility, _ = self.chance(child, alpha, beta, depth - 1, moveStartTime)
            if minChanceUtility > maxStateUtility:
                maxChildMove, maxStateUtility = move, minChanceUtility

            alpha = max(alpha, minChanceUtility)
            # Beta cutoff
            if beta <= alpha:
                break  

        return maxStateUtility, maxChildMove

    def minimize(self, grid, alpha, beta, depth, moveStartTime):
        if depth == 0 or not grid.canMove():
            return self.evaluate(grid), None

        if time.time() - moveStartTime >= self.playerMoveTimeLimit:
            raise TimeoutError()

        minStateUtility = float('inf')

        # Sorts available cells by their importance (based on heuristic weights) in descending order and select the top 4 most important cells
        emptyCells = sorted(grid.getAvailableCells(), key=lambda cell: self.getCellSnakeImportance(grid, cell), reverse=True)[:4]
        
        for cell in emptyCells:
            for tileValue, probability in self.possibleTileProbabilities:
                child = grid.clone()
                child.setCellValue(cell, tileValue)

                childMaxUtility, _ = self.maximize(child, alpha, beta, depth - 1, moveStartTime)
                minStateUtility = min(minStateUtility, childMaxUtility)

                beta = min(beta, childMaxUtility)
                if beta <= alpha:
                    break
            if beta <= alpha:
                break

        return minStateUtility, None

    # We treat chance nodes as min nodes 
    def chance(self, grid, alpha, beta, depth, moveStartTime):
        return self.minimize(grid, alpha, beta, depth, moveStartTime)

    # Computes utility of a grid state
    def evaluate(self, grid):
        # Combines heuristics using the weights
        score = (
            self.snakeWeight * self.calcSnakeAdherence(grid) +
            self.smoothnessWeight * self.calcSmoothness(grid) +
            self.emptyCellsWeight * self.getEmptyCells(grid) +
            self.maxInCornerWeight * self.checkMaxTileInCorner(grid)
        )
        return score

    """
    Calculates the adherence of the grid's tiles to the predefined snake-shaped pattern from the paper
    This encourages the tiles to align in a specific order that optimizes merging opportunities
    """
    def calcSnakeAdherence(self, grid):
        adherenceScore = 0
        for x in range(grid.size):
            for y in range(grid.size):
                # Adds the product of the tile value and its corresponding weight in the snake pattern
                adherenceScore += grid.map[x][y] * self.snakeShapedWeights[x][y]
        return adherenceScore

    """
    Measures the smoothness of the grid (aka difference between adjacent tiles)
    Lower smoothness scores represent grids with large differences between neighboring tiles,
    which are less likely to be able to merge
    """
    def calcSmoothness(self, grid):
        smoothnessScore = 0
        for x in range(grid.size):
            for y in range(grid.size):
                current = grid.map[x][y]
                # Checks horizontal neighbor for smoothness contribution
                if x + 1 < grid.size:
                    smoothnessScore -= abs(current - grid.map[x + 1][y])
                # Checks vertical neighbor for smoothness contribution    
                if y + 1 < grid.size:
                    smoothnessScore -= abs(current - grid.map[x][y + 1])
        
        return smoothnessScore

    """
    Counts the number of empty cells on the grid
    Empty cells allow for more moves and reduce the risk of deadlocks
    """
    def getEmptyCells(self, grid):
        return len(grid.getAvailableCells())

    """
    Encourages the highest-value tile to stay in the top-left corner
    Returns a positive score if the max tile is in the corner and a penalty if it is not
    """
    def checkMaxTileInCorner(self, grid):
        maxTile = max(cell for row in grid.map for cell in row)
        topLeftTile = grid.map[0][0]

        # Reward if the highest-value tile is in the top-left corner
        if topLeftTile == maxTile:
            return maxTile
        # Penalize based on how far the highest-value tile is from the top-left corner   
        else:
            return -abs(maxTile - topLeftTile)  

    """
    Determines the importance of a specific cell in the grid based on the snake-shaped pattern
    by returning the weight assigned to the cell in the predefined pattern
    """
    def getCellSnakeImportance(self, grid, cell):
        # Unpacks cell's coordinates
        x, y = cell
        return self.snakeShapedWeights[x][y]