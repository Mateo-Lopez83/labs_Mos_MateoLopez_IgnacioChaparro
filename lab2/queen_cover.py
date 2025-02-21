"""
This script solves the minimum queen cover problem using Pyomo in a class-based structure.
Description: 
Given a chessboard of size 8 x 8, the goal is to place the minimum number of queens such that every
square is attacked by at least one queen.

Created on Mon March 25 17:07:12 2024

@author:
    Juan Andrés Méndez Galvis
"""

import time
from pyomo.environ import (
    Binary,
    ConcreteModel,
    ConstraintList,
    Objective,
    RangeSet,
    Var,
    minimize
)
from pyomo.opt import SolverFactory
from queen_mapper import visualize_queens


class QueenCoverSolver:
    """
    Class to solve the minimum queen cover problem using Pyomo.

    Given a chessboard of a specified size (default 8 x 8), the goal is to place
    the minimum number of queens such that every square is attacked by at least one queen.
    """

    def __init__(self, board_size=8):
        """
        Initialize the solver with a given board size.

        Args:
            board_size (int): Size of the chessboard (default is 8).
        """
        self.board_size = board_size
        self.coverage_matrix = {}
        self.model = None

    def generate_coverage_matrix(self):
        """
        Calculates which squares each queen can attack on a chessboard.

        Returns:
            dict: A dictionary representing the coverage matrix. Keys are (square1, square2)
                  tuples, and values are 1 if a queen placed on square2 can attack square1, 0 otherwise.
        """
        coverage_matrix = {}
        casillas = []
        for col in range(self.board_size):
            for fila in range(self.board_size):
                casillas.append((col,fila))


        self.casillas = casillas
        

        for square2 in casillas:
            col2, fila2 = square2
            for square1 in casillas:
                col1, fila1 = square1
                fila_check = (fila1 == fila2)
                col_check = (col1 == col2)
                diag_check = (abs(fila1 - fila2) == abs(col1 - col2))
                coverage_matrix[(square1, square2)] = 1 if fila_check or col_check or diag_check else 0
        
        self.coverage_matrix = coverage_matrix

        return coverage_matrix
    

    def create_pyomo_model(self, coverage_matrix):
        """
        Creates the Pyomo model for the queen placement problem.

        Args:
            coverage_matrix (dict): The coverage matrix computed for the chessboard.

        Returns:
            ConcreteModel: A Pyomo ConcreteModel representing the problem.
        """
        model = ConcreteModel()
        model.columnas = RangeSet(self.board_size)
        model.filas = RangeSet(self.board_size)
        casilla_a_indice = {}
        index = 1
        #Transformar de llave (square1,square2) a un valor de 1 a 64
        for col in model.columnas:
            for row in model.filas:
                casilla_a_indice[(col-1, row-1)] = index
                index += 1

        model.squares = RangeSet(self.board_size**2)
        #Para cada casilla, se indica si hay una reina (1) o no (0)
        model.x = Var(model.squares, domain=Binary,initialize=0)

        model.obj = Objective(expr=sum(model.x[i] for i in model.squares), sense=minimize)
        model.constraints = ConstraintList()

        for col in model.columnas:
            for fila in model.filas:
                casilla = (col-1,fila-1)
                casillas_atacantes = []
                #Por cada casilla, se revisa si cada otra casilla es atacante
                #si si lo es, se añade a una lista
                for col_atacante in model.columnas:
                    for fil_atacante in model.filas:
                        casilla_atacante= (col_atacante-1,fil_atacante-1)
                        if (coverage_matrix[(casilla,casilla_atacante)]==1):
                            casillas_atacantes.append(casilla_a_indice[casilla_atacante])
                #Se revisa que para cada casilla, al menos una de las casillas atacantes
                #sea una casilla con reina
                model.constraints.add(sum(model.x[i] for i in casillas_atacantes)>=1)
            
        self.model = model
        return model

    def solve_and_visualize(self):
        """
        Solves the Pyomo model and visualizes the queen placements.

        This method should:
         - Solve the model using a specified solver (e.g., 'glpk').
         - Print the objective function value.
         - Extract the solution and visualize the queen placements using the
           'visualize_queens' function.
        """
        if self.model is None:
            raise ValueError("The model has not been created. Run create_pyomo_model() first.")

        solver = SolverFactory("glpk")
        results = solver.solve(self.model, tee=True)

        print("Objective function value: ", self.model.obj())
        print("Results:", results)

        queens = []
        for i in self.model.squares:
            if self.model.x[i]() == 1:
                pos = (i - 1) % 8, (i - 1) // 8  
                queens.append(chr(pos[0] + 97) + str(pos[1] + 1))

        visualize_queens(queens)


if __name__ == "__main__":
    solver = QueenCoverSolver(board_size=8)
    start_time = time.time()
    coverage_matrix = solver.generate_coverage_matrix()
    print("Coverage matrix computed in:", time.time() - start_time, "seconds")
    model = solver.create_pyomo_model(coverage_matrix)
    solver.solve_and_visualize()
