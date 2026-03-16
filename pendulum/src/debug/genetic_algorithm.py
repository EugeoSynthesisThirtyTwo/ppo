from __future__ import annotations
from abc import abstractmethod
import numpy as np
import random


class Individu:
    @abstractmethod
    def get_score(self) -> float:
        pass

    @abstractmethod
    def offspring(self, other: Individu) -> Individu:
        pass

    @abstractmethod
    def mutate(self, mutation_rate: float):
        pass


class Population:
    def __init__(
        self,
        size: int = 1000,
        mutation_rate: float = 0.01,
        survival_rate: float = 0.3,
        purely_random_born_rate: float = 0.05,
        minimize_score: bool = False,
    ):
        """
        size: le nombre d'individus à chaque génération
        mutation_rate: le taux de mutation moyen. Le taux de mutation est généré avec la loi exponentielle
        survival_rate: le taux de survie pendant la phase de séléction
        purely_random_born_rate: le taux d'individus généré 100% aléatoirement à chaque génération
        minimize_score: si c'est vrai alors on minimize le score des individus, si c'est faux alors on maximize
        """
        self.individus: list[tuple[Individu, float]] = []  # (individu, score)
        self.generation: int = 0
        self.population_size: int = size
        self.mutation_rate: float = mutation_rate
        self.survival_rate: float = survival_rate
        self.purely_random_born_rate: float = purely_random_born_rate
        self.minimize_score: bool = minimize_score
    
    def first_generation(self):
        while len(self.individus) < self.population_size:
            self.individus.append((self.generate_new_individu(), None)) # type: ignore

        self.individus = self.evaluate(self.individus)

    @abstractmethod
    def generate_new_individu(self) -> Individu:
        pass

    def next_generation(self):
        if len(self.individus) == 0:
            self.first_generation()
        
        self.individus = self.select()

        offspring = self.make_offspring()
        offspring = self.evaluate(offspring)
        self.individus = self.individus + offspring

        purely_random = self.create_purely_random()
        purely_random = self.evaluate(purely_random)
        self.individus = self.individus + purely_random

        self.sort()

        self.generation += 1
        return self.generation

    def select(self) -> list[tuple[Individu, float]]:
        return self.individus[: int(self.survival_rate * self.population_size)]

    def make_offspring(self) -> list[tuple[Individu, float]]:
        offspring: list[tuple[Individu, float]] = []
        size_offspring = int(
            (1 - self.purely_random_born_rate - self.survival_rate)
            * self.population_size
        )
        mutation_rate = self.generate_mutation_rate(size_offspring)

        for i in range(size_offspring):
            a = random.choice(self.individus)[0]
            b = a

            while a is b:
                b = random.choice(self.individus)[0]

            children = a.offspring(b)
            children.mutate(mutation_rate[i])
            offspring.append((children, None)) # type: ignore

        return offspring

    def create_purely_random(self, size: int = -1) -> list[tuple[Individu, float]]:
        if size == -1:
            size = self.population_size - len(self.individus)

        return [(self.generate_new_individu(), None) for _ in range(size)] # type: ignore

    def evaluate(
        self, individus: list[tuple[Individu, float]] | None = None
    ) -> list[tuple[Individu, float]]:
        if individus is None:
            individus = self.individus

        result = [None] * len(individus)

        for i, individu in enumerate(individus):
            score = individu[0].get_score()
            result[i] = (individu[0], score) # type: ignore

        return result # type: ignore

    def sort(self):
        self.individus.sort(key=lambda x: x[1], reverse=not self.minimize_score)

    def generate_mutation_rate(self, shape) -> np.ndarray:
        return np.random.exponential(self.mutation_rate, shape)

    def best(self) -> tuple[Individu, float]:
        if len(self.individus) == 0:
            self.first_generation()
        
        return self.individus[0]


def test():
    import matplotlib.pyplot as plt
    from math import sin, sqrt

    def score(x: float):
        y = sin(x) + 0.2 * x + 0.5 * sin(0.5 * x)
        return sqrt(y * y) + 0.1

    def np_score(x: np.ndarray):
        y = np.sin(x) + 0.2 * x + 0.5 * np.sin(0.5 * x)
        return np.sqrt(y * y) + 0.1

    class IndividuTest(Individu):
        def __init__(self, x: float):
            self.x = x

        def get_score(self) -> float:
            return score(self.x)

        def offspring(self, other: IndividuTest) -> IndividuTest: # type: ignore
            return IndividuTest((self.x + other.x) / 2)

        def mutate(self, mutation_rate: float):
            if random.randint(0, 1) == 0:
                self.x -= mutation_rate
            else:
                self.x += mutation_rate

    class PopulationTest(Population):
        def generate_new_individu(self) -> IndividuTest:
            return IndividuTest(random.random() * 24 - 12)

    population = PopulationTest(size=5, mutation_rate=0.01, minimize_score=True)

    x = np.linspace(-12, 12, 1000)
    y = np_score(x)

    for i in range(30):
        plt.plot(x, y)

        x_individus = [individu[0].x for individu in population.individus] # type: ignore
        y_individus = [individu[1] for individu in population.individus]
        plt.scatter(x_individus, y_individus)
        plt.show()

        population.next_generation()


if __name__ == "__main__":
    test()
