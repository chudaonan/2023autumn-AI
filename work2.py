# A class to represent a literal in a clause
class Literal:
    def __init__(self, name, args, negated):
        self.name = name  # the predicate or function name
        self.args = args  # a list of arguments (constants, variables or functions)
        self.negated = negated  # a boolean value indicating whether the literal is negated

    def __str__(self):
        # a string representation of the literal
        s = ""
        if self.negated:
            s += "¬"
        s += self.name + "("
        for i, arg in enumerate(self.args):
            s += str(arg)
            if i < len(self.args) - 1:
                s += ","
        s += ")"
        return s

    def __eq__(self, other):
        # check if two literals are equal (same name, arguments and negation)
        if isinstance(other, Literal):
            return self.name == other.name and self.args == other.args and self.negated == other.negated
        return False

    def __hash__(self):
        # a hash function for literals
        return hash((self.name, tuple(self.args), self.negated))

    def is_ground(self):
        # check if the literal is ground (has no variables)
        for arg in self.args:
            if isinstance(arg, Variable):
                return False
        return True

    def substitute(self, theta):
        # apply a substitution to the literal and return a new literal
        new_args = []
        for arg in self.args:
            if isinstance(arg, Variable) and arg in theta:
                new_args.append(theta[arg])
            else:
                new_args.append(arg)
        return Literal(self.name, new_args, self.negated)

    def complement(self, other):
        # check if two literals are complementary (same name, arguments and opposite negation)
        return self.name == other.name and self.args == other.args and self.negated != other.negated


# A class to represent a clause
class Clause:
    def __init__(self, literals):
        self.literals = literals  # a set of literals

    def __str__(self):
        # a string representation of the clause
        s = "{"
        for i, literal in enumerate(self.literals):
            s += str(literal)
            if i < len(self.literals) - 1:
                s += " ∨ "
        s += "}"
        return s

    def __eq__(self, other):
        # check if two clauses are equal (have the same literals)
        if isinstance(other, Clause):
            return self.literals == other.literals
        return False

    def __hash__(self):
        # a hash function for clauses
        return hash(frozenset(self.literals))

    def is_empty(self):
        # check if the clause is empty
        return len(self.literals) == 0

    def is_unit(self):
        # check if the clause is unit (has only one literal)
        return len(self.literals) == 1

    def is_ground(self):
        # check if the clause is ground (has no variables)
        for literal in self.literals:
            if not literal.is_ground():
                return False
        return True

    def substitute(self, theta):
        # apply a substitution to the clause and return a new clause
        new_literals = set()
        for literal in self.literals:
            new_literals.add(literal.substitute(theta))
        return Clause(new_literals)

    def resolve(self, other):
        # resolve two clauses and return a new clause
        for literal in self.literals:
            for other_literal in other.literals:
                if literal.complement(other_literal):
                    new_literals = self.literals.union(other.literals)
                    new_literals.remove(literal)
                    new_literals.remove(other_literal)
                    return Clause(new_literals)
        return None


# A class to represent a constant
class Constant:
    def __init__(self, name):
        self.name = name  # the constant name

    def __str__(self):
        # a string representation of the constant
        return self.name

    def __eq__(self, other):
        # check if two constants are equal (have the same name)
        if isinstance(other, Constant):
            return self.name == other.name
        return False

    def __hash__(self):
        # a hash function for constants
        return hash(self.name)


# A class to represent a variable
class Variable:
    def __init__(self, name):
        self.name = name  # the variable name

    def __str__(self):
        # a string representation of the variable
        return self.name

    def __eq__(self, other):
        # check if two variables are equal (have the same name)
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __hash__(self):
        # a hash function for variables
        return hash(self.name)


# A class to represent a function
class Function:
    def __init__(self, name, args):
        self.name = name  # the function name
        self.args = args  # a list of arguments (constants, variables or functions)

    def __str__(self):
        # a string representation of the function
        s = self.name + "("
        for i, arg in enumerate(self.args):
            s += str(arg)
            if i < len(self.args) - 1:
                s += ","
        s += ")"
        return s

    def __eq__(self, other):
        # check if two functions are equal (have the same name and arguments)
        if isinstance(other, Function):
            return self.name == other.name and self.args == other.args
        return False

    def __hash__(self):
        # a hash function for functions
        return hash((self.name, tuple(self.args)))


# A function to unify two expressions
def unify(x, y, theta):
    # x and y are expressions (constants, variables, functions or literals)
    # theta is the substitution so far (a dictionary)
    if theta is None:
        return None
    elif x == y:
        return theta
    elif isinstance(x, Variable):
        return unify_var(x, y, theta)
    elif isinstance(y, Variable):
        return unify_var(y, x, theta)
    elif isinstance(x, Function) and isinstance(y, Function):
        return unify(x.args, y.args, unify(x.name, y.name, theta))
    elif isinstance(x, Literal) and isinstance(y, Literal):
        return unify(x.args, y.args, unify(x.name, y.name, unify(x.negated, y.negated, theta)))
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return unify(x[1:], y[1:], unify(x[0], y[0], theta))
    else:
        return None


# A helper function to unify a variable with an expression
def unify_var(var, x, theta):
    # var is a variable
    # x is an expression
    # theta is the substitution so far
    if var in theta:
        return unify(theta[var], x, theta)
    elif x in theta:
        return unify(var, theta[x], theta)
    elif occur_check(var, x):
        return None
    else:
        return {**theta, var: x}


# A helper function to check if a variable occurs in an expression
def occur_check(var, x):
    # var is a variable
    # x is an expression
    if var == x:
        return True
    elif isinstance(x, Function):
        for arg in x.args:
            if occur_check(var, arg):
                return True
    elif isinstance(x, Literal):
        for arg in x.args:
            if occur_check(var, arg):
                return True
    return False


# A function to standardize the variables in a clause
def standardize(clause, index):
    # clause is a clause
    # index is an integer to append to the variable names
    theta = {}
    for literal in clause.literals:
        for arg in literal.args:
            if isinstance(arg, Variable):
                if arg not in theta:
                    theta[arg] = Variable(arg.name + str(index))
    return clause.substitute(theta)


# A function to apply the unit preference strategy to a set of clauses
def unit_preference(clauses):
    # clauses is a set of clauses
    # return a list of clauses ordered by the number of literals (ascending)
    return sorted(clauses, key=lambda clause: len(clause.literals))


# A function to implement the resolution algorithm with unit preference
def resolution(clauses, query):
    # clauses is a set of clauses
    # query is a clause to be proved
    # return True if the query can be proved, False otherwise
    clauses = clauses.copy()  # make a copy of the clauses
    clauses.add(query)  # add the negated query to the clauses
    index = 0  # a counter for standardizing variables
    while True:
        clauses = unit_preference(clauses)  # order the clauses by the number of literals
        new = set()  # a set of new clauses
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                c1 = clauses[i]  # the first clause
                c2 = clauses[j]  # the second clause
                c1 = standardize(c1, index)  # standardize the variables in c1
                c2 = standardize(c2, index)  # standardize the variables in c2
                index += 1  # increment the counter
                resolvents = resolve(c1, c2)  # resolve the two clauses
                if resolvents is not None:  # if the resolution is successful
                    if resolvents.is_empty():  # if the resolvent is empty
                        return True  # return True (proof successful)
                    new.add(resolvents)  # add the resolvent to the new clauses
        if new.issubset(clauses):  # if no new clauses are generated
            return False  # return False (proof failed)
        clauses = clauses.union(new)  # add the new clauses to the original clauses


# A function to resolve two clauses
def resolve(c1, c2):
    # c1 and c2 are clauses
    # return a clause that is the resolvent of c1 and c2, or None if no resolution is possible
    for literal in c1.literals:
        for other_literal in c2.literals:
            if literal.complement(other_literal):  # if the two literals are complementary
                theta = unify(literal, other_literal, {})  # try to unify them
                if theta is not None:  # if the unification is successful
                    new_literals = c1.literals.union(c2.literals)  # combine the literals of the two clauses
                    new_literals.remove(literal)  # remove the first literal
                    new_literals.remove(other_literal)  # remove the second literal
                    new_clause = Clause(new_literals)  # create a new clause
                    new_clause = new_clause.substitute(theta)  # apply the substitution to the new clause
                    return new_clause  # return the new clause
    return None  # return None if no resolution is possible


# Test case 1
clauses = {
    Clause({Literal("Austinite", [Variable("x")], False), Literal("conservative", [Variable("x")], True),
            Literal("loves", [Variable("x"), Constant("armadillo")], False)}),
    Clause({Literal("wears", [Variable("x"), Constant("maroon-and-white shirts")], False),
            Literal("Aggie", [Variable("x")], False), Literal("conservative", [Variable("x")], True)}), # 在这里添加了一个Literal("conservative", [Variable("x")], True)
    Clause({Literal("Aggie", [Variable("x")], False), Literal("loves", [Variable("x"), Variable("y")], False),
            Literal("dog", [Variable("y")], False)}),
    Clause({Literal("loves", [Variable("x"), Variable("y")], False), Literal("dog", [Variable("y")], False),
            Literal("loves", [Variable("x"), Variable("z")], True), Literal("armadillo", [Variable("z")], False)}),
    Clause({Literal("Austinite", [Constant("Clem")], False),
            Literal("wears", [Constant("Clem"), Constant("maroon-and-white shirts")], False)})
}
query = Clause({Literal("conservative", [Variable("x")], False), Literal("Austinite", [Variable("x")], False)})
print("Test case 1: ", resolution(clauses, query))

# Test case 2
# clauses = {
#     Clause({Literal("buys", [Variable("x"), Constant("carrots by the bushel")], False),
#             Literal("owns", [Variable("x"), Variable("y")], False), Literal("rabbit", [Variable("y")], False)}),
#     Clause({Literal("buys", [Variable("x"), Constant("carrots by the bushel")], False),
#             Literal("owns", [Variable("x"), Constant("grocery store")], False)}),
#     Clause({Literal("dog", [Variable("x")], False), Literal("chases", [Variable("x"), Variable("y")], False),
#             Literal("rabbit", [Variable("y")], False)}),
#     Clause({Literal("owns", [Variable("x"), Variable("y")], False), Literal("rabbit", [Variable("y")], False),
#             Literal("hates", [Variable("x"), Variable("z")], False),
#             Literal("chases", [Variable("z"), Variable("y")], False)}),
#     Clause({Literal("owns", [Constant("John"), Variable("x")], False), Literal("dog", [Variable("x")], False)}),
#     Clause({Literal("hates", [Variable("x"), Variable("y")], False),
#             Literal("owns", [Variable("z"), Variable("y")], False),
#             Literal("dates", [Variable("x"), Variable("z")], True)}),
#     Clause({Literal("buys", [Constant("Mary"), Constant("carrots by the bushel")], False)})
# }
# query = Clause({Literal("owns", [Constant("Mary"), Constant("grocery store")], True),
#                 Literal("dates", [Constant("Mary"), Constant("John")], False)})
# print("Test case 2: ", resolution(clauses, query))
