from cytoolz import compose


# 合一操作
def unify(substitution, x, y):
    if substitution is None:
        return None
    elif x == y:
        return substitution
    elif is_variable(x):
        return unify_var(substitution, x, y)
    elif is_variable(y):
        return unify_var(substitution, y, x)
    elif is_compound(x) and is_compound(y):
        return unify(compose(substitution, unify(substitution, x.head, y.head)), x.tail, y.tail)
    else:
        return None


# 对变量进行合一替换
def unify_var(substitution, var, x):
    if var in substitution:
        return unify(substitution, substitution[var], x)
    elif x in substitution:
        return unify(substitution, var, substitution[x])
    else:
        substitution[var] = x
        return substitution


# 消解操作
def resolve(clause1, clause2):
    for literal1 in clause1:
        for literal2 in clause2:
            # print("literal1.predicate:", literal1.predicate)
            # print("literal2.predicate:", literal2.predicate)
            # print("literal1.negated:", literal1.negated)
            # print("literal2.negated:", literal2.negated)
            # print(literal1.predicate == literal2.predicate and literal1.negated != literal2.negated)
            if literal1.predicate == literal2.predicate and literal1.negated != literal2.negated:
                substitution = unify({}, literal1.arguments, literal2.arguments)
                if substitution is not None:
                    resolved_clause = merge(clause1, clause2, literal1, literal2)
                    resolved_clause = apply_substitution(resolved_clause, substitution)
                    print("Clauses:", clause1, clause2)
                    print("Substitution:", substitution)
                    print("Result:", resolved_clause)
                    return {
                        "clauses": (clause1, clause2),
                        "substitution": substitution,
                        "result": resolved_clause
                    }
                else:
                    print("Clauses:", clause1, clause2)
                    print("Substitution:", {})
                    print("Result: No resolution")
                    return {
                        "clauses": (clause1, clause2),
                        "substitution": {},
                        "result": None
                    }
    return None


# 合并子句
def merge(clause1, clause2, literal1, literal2):
    merged_clause = list(clause1) + list(clause2)
    merged_clause.remove(literal1)
    merged_clause.remove(literal2)
    return merged_clause


# 合并子句
def apply_substitution(clause, substitution):
    substituted_clause = []
    for literal in clause:
        substituted_literal = Literal(literal.predicate, literal.arguments)
        substituted_literal = substitute_arguments(substituted_literal, substitution)
        substituted_clause.append(substituted_literal)
    return substituted_clause


# 应用合一替换到文字的参数
def substitute_arguments(literal, substitution):
    for i, arg in enumerate(literal.arguments):
        if arg in substitution:
            literal.arguments[i] = substitution[arg]
    return literal


# 判断是否为变量
def is_variable(term):
    return term[0].islower()


# 判断是否为复合结构
def is_compound(term):
    return isinstance(term, Compound)


class Literal:
    def __init__(self, predicate, arguments, negated=False):
        self.predicate = predicate
        self.arguments = arguments
        self.negated = negated


class Compound:
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail


def clauses1():
    # 定义子句
    clauses = [[Literal("howls", ["X"])],
               [Literal("has", ["Y", "Z"]), Literal("has", ["Y"], negated=True), Literal("has", ["Z"], negated=True)],
               [Literal("light_sleeper", ["X"]), Literal("has", ["X"], negated=True)],
               [Literal("has", ["John", "cat"]), Literal("has", ["John", "hound"])]]

    # 初始化计数器和结果记录
    step_count = 0
    result_log = []

    # 应用resolution算法
    resolved = False
    while not resolved:
        new_clauses = []
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                step_count += 1  # 计数器加1
                result_log.append(f"Step {step_count}: Resolving clauses {i} and {j}")

                result = resolve(clauses[i], clauses[j])
                # print("result: ", result)
                if result is not None:
                    if result["result"] is None:
                        result_log.append("找到矛盾，约翰没有任何老鼠。")
                        resolved = True
                        break
                    new_clauses.append(result["result"])
                    result_log.append(f"Step {step_count}: Generated new clause: {result['result']}")
            if resolved:
                break
        if len(new_clauses) == 0:
            result_log.append("没有找到矛盾，陈述为真。")
            resolved = True
        else:
            clauses += new_clauses

    # 打印最终结果和总运算步数
    print("\n".join(result_log))
    print("Total steps:", step_count)


def clauses2():
    # 定义子句 
    clauses = [[Literal("searched", ["X"]), Literal("VIP", ["X"], negated=True)], [
        Literal("entered", ["X"]), Literal("drug_dealer", ["X"])], [
                   Literal("searched", ["X"]), Literal("drug_dealer", ["Y"]), Literal("searched_by", ["X", "Y"])], [
                   Literal("VIP", ["X"]), Literal("drug_dealer", ["X"], negated=True)], [
                   Literal("customs_official", ["X"]), Literal("drug_dealer", ["X"])], [
                   Literal("searched", ["X"], negated=True)], [
                   Literal("entered", ["X"]), Literal("drug_dealer", ["X"], negated=True)]]

    # 初始化计数器和结果记录
    step_count = 0
    result_log = []

    # 应用resolution算法
    resolved = False
    while not resolved:
        new_clauses = []
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                step_count += 1  # 计数器加1
                result_log.append(f"Step {step_count}: Resolving clauses {i} and {j}")

                result = resolve(clauses[i], clauses[j])
                if result is not None:
                    if result["result"] is None:
                        result_log.append("找到矛盾，证明成功。")
                        resolved = True
                        break
                    new_clauses.append(result["result"])
                    result_log.append(f"Step {step_count}: Generated new clause: {result['result']}")
            if resolved:
                break
        if len(new_clauses) == 0:
            result_log.append("没有找到矛盾，证明失败。")
            resolved = True
        else:
            clauses += new_clauses

    # 打印最终结果和总运算步数
    print("\n".join(result_log))
    print("Total steps:", step_count)


if __name__ == "__main__":
    print("QUESTION1: Howling Hounds")
    clauses1()
    print()
    print("QUESTION2: Howling Hounds")
    clauses2()
