# app.py (최종 수정 완료 버전)

# foundations/factor_structure.py

import re
from typing import List, Union, Any

# --- AST 노드 클래스 정의 ---
# 팩터 공식을 구조적으로 표현하기 위한 기본 단위입니다.

class ASTNode:
    """모든 AST 노드의 기본이 되는 추상 클래스입니다."""
    def __repr__(self):
        return self.__str__()

class OperatorNode(ASTNode):
    """연산자(e.g., rank, +, -)를 나타내는 노드입니다."""
    def __init__(self, op: str, children: List[ASTNode]):
        self.op = op
        self.children = children

    def __str__(self):
        # 자식 노드들을 콤마로 구분하여 문자열로 표현합니다.
        children_str = ', '.join(map(str, self.children))
        return f"{self.op}({children_str})"

class VariableNode(ASTNode):
    """변수(e.g., close, volume)를 나타내는 노드입니다."""
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"${self.name}"

class LiteralNode(ASTNode):
    """리터럴(상수 값, e.g., 10, 'sector')을 나타내는 노드입니다."""
    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        # 값이 문자열인 경우 따옴표로 감싸줍니다.
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


# --- 팩터 공식 파서 ---

class FactorParser:
    """
    팩터 공식 문자열을 AST로 변환하는 파서입니다.
    재귀 하강 파싱(Recursive Descent Parsing) 방식을 사용하여 구현합니다.
    """
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, formula: str) -> ASTNode:
        """
        메인 파싱 함수. 공식 문자열을 토큰화하고 파싱을 시작합니다.

        Args:
            formula (str): AST로 변환할 팩터 공식입니다.

        Returns:
            ASTNode: 파싱 결과로 생성된 AST의 루트 노드입니다.
        """
        self.tokens = self._tokenize(formula)
        self.pos = 0
        ast = self._parse_expression()
        if self.pos < len(self.tokens):
            # 파싱이 끝났는데 토큰이 남아있으면 에러
            raise ValueError("파싱 후 남은 토큰이 있습니다: {}".format(self.tokens[self.pos:]))
        return ast

    def _tokenize(self, formula: str) -> List[str]:
        """
        정규표현식을 사용하여 공식 문자열을 토큰 리스트로 변환합니다.
        """
        # 연산자, 괄호, 변수, 숫자 등을 개별 토큰으로 분리합니다.
        token_regex = re.compile(r'([A-Za-z_][A-Za-z0-9_\.]*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||[()+\-*/?^:,<>])')
        tokens = token_regex.findall(formula)
        # '$'는 변수 식별자로 사용되므로 제거합니다.
        return [token for token in tokens if token.strip() and token != '$']

    def _peek(self) -> Union[str, None]:
        """현재 위치의 토큰을 확인합니다 (소비하지 않음)."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        """현재 위치의 토큰을 소비하고 다음으로 이동합니다."""
        token = self._peek()
        self.pos += 1
        return token

    def _parse_primary(self) -> ASTNode:
        """가장 기본적인 표현 단위(원자)를 파싱합니다: 숫자, 변수, 괄호 표현식, 함수 호출"""
        token = self._peek()

        # 음수 처리
        if token == '-':
            self._consume() # '-' 소비
            # 단항 연산자(negation)로 처리
            return OperatorNode('neg', [self._parse_primary()])

        # 숫자 리터럴
        if token.replace('.', '', 1).isdigit():
            return LiteralNode(float(self._consume()))

        # 변수 또는 함수 호출
        if token.isalnum() or '_' in token or '.' in token:
            self._consume() # 변수/함수명 소비
            if self._peek() == '(': # 함수 호출인 경우
                self._consume() # '(' 소비
                args = []
                if self._peek() != ')':
                    while True:
                        args.append(self._parse_expression())
                        if self._peek() == ')':
                            break
                        if self._peek() != ',':
                            raise ValueError("함수 인자 사이에 콤마(,)가 필요합니다.")
                        self._consume() # ',' 소비
                self._consume() # ')' 소비
                return OperatorNode(token, args)
            else: # 변수인 경우
                return VariableNode(token)

        # 괄호 표현식
        if token == '(':
            self._consume() # '(' 소비
            expr = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("괄호가 닫히지 않았습니다.")
            return expr

        raise ValueError(f"예상치 못한 토큰입니다: {token}")

    def _parse_binary_op(self, parse_next_level, ops: List[str]) -> ASTNode:
        """이항 연산자를 처리하는 헬퍼 함수 (e.g., +, -, *, /)"""
        node = parse_next_level()
        while self._peek() in ops:
            op = self._consume()
            right = parse_next_level()
            node = OperatorNode(op, [node, right])
        return node

    def _parse_power(self) -> ASTNode:
        return self._parse_binary_op(self._parse_primary, ['^'])

    def _parse_multiplicative(self) -> ASTNode:
        return self._parse_binary_op(self._parse_power, ['*', '/'])

    def _parse_additive(self) -> ASTNode:
        return self._parse_binary_op(self._parse_multiplicative, ['+', '-'])
        
    def _parse_comparison(self) -> ASTNode:
        return self._parse_binary_op(self._parse_additive, ['>', '<', '>=', '<=', '==', '!='])

    def _parse_logical(self) -> ASTNode:
        return self._parse_binary_op(self._parse_comparison, ['&&', '||'])

    def _parse_expression(self) -> ASTNode:
        """
        가장 낮은 우선순위의 연산자부터 파싱을 시작합니다.
        여기서는 삼항 연산자(a ? b : c)를 처리합니다.
        """
        node = self._parse_logical()
        if self._peek() == '?':
            self._consume() # '?' 소비
            true_expr = self._parse_expression()
            if self._consume() != ':':
                raise ValueError("삼항 연산자에 ':'가 필요합니다.")
            false_expr = self._parse_expression()
            # AlphaAgent 논문에 따라 'If' 연산자로 변환
            return OperatorNode('If', [node, true_expr, false_expr])
        return node


# --- 팩터 복잡도 분석기 ---

class ComplexityAnalyzer:
    """AST를 기반으로 팩터의 복잡도를 계산합니다."""

    def calculate_symbolic_length(self, node: ASTNode) -> int:
        """
        팩터의 상징적 길이(Symbolic Length)를 계산합니다. AST의 총 노드 수와 같습니다.

        Args:
            node (ASTNode): 복잡도를 계산할 AST의 노드입니다.

        Returns:
            int: AST의 총 노드 수.
        """
        if isinstance(node, OperatorNode):
            # 현재 노드(1) + 모든 자식 노드의 길이 합
            return 1 + sum(self.calculate_symbolic_length(child) for child in node.children)
        # 변수 또는 리터럴 노드는 자식이 없으므로 길이가 1
        return 1

    def calculate_parameter_count(self, node: ASTNode) -> int:
        """
        팩터에 사용된 하이퍼파라미터(숫자 상수)의 개수를 계산합니다.

        Args:
            node (ASTNode): 복잡도를 계산할 AST의 노드입니다.

        Returns:
            int: 숫자 리터럴 노드의 총 개수.
        """
        count = 0
        if isinstance(node, LiteralNode) and isinstance(node.value, (int, float)):
            count = 1
        
        if isinstance(node, OperatorNode):
            # 모든 자식 노드의 파라미터 개수를 재귀적으로 더함
            count += sum(self.calculate_parameter_count(child) for child in node.children)
            
        return count

# --- 테스트 코드 ---
if __name__ == '__main__':
    # Alpha#5 팩터를 예시로 사용
    sample_formula = "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"
    
    print("="*50)
    print(f"테스트 공식: {sample_formula}")
    print("="*50)

    # 1. 파서 테스트
    parser = FactorParser()
    try:
        ast_tree = parser.parse(sample_formula)
        print("\n[파서 테스트]")
        print("파싱 결과 (AST):")
        print(ast_tree)
    except ValueError as e:
        print(f"파싱 오류: {e}")

    # 2. 복잡도 분석기 테스트
    if 'ast_tree' in locals():
        analyzer = ComplexityAnalyzer()
        
        sl = analyzer.calculate_symbolic_length(ast_tree)
        pc = analyzer.calculate_parameter_count(ast_tree)
        
        print("\n[복잡도 분석기 테스트]")
        print(f"상징적 길이 (Symbolic Length): {sl}")
        print(f"파라미터 개수 (Parameter Count): {pc}")
        print("="*50)

