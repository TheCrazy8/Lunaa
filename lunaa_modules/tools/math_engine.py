"""Mathematical and graphing capabilities"""
import ast
import operator

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats, optimize
    _MATH_AVAILABLE = True
except ImportError:
    _MATH_AVAILABLE = False
    np = None

class MathEngine:
    def __init__(self):
        self.figures = []
        # Safe operators for math evaluation
        self.safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
    def _safe_eval(self, expression: str, variables: dict = None):
        """Safely evaluate mathematical expression using AST"""
        if not _MATH_AVAILABLE:
            return None
        
        try:
            # Parse the expression
            node = ast.parse(expression, mode='eval').body
            return self._eval_node(node, variables or {})
        except Exception as e:
            # If AST parsing fails, return error instead of fallback
            raise ValueError(f"Cannot evaluate expression: {e}")
    
    def _eval_node(self, node, variables):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            # Allow numpy constants
            allowed_constants = {'pi', 'e'}
            if node.id in allowed_constants and hasattr(np, node.id):
                return getattr(np, node.id)
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.safe_operators:
                left = self._eval_node(node.left, variables)
                right = self._eval_node(node.right, variables)
                return self.safe_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.safe_operators:
                operand = self._eval_node(node.operand, variables)
                return self.safe_operators[op_type](operand)
        elif isinstance(node, ast.Call):
            # Allow specific numpy functions only from whitelist
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                allowed_funcs = {
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'sqrt': np.sqrt,
                    'log': np.log,
                    'exp': np.exp,
                    'abs': abs,
                    'min': min,
                    'max': max,
                }
                if func_name in allowed_funcs:
                    args = [self._eval_node(arg, variables) for arg in node.args]
                    return allowed_funcs[func_name](*args)
                else:
                    raise ValueError(f"Function not allowed: {func_name}")
        raise ValueError(f"Unsafe operation: {ast.dump(node)}")
        
    def calculate(self, expression: str):
        """Evaluate mathematical expression"""
        if not _MATH_AVAILABLE:
            return "Math dependencies not installed"
        
        try:
            result = self._safe_eval(expression)
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"
    
    def plot_function(self, expression: str, x_range: tuple = (-10, 10), filename: str = 'plot.png'):
        """Plot a mathematical function"""
        if not _MATH_AVAILABLE:
            return "Math dependencies not installed"
        
        try:
            x = np.linspace(x_range[0], x_range[1], 1000)
            # Evaluate with x as a variable
            y = self._safe_eval(expression, {"x": x})
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Plot of {expression}')
            plt.savefig(filename)
            plt.close()
            
            return f"Plot saved to {filename}"
        except Exception as e:
            return f"Error plotting function: {e}"
    
    def statistics(self, data: list):
        """Calculate statistics for data"""
        if not _MATH_AVAILABLE:
            return "Math dependencies not installed"
        
        try:
            arr = np.array(data)
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        except Exception as e:
            return f"Error calculating statistics: {e}"
