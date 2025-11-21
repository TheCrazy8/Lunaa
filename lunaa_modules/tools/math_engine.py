"""Mathematical and graphing capabilities"""
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats, optimize
    _MATH_AVAILABLE = True
except ImportError:
    _MATH_AVAILABLE = False

class MathEngine:
    def __init__(self):
        self.figures = []
        
    def calculate(self, expression: str):
        """Evaluate mathematical expression"""
        if not _MATH_AVAILABLE:
            return "Math dependencies not installed"
        
        try:
            # Safe evaluation with numpy functions
            result = eval(expression, {"__builtins__": {}}, {
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "log": np.log,
                "exp": np.exp,
                "pi": np.pi,
                "e": np.e
            })
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"
    
    def plot_function(self, expression: str, x_range: tuple = (-10, 10), filename: str = 'plot.png'):
        """Plot a mathematical function"""
        if not _MATH_AVAILABLE:
            return "Math dependencies not installed"
        
        try:
            x = np.linspace(x_range[0], x_range[1], 1000)
            y = eval(expression, {"__builtins__": {}}, {
                "x": x,
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "log": np.log,
                "exp": np.exp,
                "pi": np.pi,
                "e": np.e
            })
            
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
