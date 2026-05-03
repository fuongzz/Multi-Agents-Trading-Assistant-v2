import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

class TechnicalAnalysis:
    """
    Technical Analysis wrapper for pta_reload.
    
    Provides a simple interface to access all technical indicators across 
    different categories by dynamically loading them from module files.
    """
    
    def __init__(self):
        self._categories = [
            "momentum", "overlap", "statistics", 
            "trend", "volatility", "volume"
        ]
        # Store indicators by category
        self._indicators_by_category: Dict[str, List[str]] = {cat: [] for cat in self._categories}
        # Store indicator metadata
        self._indicator_info: Dict[str, Dict[str, Any]] = {}
        
        # Load all indicators
        self._load_indicators()
    
    def _load_indicators(self):
        """Load all indicator modules from each category."""
        # Get the package directory
        package_dir = Path(__file__).parent
        
        # Process each category
        for category in self._categories:
            category_path = package_dir / category
            if not category_path.is_dir():
                continue
                
            # Process each module in the category
            for module_file in category_path.glob("*.py"):
                if module_file.name == "__init__.py":
                    continue
                    
                # Get the module name without extension
                module_name = module_file.stem
                
                # Import the module
                try:
                    # Construct the full module path
                    module_path = f"pta_reload.{category}.{module_name}"
                    
                    # Reload the module if it's already imported
                    if module_path in sys.modules:
                        module = importlib.reload(sys.modules[module_path])
                    else:
                        module = importlib.import_module(module_path)
                    
                    # Get the main function from the module
                    if hasattr(module, module_name):
                        indicator_func = getattr(module, module_name)
                        
                        # Handle duplicate indicator names
                        if hasattr(self, module_name) and not module_name.startswith('__'):
                            existing_func = getattr(self, module_name)
                            if existing_func.__module__ != indicator_func.__module__:
                                # Use category_name prefix to avoid conflicts
                                prefixed_name = f"{category}_{module_name}"
                                setattr(self, prefixed_name, indicator_func)
                                self._indicators_by_category[category].append(prefixed_name)
                                self._indicator_info[prefixed_name] = {
                                    'category': category,
                                    'module': module_path,
                                    'function': module_name,
                                    'docstring': inspect.getdoc(indicator_func) or "",
                                    'signature': str(inspect.signature(indicator_func)),
                                }
                            else:
                                # Already loaded this indicator, skip
                                continue
                        else:
                            # Add the function to this class
                            setattr(self, module_name, indicator_func)
                            # Add to the indicators list for this category
                            self._indicators_by_category[category].append(module_name)
                            self._indicator_info[module_name] = {
                                'category': category,
                                'module': module_path,
                                'function': module_name,
                                'docstring': inspect.getdoc(indicator_func) or "",
                                'signature': str(inspect.signature(indicator_func)),
                            }
                except Exception as e:
                    print(f"Error loading module {module_path}: {str(e)}")
    
    def reload(self):
        """
        Reload all indicator modules.
        
        Useful during development when indicator code changes.
        
        Returns:
            Self for method chaining
        """
        # Clear existing indicators
        for indicator in self.list_indicators():
            if hasattr(self, indicator):
                delattr(self, indicator)
        
        # Reset indicator lists
        self._indicators_by_category = {cat: [] for cat in self._categories}
        self._indicator_info = {}
        
        # Reload all modules
        self._load_indicators()
        return self
    
    def list_indicators(self, category: Optional[str] = None) -> List[str]:
        """
        List available indicators, optionally filtered by category.
        
        Args:
            category: Optional category name to filter indicators
            
        Returns:
            List of indicator names
        """
        if category:
            if category not in self._categories:
                raise ValueError(f"Unknown category: {category}. Available: {self._categories}")
            return sorted(self._indicators_by_category[category])
        else:
            # Return all indicators
            all_indicators = []
            for cat in self._categories:
                all_indicators.extend(self._indicators_by_category[cat])
            return sorted(all_indicators)
    
    def get_categories(self) -> List[str]:
        """Return the list of available categories."""
        return self._categories
    
    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get metadata about a specific indicator.
        
        Args:
            indicator_name: The name of the indicator
            
        Returns:
            Dictionary with indicator metadata
        """
        if indicator_name not in self._indicator_info:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        return self._indicator_info[indicator_name]
    
    def help(self, indicator_name: Optional[str] = None):
        """
        Print help information about indicators.
        
        Args:
            indicator_name: Optional name of indicator to get help for
        """
        if indicator_name is None:
            # Print list of all indicators by category
            print("Available Technical Indicators:")
            print("-" * 30)
            for category in sorted(self._categories):
                indicators = self.list_indicators(category)
                if indicators:
                    print(f"\n{category.upper()} ({len(indicators)}):")
                    # Print in columns
                    col_width = max(len(ind) for ind in indicators) + 2
                    num_cols = max(1, 80 // col_width)
                    for i in range(0, len(indicators), num_cols):
                        row = indicators[i:i+num_cols]
                        print("  " + "".join(ind.ljust(col_width) for ind in row))
            print("\nUse ta.help('indicator_name') for more information on a specific indicator.")
            return
        
        # Print help for specific indicator
        if indicator_name not in self._indicator_info:
            print(f"Unknown indicator: {indicator_name}")
            print(f"Available indicators: {', '.join(sorted(self._indicator_info.keys()))}")
            return
        
        info = self._indicator_info[indicator_name]
        print(f"Indicator: {indicator_name}")
        print(f"Category: {info['category']}")
        print(f"Module: {info['module']}")
        print(f"Signature: {indicator_name}{info['signature']}")
        print("\nDescription:")
        print(info['docstring'] or "No description available.")
    
    def __dir__(self):
        """Return list of available methods including indicators."""
        methods = [attr for attr in super().__dir__() 
                  if not attr.startswith('_') or attr in ('__call__',)]
        methods.extend(self.list_indicators())
        return sorted(methods)
    
    def __repr__(self):
        """String representation of the TechnicalAnalysis class."""
        indicator_count = sum(len(inds) for inds in self._indicators_by_category.values())
        return f"<TechnicalAnalysis: {indicator_count} indicators in {len(self._categories)} categories>"

# Create a singleton instance
ta = TechnicalAnalysis()
