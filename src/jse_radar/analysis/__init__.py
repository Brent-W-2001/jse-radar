"""
Analysis layer — signals, macro regime classification, and correlation analysis.

Import shortcuts:
    from jse_radar.analysis.signals import SignalEngine
    from jse_radar.analysis.macro_regime import MacroRegimeClassifier
    from jse_radar.analysis.correlation import CorrelationAnalyser
"""

from jse_radar.analysis.signals import SignalEngine
from jse_radar.analysis.macro_regime import MacroRegimeClassifier
from jse_radar.analysis.correlation import CorrelationAnalyser

__all__ = ["SignalEngine", "MacroRegimeClassifier", "CorrelationAnalyser"]