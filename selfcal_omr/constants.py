from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:
    green: tuple = (0, 255, 0)
    red: tuple = (0, 0, 255)
    yellow: tuple = (0,255,255)
    white: tuple = (255, 255, 255)
    
@dataclass(frozen=True)
class RenderSize:
    width: int = 600
    height: int = 800
