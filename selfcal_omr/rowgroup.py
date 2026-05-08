from dataclasses import dataclass, field
import math
from operator import attrgetter
import statistics
from selfcal_omr.helper import Point


@dataclass
class RowGroup:
    id: int
    points : list = field(default_factory=list)
    next = None
    topleft: Point = field(default_factory=lambda: Point(0, 0))
    bottomright: Point = field(default_factory=lambda: Point(0, 0))
    def __init__(self, id: int, points: list[Point], next = None, prev = None):
        self.id = id
        self.points = points
        self.next = next
        self.prev = prev
        self.topleft = points[0].copy()
        self.bottomright = points[-1].copy()
        
    def append(self, new):
        cur = self
        while cur.next:
            cur = cur.next
        cur.next = new  
        new.prev = cur 
         
    def __iter__(self):
        cur = self
        while cur:
            yield cur
            cur = cur.next
    
    def distance(self, point:Point):
        if(len(self.points) == 1):
            return math.fabs(point.y - self.topleft.y)    
        elif(point.y < self.topleft.y):
            return math.fabs(point.y - self.topleft.y)
        elif(point.y > self.bottomright.y):
            return math.fabs(point.y - self.bottomright.y)
        return 0
    
    def mean(self):
        return statistics.mean([point.y for point in self.points])  
    
    def Devour(self, instance, dx:int, dy: int):
        if(math.fabs(self.mean() - instance.mean()) > 1.5 *dy):
            return
        if( not self.HaveCommon(instance, dx)):
            return
        #self.topleft.x  = min(self.topleft.x, instance.topleft.x)
        if (self.topleft.x > instance.topleft.x):
            self.topleft.x =  instance.topleft.x
        if (self.topleft.y > instance.topleft.y):
            self.topleft.y =  instance.topleft.y
        if (self.bottomright.x < instance.bottomright.x):
            self.bottomright.x =  instance.bottomright.x
        if (self.bottomright.y < instance.bottomright.y):
            self.bottomright.y =  instance.bottomright.y
        if(self.id > instance.id):
            self.id = instance.id 
            instance.append(self) 
        else:
            cur = instance
            cur.id = self.id
            while cur.prev:                
                cur = cur.prev
                cur.id = self.id
            
      
    def distance(self, instance):
        if(instance.bottomright.x < self.topleft.x):
            return self.topleft.x - instance.bottomright.x
        elif(instance.topleft.x > self.bottomright.x):
            return instance.topleft.x  - self.bottomright.x
        return 0     

    def HaveCommon(self, instance, dx: int):
        #if ((self.topleft.x - dx/5) > instance.bottomright.x or self.bottomright.x < (instance.topleft.x - dx/5)):
        if(self.distance(instance) > dx/5):
            return False
        return True
    
def CombineGroups(currents: list[RowGroup], pervs: list[RowGroup], dx: int, dy: int):
    for perv in pervs:
        for current in currents:
            current.Devour(perv, dx, dy)
            
def MakeRowGroups(rows: list[list[Point]], dx: int, dy: int):
    currentGroups: list[RowGroup] = []
    pervGroups: list[RowGroup] = []
    RowGroups: list[RowGroup] = []
    currentId = 0
    for row in rows:
        _row = sorted(row, key= attrgetter("x"))
        starting_ind = 0
        currentId +=1
        for i in range(1, len(_row)):
            delta = math.fabs(_row[i].x - _row[i-1].x)
            if(delta > 1.1 * dx):
                currentGroups.append(RowGroup(currentId, [x for x in _row[starting_ind:i]]))
                starting_ind = i
                currentId+=1
        currentGroups.append(RowGroup(currentId, [x for x in _row[starting_ind:]]))
        CombineGroups(currentGroups, pervGroups, dx, dy)
        pervGroups = currentGroups
        currentGroups = []
        RowGroups.extend(pervGroups)
    return RowGroups
    
def MakeBlocks(rows: list[list[Point]], dx: int, dy: int):
    Blocks : list[RowGroup] = []
    RowGroups = MakeRowGroups(rows, dx, dy)
    #ids = distinct_by([group for group in RowGroups], key = lambda p: p.id)
    ids = list(set([group.id for group in RowGroups]))    
    for i in range(len(ids)):
        groups = [g for g in RowGroups if g.id == ids[i]]
        points = [p
                  for group in groups
                  for p in group.points]
        rg = RowGroup(i+1, points)
        rg.topleft = groups[-1].topleft
        rg.bottomright = groups[-1].bottomright
        Blocks.append(rg)
    return Blocks

def FindBlockMissingPoints(block: RowGroup, dx:int, dy:int):
    x_tl, y_tl = block.topleft
    x_br, y_br = block.bottomright
    points = block.points
    
    # Calculate Grid Dimensions
    C = round((x_br - x_tl) / dx) + 1
    R = round(abs(y_br - y_tl) / dy) + 1
    y_dir = 1 if y_br >= y_tl else -1
    
    # Initialize and Map Existing Points
    grid = [[None for _ in range(C)] for _ in range(R)]
    points = sorted(points, key= lambda p: p.y  )
    for x, y in points:
        c = round((x - x_tl) / dx)
        r = round(abs(y - y_tl) / dy)
        if 0 <= c < C and 0 <= r < R:
            grid[r][c] = (x, y)
            
    missing_points = []
    
    for r in range(R):
        for c in range(C):
            if grid[r][c] is None:
                
                xs = []
                for i in range(r - 1, -1, -1): # Search Up
                    if grid[i][c] is not None:
                        xs.append(grid[i][c][0])
                        break
                for i in range(r + 1, R):      # Search Down
                    if grid[i][c] is not None:
                        xs.append(grid[i][c][0])
                        break
                        
                calc_x = sum(xs) / len(xs) if xs else x_tl + (c * dx) # Fallback to theoretical X
                
                ys = []
                for j in range(c - 1, -1, -1): # Search Left
                    if grid[r][j] is not None:
                        ys.append(grid[r][j][1])
                        break
                for j in range(c + 1, C):      # Search Right
                    if grid[r][j] is not None:
                        ys.append(grid[r][j][1])
                        break
                        
                calc_y = sum(ys) / len(ys) if ys else y_tl + (y_dir * r * dy) # Fallback to theoretical Y
                
                grid[r][c] = (calc_x, calc_y)
                missing_points.append(Point(calc_x, calc_y))
                #points.append(Point(calc_x, calc_y))
                
    return missing_points

def AddMissingPoints(points: list[Point], dx: int, dy: int):
    missings: list[Point] = []
    rows = Cluster_to_Rows(points, dx)
    for row in rows:
        _row = sorted(row, key=lambda x : x.x )
        for i in range(1, len(_row)):
            delta = math.fabs(_row[i].x - _row[i-1].x)
            diff = delta/dx
            floor = round(diff)
            floating = math.fabs(diff - floor)
            if(floating > .97 or floating < .03):
                for f in range(int(floor) - 1):
                    missings.append(Point(_row[i-1].x + (f+1) * dx, (_row[i].y + _row[i-1].y)/ 2))
    columns = Cluster_to_Columns(points, dy)
    for column in columns:
        _column = sorted(column, key=lambda x : x.y )
        for i in range(1, len(_column)):
            delta = math.fabs(_column[i].y - _column[i-1].y)
            diff = delta/dy
            floor = round(diff)
            floating = math.fabs(diff - floor)
            if(floating > .95 or floating < .05):
                for f in range(int(floor) - 1):
                    missings.append(Point((_column[i].x + _column[i-1].x)/ 2, _column[i-1].y + (f+1) * dy))
    missings = distinct_by(missings, key= lambda p: p, cond = lambda x,y : x.distance_to(y) <= dx/5)
    return missings

def Cluster_to_Rows(points: list[Point], dx: int):
    rows=[]
    sorted_points = sorted(points, key= attrgetter("y") )
    current_group = [sorted_points[0]]
    for pt in sorted_points[1:]:
        if(math.fabs(pt.y - current_group[-1].y) < dx/2) :
            current_group.append(pt)
        else:
            rows.append(current_group)
            current_group = [pt]
    sorted_group = sorted(current_group, key= attrgetter("x") )
    rows.append(sorted_group)    
    return rows

def Cluster_to_Columns(points: list[Point], dy: int): 
    columns=[]
    sorted_points = sorted(points, key= attrgetter("x"))
    current_group = [sorted_points[0]]
    for pt in sorted_points[1:]:
        if(math.fabs(pt.x -  current_group[-1].x) < dy/2) :
            current_group.append(pt)
        else:
            columns.append(current_group)
            current_group = [pt]
    columns.append(current_group)
    return columns
    
def distinct_by(seq, key):
    seen = set()
    out = []
    for item in seq:
        k = key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out
def distinct_by(seq, key, cond):
    out = []
    for item in seq:
        k = key(item)
        if not any(cond(k, other) for other in out):
            out.append(k)
    return out