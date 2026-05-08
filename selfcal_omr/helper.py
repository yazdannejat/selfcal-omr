import cv2
import numpy as np

from selfcal_omr.constants import Colors, RenderSize

def Make_Normalized_Sheet(blocks, points, missing, img):
    h, w, _ = img.shape
    img = np.zeros((h,w,3), np.uint8)
    for block in blocks:
        cv2.rectangle(img, block.topleft.to_cv(), block.bottomright.to_cv(), Colors.green, 2)
    for x,y in missing:
        cv2.circle(img,(int(x), int(y)), 10, Colors.yellow, -1)
    for point in points:
        cv2.circle(img,(int(point.x),int(point.y)), 5, Colors.red, -1)
    resize = cv2.resize(img, (RenderSize.width, RenderSize.height))
    show_images(['image'], [resize])
    
def draw_contours(img, contours, missing):
    cv2.drawContours(img, contours, -1, Colors.red, 3)
    for p in missing:
        cv2.circle(img, p.to_cv(), 8, (0, 255, 255), -1)
    resize = cv2.resize(img, (RenderSize.width, RenderSize.height))
    show_images(['image'], [resize])
        
def show_images(titles, images, wait=True):
    """Display multiple images with one line of code"""
    for (title, image) in zip(titles, images):
        cv2.imshow(title, image)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Point:
    def __init__(self, x=0, y=0, id=0):
        self.x = x
        self.y = y
        self.id = id
        self.data= np.array([x, y], dtype=np.float32)
        
    def __iter__(self):
        yield self.x
        yield self.y
        
    def to_numpy(self):
        return self.data   
    
    def copy(self):
        return Point(self.x, self.y)   
    
    def to_cv(self):
        return (int(self.x), int(self.y))  
             
    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()
    
def reduce_by_variance(nums, key=lambda x: x):
    nums = list(nums) #shallow copy
    nums_ = [key(num) for num in nums]
    original_size = len(nums)
    
    mean = sum(nums_) / original_size
    #variance = sum((x - mean)**2 for x in nums) / original_size

    target_size = 4 * original_size // 5

    while len(nums) > target_size:
        distances = [(abs(x - mean), i) for i, x in enumerate(nums_)]
        
        _, idx_to_remove = max(distances, key=lambda t: t[0])
        
        nums_.pop(idx_to_remove)
        nums.pop(idx_to_remove)
        
    return nums, mean